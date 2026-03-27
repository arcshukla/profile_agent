"""
semantic_rag_engine.py
----------------------
Generic LLM-powered semantic RAG engine for small structured documents.

Core pattern:
  INGEST:   LLM splits each file into named topic sections using a prompt
            and topic list you provide. Each section gets a `topic` metadata
            label stored in ChromaDB.

  RETRIEVE: A fast LLM call classifies the user's query using an intent
            prompt you provide, then does a direct metadata filter fetch.
            No ANN, no reranker — pure label lookup.

This class knows nothing about profiles, products, or any other domain.
All domain knowledge (topic labels, prompts) is supplied by the caller.

For medium/large unstructured corpora, use rag_pipeline.py instead.

Usage:
    from utils.semantic_rag_engine import SemanticRAGEngine

    engine = SemanticRAGEngine(
        topic_labels   = ["contact", "experience", "education", "skills"],
        split_prompt   = SPLIT_PROMPT,    # your domain-specific prompt
        intent_prompt  = INTENT_PROMPT,   # your domain-specific prompt
        db_path        = ".chromadb_myagent",
        collection_name= "myagent_docs",
    )
    engine.ingest("docs/profile.pdf")
    chunks = engine.retrieve("how do I contact her")
"""

import hashlib
import json
import re
import chromadb
from chromadb.config import Settings
from pathlib import Path

from utils.file_reader import read_file
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH        = ".chromadb_semantic"
DEFAULT_COLLECTION     = "semantic_docs"


class SemanticRAGEngine:
    """
    Generic LLM-powered semantic RAG engine.

    All domain knowledge is injected at construction time:
      topic_labels  — the allowed section labels for this domain
      split_prompt  — LLM prompt template for splitting a document into sections
                      Must contain {topic_labels}, {source_name}, {text}
      intent_prompt — LLM prompt template for classifying a query into topic labels
                      Must contain {topic_labels}, {query}

    The engine stores nothing domain-specific internally. Swap the three
    constructor args and you have a completely different agent.
    """

    def __init__(
        self,
        topic_labels:    list[str],
        split_prompt:    str,
        intent_prompt:   str,
        db_path:         str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        if not topic_labels:
            raise ValueError("topic_labels must not be empty")

        self.topic_labels  = topic_labels
        self.split_prompt  = split_prompt
        self.intent_prompt = intent_prompt
        self.llm           = LLMClient()

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=True),
        )
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "SemanticRAGEngine ready | Path '%s' | collection '%s' | %d existing chunks | topics: %s",
            db_path, collection_name, self.collection.count(), topic_labels,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, source: str | Path) -> int:
        """
        Read a file, split it into labelled topic sections via LLM, store in ChromaDB.
        Idempotent — skips chunks already indexed.
        Returns number of new chunks added.
        """
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Not found: {path}")

        logger.info("Ingesting: %s", path)
        try:
            raw_text = read_file(path)
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e, exc_info=True)
            return 0

        logger.debug("Splitting document: %d chars", len(raw_text))
        sections = self._split_into_sections(raw_text, source_name=str(path))
        if not sections:
            logger.warning("No sections extracted from %s", path)
            return 0

        added = 0
        for section in sections:
            uid      = hashlib.sha256(section["text"].encode()).hexdigest()[:32]
            existing = self.collection.get(ids=[uid])["ids"]
            if existing:
                continue
            try:
                self.collection.add(
                    ids       = [uid],
                    documents = [section["text"]],
                    metadatas = [{
                        "source": str(path),
                        "topic":  section["topic"],
                    }],
                    embeddings = [[0.0]],   # not used — retrieval is by metadata filter
                )
                added += 1
                logger.debug("  + [%s] %s...", section["topic"], section["text"][:60])
            except Exception as e:
                logger.warning("Failed to store section: %s", e, exc_info=True)

        logger.info("  Added %d / %d sections from %s", added, len(sections), path)
        return added

    def _split_into_sections(self, text: str, source_name: str) -> list[dict]:
        """
        Call LLM with the caller-supplied split_prompt to divide the document
        into labelled sections. Returns [{"topic": str, "text": str}, ...].
        """
        topic_labels_str = ", ".join(self.topic_labels)
        prompt = self.split_prompt.format(
            topic_labels = topic_labels_str,
            source_name  = source_name,
            text         = text,
        )

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens = 4096,
                temperature = 0.1,
            )
            content = response.choices[0].message.content or "[]"
            logger.debug("Section split response: %d chars. Actual contents: %s", len(content), content)

            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content).strip()

            try:
                sections = json.loads(content)
            except json.JSONDecodeError as e:
                # Partial recovery if response was truncated at max_tokens
                logger.warning("JSON decode failed: %s — attempting partial recovery", e, exc_info=True)
                # Try to fix unterminated strings
                partial = content
                # Find unterminated strings and close them
                # Simple fix: find the last " and if not followed by , or }, add "
                # But it's tricky. For now, just try to close at the end.
                last_quote = partial.rfind('"')
                if last_quote > 0 and not partial[last_quote+1:].strip().startswith((',', '}', ']')):
                    partial = partial[:last_quote+1] + '"}'  # assume it's the last in object
                last_close = partial.rfind("}")
                if last_close > 0:
                    partial = partial[:last_close + 1]
                    if not partial.rstrip().endswith("]"):
                        partial += "]"
                    try:
                        sections = json.loads(partial)
                        logger.info("Partial recovery with string fix: %d objects", len(sections))
                    except json.JSONDecodeError:
                        raise e  # re-raise original
                else:
                    raise

            if not isinstance(sections, list):
                raise ValueError(f"Expected list, got {type(sections)}")

            valid = []
            for s in sections:
                topic = s.get("topic", "other").lower().strip()
                text  = s.get("text", "").strip()
                if not text:
                    continue
                if topic not in self.topic_labels:
                    logger.warning("Unknown topic '%s' — remapping to last label", topic)
                    topic = self.topic_labels[-1]   # last label is the catch-all by convention
                valid.append({"topic": topic, "text": text})

            logger.info("  Split into %d sections: %s", len(valid), [s["topic"] for s in valid])
            return valid

        except Exception as e:
            logger.error("Section splitting failed: %s", e, exc_info=True)
            return [{"topic": self.topic_labels[-1], "text": text}]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 4) -> list[str]:
        """
        Classify query intent → fetch matching topic sections.
        No embeddings, no ANN, no reranker.
        """
        if self.collection.count() == 0:
            logger.warning("Collection is empty — call ingest() first.")
            return []

        topics = self._classify_intent(query)
        logger.info("Query '%s' → topics: %s", query[:60], topics)

        # ChromaDB's $in operator requires at least 2 values.
        # Use $eq for single-topic queries to avoid a silent filter failure.
        if len(topics) == 1:
            where = {"topic": {"$eq": topics[0]}}
        else:
            where = {"topic": {"$in": topics}}

        results = self.collection.get(
            where   = where,
            include = ["documents", "metadatas"],
        )

        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])

        # Warn about topics requested but absent from the index
        indexed_topics = {m.get("topic") for m in metas}
        missing = [t for t in topics if t not in indexed_topics]
        if missing:
            logger.warning("Topics not in index: %s", missing)

        if not docs:
            return []

        # Order by topic priority (same order as topics list), deduplicated
        topic_order = {t: i for i, t in enumerate(topics)}
        pairs = sorted(zip(docs, metas), key=lambda x: topic_order.get(x[1].get("topic", ""), 99))

        seen, ordered = set(), []
        for doc, _ in pairs:
            h = hashlib.md5(doc.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                ordered.append(doc)

        logger.debug("Retrieved %d chunks for topics %s", len(ordered), topics)
        return ordered[:k]

    def _classify_intent(self, query: str) -> list[str]:
        """
        Call LLM with the caller-supplied intent_prompt to map query → topic labels.
        Falls back to all non-catch-all topics on failure.
        """
        topic_labels_str = ", ".join(self.topic_labels)
        prompt = self.intent_prompt.format(
            topic_labels = topic_labels_str,
            query        = query,
        )

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens  = 60,
                temperature = 0.0,
            )
            content = response.choices[0].message.content or "[]"
            content = re.sub(r"^```(?:json)?\s*", "", content.strip())
            content = re.sub(r"\s*```$", "", content).strip()

            parsed = json.loads(content)
            if isinstance(parsed, list):
                valid = [t for t in parsed if t in self.topic_labels]
                if valid:
                    return valid

        except Exception as e:
            logger.warning("Intent classification failed: %s", e, exc_info=True)

        # Fallback: all labels except the catch-all (last by convention)
        return self.topic_labels[:-1]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_all_topics(self) -> dict[str, int]:
        """Count of indexed chunks per topic."""
        counts: dict[str, int] = {}
        for meta in self.collection.get(include=["metadatas"]).get("metadatas", []):
            t = meta.get("topic", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def get_by_topic(self, topic: str) -> list[str]:
        """Fetch all chunks for a given topic label."""
        results = self.collection.get(
            where   = {"topic": topic},
            include = ["documents"],
        )
        return results.get("documents", [])

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info("Collection cleared.")