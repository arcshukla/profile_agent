"""
rag_pipeline.py
---------------
Generic RAG pipeline: ingest, chunk, embed, retrieve for PDF / TXT / MD / CSV files.

Dependencies:
    pip install chromadb sentence-transformers pymupdf

Usage:
    from utils.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.ingest("docs/")
    contexts = rag.retrieve_reranked("What is X?", k=3)
"""
import hashlib
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

from utils.file_reader import read_file
from utils.logger import get_logger
# Add to imports
from utils.llm_client import LLMClient

logger = get_logger(__name__)

DEFAULT_EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_DB_PATH       = ".chromadb"
DEFAULT_COLLECTION    = "rag_docs"
DEFAULT_CHUNK_SIZE    = 1024
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_HYDE_QUESTIONS = 5  # questions generated per chunk





def _make_splitter(chunk_size: int, chunk_overlap: int):
    class SimpleSplitter:
        def __init__(self, size, overlap):
            self.size = size
            self.overlap = overlap

        def split_text(self, text: str) -> list[str]:
            chunks, start = [], 0
            while start < len(text):
                chunks.append(text[start : start + self.size])
                start += self.size - self.overlap
            return [c for c in chunks if c.strip()]

    return SimpleSplitter(chunk_size, chunk_overlap)


def _chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class RAGPipeline:

    def __init__(
        self,
        embed_model:     str = DEFAULT_EMBED_MODEL,
        rerank_model:    str = DEFAULT_RERANK_MODEL,
        db_path:         str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        chunk_size:      int = DEFAULT_CHUNK_SIZE,
        chunk_overlap:   int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        logger.info("Loading embedding model: %s", embed_model)
        self.model    = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(rerank_model)
        self.splitter = _make_splitter(chunk_size, chunk_overlap)
        self.llm      = LLMClient()   

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready at '%s' | collection '%s' | %d existing chunks",
            db_path, collection_name, self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | Path | list[str | Path],
        *,
        batch_size: int  = 64,
        use_hyde:   bool = False,
    ) -> int:
        sources = source if isinstance(source, list) else [source]
        files: list[Path] = []

        for s in sources:
            s = Path(s)
            if s.is_dir():
                for ext in ("*.pdf", "*.txt", "*.md", "*.markdown", "*.csv"):
                    files.extend(s.rglob(ext))
            elif s.is_file():
                files.append(s)
            else:
                raise FileNotFoundError(f"Not found: {s}")

        total = sum(self._ingest_file(f, batch_size=batch_size, use_hyde=use_hyde) for f in files)
        logger.info("Ingestion complete. Total new chunks added: %d", total)
        return total

    def _generate_hyde_questions(self, chunk: str) -> list[str]:
        """
        Ask LLM to generate hypothetical questions this chunk would answer.
        These are stored as additional embeddings pointing to the same chunk.
        """
        prompt = f"""You are indexing a document chunk for semantic search.
            Generate exactly {DEFAULT_HYDE_QUESTIONS} short questions that this chunk would answer.
            Return ONLY a JSON array of strings. No explanation, no markdown.

            Chunk:
            {chunk[:800]}

            Example output:
            ["What is her email address?", "How can I contact her?", ...]"""

        try:
            response = self.llm.chat([
                {"role": "user", "content": prompt}
            ], response_format={"type": "json_object"})        
        
            content = response.choices[0].message.content or "[]"
            
            # LLM sometimes returns {"questions": [...]} or just [...]
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed[:DEFAULT_HYDE_QUESTIONS]
            if isinstance(parsed, dict):
                # grab first list value found
                for v in parsed.values():
                    if isinstance(v, list):
                        return v[:DEFAULT_HYDE_QUESTIONS]
            return []
        except Exception as e:
            logger.warning("HyDE generation failed for chunk: %s", e, exc_info=True)
            return []


    def _ingest_file(self, path: Path, *, batch_size: int, use_hyde: bool = False) -> int:
        logger.info("Processing: %s", path)
        try:
            raw_text = read_file(path)
        except Exception as exc:
            logger.warning("Skipping %s — %s", path, exc, exc_info=True)
            return 0

        chunks = self.splitter.split_text(raw_text)
        if not chunks:
            logger.warning("No text extracted from %s", path)
            return 0

        ids       = [_chunk_id(c) for c in chunks]
        metadatas = [{"source": str(path), "chunk_index": i} for i, _ in enumerate(chunks)]
        existing  = set(self.collection.get(ids=ids)["ids"])

        new_chunks    = [c for c, uid in zip(chunks,    ids) if uid not in existing]
        new_ids       = [uid for uid in ids       if uid not in existing]
        new_metadatas = [m for m, uid in zip(metadatas, ids) if uid not in existing]

        if not new_chunks:
            logger.info("  All %d chunks already indexed — skipping.", len(chunks))
            return 0

        # Embed and store the actual chunks
        for start in range(0, len(new_chunks), batch_size):
            batch_texts = new_chunks[start : start + batch_size]
            self.collection.add(
                ids        = new_ids[start : start + batch_size],
                documents  = batch_texts,
                embeddings = self.model.encode(batch_texts, show_progress_bar=False).tolist(),
                metadatas  = new_metadatas[start : start + batch_size],
            )

        # HyDE — generate questions and store as additional embeddings
        if use_hyde:
            logger.info("  Generating HyDE questions for %d chunks...", len(new_chunks))
            hyde_added = 0
            for chunk, meta in zip(new_chunks, new_metadatas):
                questions = self._generate_hyde_questions(chunk)
                logger.info(" Questions: " + str(questions))
                for i, question in enumerate(questions):
                    hyde_id = _chunk_id(f"hyde_{question}")
                    # Skip if already exists
                    if hyde_id in set(self.collection.get(ids=[hyde_id])["ids"]):
                        continue
                    self.collection.add(
                        ids        = [hyde_id],
                        documents  = [chunk],          # ← stores original chunk, not question
                        embeddings = self.model.encode([question], show_progress_bar=False).tolist(),
                        metadatas  = [{
                            **meta,
                            "hyde": True,
                            "hyde_question": question,
                        }],
                    )
                    hyde_added += 1
            logger.info("  Added %d HyDE question embeddings.", hyde_added)

        logger.info("  Added %d / %d chunks.", len(new_chunks), len(chunks))
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list:
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.model.encode([prefixed], show_progress_bar=False).tolist()

    def retrieve(self, query: str, k: int = 3, *, where: dict | None = None) -> list[str]:
        """Basic embedding retrieval. Use retrieve_reranked for better accuracy."""
        if self.collection.count() == 0:
            logger.warning("Collection is empty — call ingest() first.")
            return []

        kwargs = dict(
            query_embeddings = self._embed_query(query),
            n_results        = min(k, self.collection.count()),
            include          = ["documents"],
        )
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results["documents"][0] if results["documents"] else []

    def retrieve_reranked(
        self,
        query:            str,
        k:                int = 3,
        *,
        fetch_per_source: int = 3,
    ) -> list[str]:
        """
        Fetch top chunks from EACH source separately, then rerank together.
        Prevents large sources from dominating results.
        """
        if self.collection.count() == 0:
            logger.warning("Collection is empty — call ingest() first.")
            return []

        sources    = list({m["source"] for m in self.collection.get(include=["metadatas"])["metadatas"]})
        q_emb      = self._embed_query(query)
        candidates = []

        for source in sources:
            try:
                results = self.collection.query(
                    query_embeddings = q_emb,
                    n_results        = min(fetch_per_source, self.collection.count()),
                    include          = ["documents"],
                    where            = {"source": source},
                )
                candidates.extend(results["documents"][0])
            except Exception as e:
                logger.warning("Skipping source %s: %s", source, e, exc_info=True)

        if not candidates:
            return []

        scores = self.reranker.predict([(query, doc) for doc in candidates])
        ranked = sorted(zip(scores, candidates), reverse=True)
        logger.debug("Reranked top-%d: %s", k, [(round(float(s), 3), d[:60]) for s, d in ranked[:k]])
        return [doc for _, doc in ranked[:k]]

    def retrieve_with_metadata(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve with source/score metadata. Useful for debugging."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings = self._embed_query(query),
            n_results        = min(k, self.collection.count()),
            include          = ["documents", "metadatas", "distances"],
        )
        return [
            {
                "text":   text,
                "source": meta.get("source", "unknown"),
                "chunk":  meta.get("chunk_index", -1),
                "score":  round(1 - dist, 4),
            }
            for text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ------------------------------------------------------------------
    # Generic filter utility — callers decide what to pin/fetch
    # ------------------------------------------------------------------

    def get_chunks_by_filter(
        self,
        *,
        chunk_index:     int | None = None,
        source_contains: str        = "",
    ) -> list[str]:
        results = self.collection.get(include=["documents", "metadatas"])
        matched = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            if source_contains and source_contains.lower() not in meta.get("source", "").lower():
                continue
            if chunk_index is not None and meta.get("chunk_index") != chunk_index:
                continue
            matched.append(doc)
        return matched

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info("Collection cleared.")
