"""
profile_rag.py
--------------
Profile-specific semantic RAG — extends SemanticRAGEngine with:
  - PROFILE_TOPICS: the 8-label taxonomy for professional profiles
  - SPLIT_PROMPT:   how to split a CV/LinkedIn/recommendations file
  - INTENT_PROMPT:  how to route profile questions to topic labels

This is the ONLY file that needs to change if you want to tweak how
profile documents are split or how queries are routed. The engine
machinery in semantic_rag_engine.py stays untouched.

For a different agent (product docs, legal contracts, etc.) create a
sibling file (e.g. product_rag.py) with its own topics + prompts.
"""

from utils.semantic_rag_engine import SemanticRAGEngine
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Profile topic taxonomy ────────────────────────────────────────────────────
# Intentionally broad so they cover any professional profile format.
# "other" is last by convention — SemanticRAGEngine uses it as the catch-all.

PROFILE_TOPICS = [
    "contact",          # name, email, phone, LinkedIn URL, location
    "summary",          # executive summary, headline, career overview
    "experience",       # work history, companies, roles, dates, achievements
    "education",        # degrees, universities, years
    "skills",           # technical stack, tools, languages, platforms, cloud
    "awards",           # patents, certifications, awards, publications
    "recommendations",  # testimonials and endorsements from other people
    "other",            # catch-all — anything that doesn't fit above
]

# ── Prompts ───────────────────────────────────────────────────────────────────
# These are the two prompt templates SemanticRAGEngine needs.
# {topic_labels}, {source_name}, {text} are filled in by the engine.

PROFILE_SPLIT_PROMPT = """You are processing a professional profile document for a semantic search index.

Split the document into logical sections. For each section assign exactly one topic label
from this list: {topic_labels}

Rules:
- Split at natural content boundaries (don't mid-sentence split)
- Each section should be self-contained and answerable as a unit
- contact: name, title, email, phone, LinkedIn, location
- summary: executive summary, career overview, objective statement
- experience: all work history (can be one section or split by employer)
- education: all degrees, universities, years
- skills: technology stack, tools, languages, platforms, cloud, frameworks
- awards: patents, certifications, awards, publications, recognitions
- recommendations: testimonials or endorsements written by other people
- other: anything else

Return ONLY a JSON array wrapped in ```json code blocks. Each element: {{"topic": "<label>", "text": "<full section text>"}}
No markdown, no explanation, just the array.

Document ({source_name}):
{text}

JSON array:"""

PROFILE_INTENT_PROMPT = """You classify user questions about a professional profile into topic categories.

Available topics: {topic_labels}

Return ONLY a JSON array wrapped in ```json code blocks of 1-3 topic labels that best match what the user is asking.
Examples:
- "how do I contact her" → ["contact"]
- "what is her education" → ["education"]
- "which companies did she work at" → ["experience"]
- "what are her technical skills" → ["skills"]
- "tell me about her background" → ["summary", "experience"]
- "what did her colleagues say" → ["recommendations"]
- "any patents or awards" → ["awards"]
- "tell me about her" → ["summary", "experience", "skills"]

Question: {query}

JSON array:"""


# ── ProfileRAG ────────────────────────────────────────────────────────────────

class ProfileRAG(SemanticRAGEngine):
    """
    Semantic RAG configured for professional profiles (CV, LinkedIn, recommendations).

    Drop-in replacement for SemanticRAGPipeline — same public API:
      ingest(path), retrieve(query, k), get_by_topic(topic),
      get_all_topics(), count(), clear()

    To use for a different profile: just swap the me/ files and .env vars.
    To create a different agent type: subclass SemanticRAGEngine with
    your own topics + prompts (see product_rag.py example below).
    """

    def __init__(
        self,
        db_path:         str = ".chromadb_profile",
        collection_name: str = "profile_docs",
    ) -> None:
        super().__init__(
            topic_labels    = PROFILE_TOPICS,
            split_prompt    = PROFILE_SPLIT_PROMPT,
            intent_prompt   = PROFILE_INTENT_PROMPT,
            db_path         = db_path,
            collection_name = collection_name,
        )

    def build_snapshot(self, topics: list[str] | None = None) -> str:
        """
        Build a compact holistic view of the profile — one chunk per topic.
        Used by profile.py for followup question generation.

        topics: which topics to include (default: all except recommendations + other)
        """
        include = topics or ["contact", "summary", "experience", "education", "skills", "awards"]
        parts   = []
        for topic in include:
            chunks = self.get_by_topic(topic)
            if chunks:
                parts.append(f"[{topic.upper()}]\n{chunks[0][:400]}")
        return "\n\n".join(parts)


# ── Example: how you'd create a product-docs agent (not implemented here) ─────
#
# from utils.semantic_rag_engine import SemanticRAGEngine
#
# PRODUCT_TOPICS = ["overview", "features", "pricing", "integrations", "faqs", "other"]
#
# PRODUCT_SPLIT_PROMPT = """..."""
# PRODUCT_INTENT_PROMPT = """..."""
#
# class ProductRAG(SemanticRAGEngine):
#     def __init__(self):
#         super().__init__(
#             topic_labels    = PRODUCT_TOPICS,
#             split_prompt    = PRODUCT_SPLIT_PROMPT,
#             intent_prompt   = PRODUCT_INTENT_PROMPT,
#             db_path         = ".chromadb_product",
#             collection_name = "product_docs",
#         )