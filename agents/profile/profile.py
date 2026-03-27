"""
profile.py
----------
ProfileAgent — orchestrates RAG retrieval, LLM chat, and followup generation.

No prompt strings live here. All prompts are in prompts.py.
To change wording, tone, or instructions — edit prompts.py only.
"""

import json
import os
import re
import time
from pathlib import Path

from utils.llm_client import LLMClient
from agents.profile.profile_rag import ProfileRAG
from utils.logger import get_logger, get_session_logger, new_session_id, set_current_session_id, log_chat
from services.tools import tools, handle_tool_call
from .prompts import (
    SYSTEM_PROMPT,
    INITIAL_FOLLOWUPS_PROMPT,
    TURN_FOLLOWUPS_PROMPT,
    WELCOME_MESSAGE,
    CHAT_PLACEHOLDER,
    UNKNOWN_PHRASES,
    FALLBACK_FOLLOWUPS,
)

logger = get_logger(__name__)


# ── Startup helpers ───────────────────────────────────────────────────────────

def _load_profile_context() -> dict:
    prefix = "CAREER_PROFILE_"
    name   = os.getenv(prefix + "NAME", "")
    return {
        "name":                name,
        "short_name":          os.getenv(prefix + "SHORT_NAME", name.split()[0] if name else ""),
        "profile_folder":      os.getenv(prefix + "FOLDER", ""),
    }

def _should_ingest_profile_files(profile_folder_path: str) -> tuple[bool, list[Path]]:
    """
    Check if profile files need re-ingestion based on modification times and env vars.
    
    Returns (should_ingest, files_list) where files_list is all files in the folder.
    """
    profile_folder = Path(profile_folder_path)
    cache_minutes = int(os.getenv("PROFILE_CACHE_MINUTES", "20"))
    force_reingest = os.getenv("FORCE_PROFILE_REINGEST", "").lower() in ("true", "1", "yes")
    
    files = list(profile_folder.rglob("*")) if profile_folder.exists() else []
    file_paths = [f for f in files if f.is_file()]
    
    if force_reingest:
        logger.info("FORCE_PROFILE_REINGEST is set — forcing re-ingestion")
        return True, file_paths
    
    if not file_paths:
        return False, file_paths  # No files, no ingestion needed
    
    max_mtime = max(f.stat().st_mtime for f in file_paths)
    time_since_change = time.time() - max_mtime
    should_ingest = time_since_change < cache_minutes * 60
    
    if not should_ingest:
        logger.info("Profile files unchanged in last %d minutes — skipping re-ingestion", cache_minutes)
    else:
        logger.info("Profile files changed recently — re-ingesting")
    
    return should_ingest, file_paths

def _call_llm_for_followups(llm: LLMClient, prompt: str) -> list[str]:

    """
    Shared helper: call LLM, parse JSON array of strings.
    Returns empty list on any failure — callers decide the fallback.
    """
    try:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.6,
        )
        content = response.choices[0].message.content or "[]"
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content).strip()
        parsed  = json.loads(content)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, str) and q.strip()][:3]
    except Exception as e:
        logger.warning("Followup LLM call failed: %s", e, exc_info=True)
    return []


# ── ProfileAgent ──────────────────────────────────────────────────────────────

class ProfileAgent:
    """
    Generic profile agent. To use for a different person:
      1. Update me/ files (linkedin.pdf, summary.txt, recommendations.csv)
      2. Update .env (CAREER_PROFILE_NAME, etc.)
      3. Delete .chromadb_semantic/ to force re-ingestion
      No code or prompt changes needed.
    """

    def __init__(self) -> None:
        ctx = _load_profile_context()

        self.name             = ctx["name"]
        self.short_name       = ctx["short_name"]        
        self.welcome_message  = WELCOME_MESSAGE.format(name=self.short_name)
        self.chat_placeholder = CHAT_PLACEHOLDER

        self.llm = LLMClient()
        self.rag = ProfileRAG()

        # Check if profile files need re-ingestion (optimization)
        should_ingest, files = _should_ingest_profile_files(ctx["profile_folder"])

        if should_ingest:
            # Ingest all profile files — LLM splits each into topic sections       
            for f in files:
                self.rag.ingest(f)
        else:
            logger.info("Using cached profile data")

        logger.info("Topics indexed: %s", self.rag.get_all_topics())

        # Generate opening followups from all indexed topics (holistic view)
        # Done after ingest so content is available
        self.followup_questions = self._generate_initial_followups()
        logger.info("Initial followups: %s", self.followup_questions)

        # System prompt finalised after followups are known
        self.system_instructions = SYSTEM_PROMPT.format(
            name      = self.short_name,
            followups = self.followup_questions,
        )

    # ── Context ───────────────────────────────────────────────────────────────

    def _build_context(self, message: str, slog=None) -> str:
        log     = slog or logger
        chunks  = self.rag.retrieve(message, k=4)
        context = "\n\n".join(chunks)

        all_data  = self.rag.collection.get(include=["documents", "metadatas"])
        doc_topic = {d: m.get("topic", "?") for d, m in zip(all_data["documents"], all_data["metadatas"])}
        chunk_topics = [doc_topic.get(c, "?") for c in chunks]

        log.info("RAG → %d chunk(s) | topics: %s | %d chars", len(chunks), chunk_topics, len(context))
        logger.debug("Context preview:\n%s", context[:500])
        return context

    # ── Followup generation ───────────────────────────────────────────────────

    def _generate_initial_followups(self) -> list[str]:
        """
        Called once at startup. Reads all indexed topics holistically.
        Prompt: INITIAL_FOLLOWUPS_PROMPT in prompts.py.
        """
        snapshot = self.rag.build_snapshot()
        if not snapshot.strip():
            return FALLBACK_FOLLOWUPS

        prompt    = INITIAL_FOLLOWUPS_PROMPT.format(
            name            = self.short_name,
            profile_context = snapshot,
        )
        questions = _call_llm_for_followups(self.llm, prompt)
        return questions if len(questions) == 3 else FALLBACK_FOLLOWUPS

    def _generate_turn_followups(self, question: str, answer: str, was_answered: bool) -> list[str]:
        """
        Called after every chat turn. Suggests questions pointing to unexplored areas.
        was_answered=False tells the prompt to pivot away — not suggest more of the same.
        Prompt: TURN_FOLLOWUPS_PROMPT in prompts.py.
        """
        snapshot  = self.rag.build_snapshot()
        prompt    = TURN_FOLLOWUPS_PROMPT.format(
            name            = self.short_name,
            question        = question,
            answer          = answer[:300],
            was_answered    = "true" if was_answered else "false",
            profile_context = snapshot,
        )
        questions = _call_llm_for_followups(self.llm, prompt)
        return questions if len(questions) == 3 else self.followup_questions

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, message: str, history: list, session_id: str = "") -> tuple[str, list[str]]:
        # Session-scoped logger — every line tagged [sid[:8]] for multi-user correlation
        sid  = session_id or new_session_id()
        set_current_session_id(sid)
        slog = get_session_logger(logger, sid)

        slog.info("─" * 52)
        slog.info("User query: %s", message)

        context_block = self._build_context(message, slog)
        messages = [
            {"role": "system", "content": self.system_instructions},
            {"role": "system", "content": context_block},
            *history[-4:],
            #*[{"role": m["role"], "content": m["content"]} for m in history[-4:]],
        ]
        slog.info("Calling LLM | history turns: %d | context chars: %d", len(history), len(context_block))

        tool_call_count = 0
        while True:
            try:
                response = self.llm.chat(
                    messages        = messages,
                    tools           = tools,
                    response_format = {"type": "json_object"},
                    session_id      = sid,
                )
            except Exception as e:
                slog.error("LLM error: %s", e)
                return self._error_response(str(e))

            choice = response.choices[0]
            if choice.message.tool_calls:
                tool_names = [tc.function.name for tc in choice.message.tool_calls]
                tool_call_count += 1
                slog.info("Tool call #%d: %s", tool_call_count, tool_names)
                tool_results = handle_tool_call(choice.message.tool_calls, session_id=sid)
                messages.append(choice.message)
                messages.extend(tool_results)
                continue
            break

        slog.info("LLM responded | tokens: %s",
                  getattr(getattr(response, "usage", None), "total_tokens", "n/a"))
        return self._parse_reply(message, response.choices[0].message.content or "", slog, sid)

    def _parse_reply(self, message: str, reply: str, slog=None, sid: str = "") -> tuple[str, list[str]]:
        log = slog or logger
        logger.info("Raw LLM reply: %s", reply)
        try:
            data   = json.loads(reply)
            answer = data["answer"]

            was_answered = not any(p in answer.lower() for p in UNKNOWN_PHRASES)
            if not was_answered:
                log.info("Answer NOT informative — followups will pivot to new topics")

            followups = self._generate_turn_followups(message, answer, was_answered)
            log.info("Answer (%d chars) | was_answered=%s | followups: %s",
                     len(answer), was_answered, followups)
            log_chat(message, answer, session_id=sid)
            return answer, followups
        except Exception:
            log.warning("Could not parse LLM reply as JSON — returning raw")
            return reply, self.followup_questions

    def _error_response(self, error: str) -> tuple[str, list[str]]:
        if "quota" in error.lower() or "402" in error:
            msg = "I'm currently experiencing high demand. Please try again shortly."
        else:
            msg = "I'm experiencing an internal issue. Please try again shortly."
        return msg, self.followup_questions