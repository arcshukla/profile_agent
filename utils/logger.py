"""
utils/logger.py
---------------
Centralised logging utility for the application.

Usage:
    from utils.logger import get_logger, get_session_logger, new_session_id
    from utils.logger import log_chat, log_unknown, log_email, log_message, push_logs

    log = get_logger(__name__)
    log.info("something happened")

    # Inside chat() — tags every line with [session_id[:8]] for multi-user correlation
    session_log = get_session_logger(log, session_id)
    session_log.info("User query: %s", message)
    # → 2026-03-20 01:40:09.412  INFO  agents.profile.profile  [e051f394] User query: ...

    log_chat("user question", "agent answer")
    log_unknown("unanswered question")
    log_email("user@example.com")
    log_message("some debug message")

    push_logs()  # upload CSV to HuggingFace
"""

import csv
import contextvars
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import MutableMapping, Any

from dotenv import load_dotenv

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants  —  read from .env with sensible defaults
# ---------------------------------------------------------------------------
LOG_FILE   = os.getenv("LOG_FILE",  "logs/chat_logs.csv")
LOG_LEVEL  = os.getenv("LOG_LEVEL", logging.INFO)
HF_DATASET = os.getenv("HF_LOG_DATASET", "")
SESSION_ID = "system"

CSV_HEADERS = ["timestamp", "session_id", "event_type", "question", "answer", "email", "message"]

# context var for current session id (Gradio session_hash)
current_session_id = contextvars.ContextVar("current_session_id", default=SESSION_ID)

# ---------------------------------------------------------------------------
# Python logger setup
# ---------------------------------------------------------------------------
class _SessionIdFilter(logging.Filter):
    """Attach session_id from contextvar to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = get_current_session_id()
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-level logger with consistent formatting.

    Format: 2026-03-20 01:40:09.412  INFO  [e051f394] utils.rag:_build_context:123  Message here
             ^--- date/time + ms ---^ ^session^ ^level^ ^module:function:line^

    Milliseconds are added via %(msecs)03d combined with the datefmt.
    LOG_LEVEL is read from .env (default INFO).

    Call once at the top of each file:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(_SessionIdFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d  %(levelname)-8s [%(session_id)s] %(name)s:%(funcName)s:%(lineno)d  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Session-scoped logger adapter
# ---------------------------------------------------------------------------

class _SessionAdapter(logging.LoggerAdapter):
    """
    Wraps a logger and prepends [session_id] to every message so you can
    correlate all log lines for one user when multiple visitors are active.

    With two concurrent users the log will look like:
        01:40:09.412  INFO  profile  [e051f39412345678] User query: how to contact archana
        01:40:09.615  INFO  profile  [a3b7c29187654321] User query: what are her skills
        01:40:10.001  INFO  profile  [e051f39412345678] RAG → 1 chunk(s) | topics: ['contact']
        01:40:10.204  INFO  profile  [a3b7c29187654321] RAG → 2 chunk(s) | topics: ['skills']

    grep "[e051f39412345678]" to follow a single user's entire session.
    """

    def process(self, msg: str, kwargs: MutableMapping[str, Any]):
        sid      = self.extra.get("session_id", "--------")
        return f"[{sid}] {msg}", kwargs


def set_current_session_id(session_id: str) -> None:
    """Set session id for current context (used by log_event, wrappers, LLM calls)."""
    current_session_id.set(session_id)


def get_current_session_id() -> str:
    """Get effective session id for current context.

    Returns:
        - real Gradio session_hash if set
        - "system" if not set (for non-user actions / startup)
    """
    return current_session_id.get() or "system"


def get_session_logger(base_logger: logging.Logger, session_id: str) -> _SessionAdapter:
    """
    Returns a LoggerAdapter that tags every line with [session_id].

    Call once per chat() invocation and use the returned logger for all
    turn-scoped log lines:

        sid  = request.session_hash or new_session_id()
        slog = get_session_logger(logger, sid)
        slog.info("User query: %s", message)
    """
    set_current_session_id(session_id)
    return _SessionAdapter(base_logger, {"session_id": session_id})


def new_session_id() -> str:
    """Generate a fresh UUID — fallback when Gradio session hash is unavailable."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# CSV chat log
# ---------------------------------------------------------------------------
def _ensure_log_file() -> None:
    """Create the CSV log file with headers if it doesn't exist."""
    path = Path(LOG_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADERS)


def log_event(
    event_type: str,
    question:   str = "",
    answer:     str = "",
    email:      str = "",
    message:    str = "",
    session_id: str = "",
) -> None:
    """Write a single event row to the CSV log.
    
    Args:
        event_type: Type of event (chat, unknown_question, email_capture, debug)
        question: User question
        answer: Assistant answer
        email: User email if captured
        message: Debug message
        session_id: User's session ID from Gradio (request.session_hash).
                   Fallback to global SESSION_ID if not provided.
    """
    try:
        _ensure_log_file()
        uid = session_id or get_current_session_id()
        if not uid:
            uid = "system"

        row = [
            datetime.utcnow().isoformat(),
            uid,
            event_type,
            question,
            answer,
            email,
            message,
        ]
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        _log.info("Event [%s]: %s", uid, event_type)

    except Exception as e:
        _log.error("Failed to write log event: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Convenience wrappers  —  now accept session_id for per-user tracking
# ---------------------------------------------------------------------------
def log_chat(question: str, answer: str, session_id: str = "") -> None:
    """Log a chat exchange to CSV.
    
    Args:
        question: User question
        answer: Assistant answer
        session_id: User's Gradio session hash (from request.session_hash)
    """
    log_event("chat", question=question, answer=answer, session_id=session_id)

def log_unknown(question: str, session_id: str = "") -> None:
    """Log an unanswered question to CSV.
    
    Args:
        question: User question that couldn't be answered
        session_id: User's Gradio session hash
    """
    log_event("unknown_question", question=question, session_id=session_id)

def log_email(email: str, session_id: str = "") -> None:
    """Log a captured email to CSV.
    
    Args:
        email: User's email address
        session_id: User's Gradio session hash
    """
    log_event("email_capture", email=email, session_id=session_id)

def log_message(message: str, session_id: str = "") -> None:
    """Log a debug message to CSV.
    
    Args:
        message: Debug message
        session_id: User's Gradio session hash
    """
    log_event("debug", message=message, session_id=session_id)


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------
def push_logs() -> bool:
    """
    Upload the CSV log to a HuggingFace dataset repo.
    Set HF_LOG_DATASET in .env to enable.
    Returns True on success, False on failure.
    """
    if not HF_DATASET:
        _log.warning("HF_LOG_DATASET not set — skipping push.")
        return False

    if not os.path.exists(LOG_FILE):
        _log.warning("Log file '%s' not found — skipping push.", LOG_FILE)
        return False

    if HfApi is None:
        _log.error("HfApi is not available; install huggingface_hub to push logs")
        return False

    try:
        HfApi().upload_file(
            path_or_fileobj=LOG_FILE,
            path_in_repo="chat_logs.csv",
            repo_id=HF_DATASET,
            repo_type="dataset",
        )
        _log.info("Logs pushed to HuggingFace dataset: %s", HF_DATASET)
        return True

    except Exception as e:
        _log.error("Failed to push logs to HuggingFace: %s", e)
        return False