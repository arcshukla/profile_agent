# llm_client
# Full call (agent)
#llm.chat(messages, tools=tools, response_format={"type": "json_object"})

# HyDE call (no tools, no format)
#llm.chat(messages)

# JSON only, no tools
#llm.chat(messages, response_format={"type": "json_object"})

# Custom tokens
#llm.chat(messages, max_tokens=1000, temperature=0.7)

import os
from openai import OpenAI
from services.notifier import Notifier
from utils.logger import get_logger, set_current_session_id, get_current_session_id

logger = get_logger(__name__)

GROQ_BASE_URL = "api.groq.com"

class LLMClient:

    def __init__(self):
        base_url = os.getenv("OPENROUTER_BASE_URL", "")
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = os.getenv("AI_MODEL")
        self._is_groq = GROQ_BASE_URL in base_url
        if self._is_groq:
            logger.info("LLMClient: Groq backend detected — compatibility mode enabled")

    def chat(
        self,
        messages:        list,
        tools:           list | None = None,
        response_format: dict | None = None,
        max_tokens:      int         = 400,
        temperature:     float       = 0.2,
        session_id:      str         = "",
    ):
        # Propagate session_id via context variable for logger and downstream calls
        if session_id:
            set_current_session_id(session_id)
        logger.debug("Calling LLM with message %s | session_id=%s", str(messages), get_current_session_id())

        params = {
            "model":       self.model,
            "messages":    self._clean_messages(messages),
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        if tools:
            params["tools"] = tools
            # Groq does not support response_format + tools together.
            # OpenAI and Gemini support both — include for those providers.
            if response_format and not self._is_groq:
                params["response_format"] = response_format
            elif response_format and self._is_groq:
                # Groq workaround: inject a system message to enforce JSON output
                # instead of using response_format param (which Groq rejects with tools)
                params["messages"] = self._inject_json_instruction(params["messages"])
        elif response_format:
            # No tools — safe to include response_format for all providers
            params["response_format"] = response_format
        
        response = None
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            # Check for 402/429 errors (payment required / rate limit)
            status_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
            if status_code in (402, 429):
                error_msg = f"LLM API error {status_code}: {str(e)}"
                logger.error(error_msg)
                notifier = Notifier()
                notifier.notify_error(f"LLM API {status_code}", error_msg, session_id=session_id)
            raise

        logger.debug("LLM responded")
        return response

    def _inject_json_instruction(self, messages: list) -> list:
        """
        Groq workaround: when tools + response_format can't be used together,
        append a system message that strongly instructs the model to reply in JSON.
        Placed just before the last user message for maximum effect.
        """
        JSON_INSTRUCTION = {
            "role": "system",
            "content": (
                "IMPORTANT: Your final response (after any tool calls) "
                "MUST be valid JSON only. No prose, no markdown, no explanation. "
                "Return only the raw JSON object as instructed."
            )
        }
        # Insert before the last user message so it's fresh in context
        msgs = list(messages)
        for i in reversed(range(len(msgs))):
            if isinstance(msgs[i], dict) and msgs[i].get("role") == "user":
                msgs.insert(i, JSON_INSTRUCTION)
                break
        else:
            msgs.append(JSON_INSTRUCTION)
        return msgs

    def _clean_messages(self, messages: list) -> list:
        """
        Normalize messages to plain dicts and strip provider-specific fields.

        - Converts OpenAI response objects (e.g. choice.message) to dicts
          so they can be safely replayed as history.
        - Removes 'metadata' which OpenAI returns but Groq rejects.
        - Removes None-valued keys to keep payloads clean.

        This is a safe no-op for OpenAI and Gemini — they ignore unknown
        fields, so stripping extras never breaks them.
        """
        # Fields that Groq rejects but OpenAI may include in response objects
        UNSUPPORTED_FIELDS = {"metadata"}

        cleaned = []
        for m in messages:

            # Convert OpenAI SDK objects → plain dict
            if hasattr(m, "model_dump"):
                m = m.model_dump(exclude_none=True)
            elif hasattr(m, "__dict__"):
                m = {k: v for k, v in m.__dict__.items() if v is not None}

            if isinstance(m, dict):
                m = {k: v for k, v in m.items()
                     if k not in UNSUPPORTED_FIELDS and v is not None}

            cleaned.append(m)
        return cleaned