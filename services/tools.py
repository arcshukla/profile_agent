import json

from services.notifier import Notifier
from utils.metrics import metrics
from utils.logger import get_logger, log_email, log_unknown, log_message

logger = get_logger(__name__)
notifier = Notifier()

def record_user_details(email, name="Lead", notes="", session_id=""):
    """Record user details (email, name) and notify.
    
    Args:
        email: User's email address
        name: User's name (default: 'Lead')
        notes: Additional notes
        session_id: User's Gradio session hash for CSV logging
    """
    log_email(email, session_id=session_id)
    notifier.notify_lead(name, email, session_id=session_id)

    metrics.record_lead()

    return {"status": "lead recorded"}


def record_unknown_question(question, session_id=""):
    """Record an unanswered question and notify.
    
    Args:
        question: The user's question that couldn't be answered
        session_id: User's Gradio session hash for CSV logging
    """
    log_unknown(question, session_id=session_id)
    notifier.notify_unknown(question, session_id=session_id)

    metrics.record_unknown()

    return {"status": "unknown recorded"}


tools = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Use when the assistant cannot find the answer in the profile context. This records unanswered recruiter questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question that could not be answered"
                    }
                },
                "required": ["question"]
            }
        }
    }
]


def handle_tool_call(tool_calls, session_id=""):
    """Execute tool calls and collect results.
    
    Args:
        tool_calls: List of tool calls from LLM
        session_id: User's Gradio session hash to pass to tool functions
    
    Returns:
        List of assistant messages for tool call results
    """
    results = []
    json_dump = ""

    for call in tool_calls:

        name = call.function.name
        args = json.loads(call.function.arguments)
        args["session_id"] = session_id  # Inject session_id into tool function args

        fn = globals().get(name)

        result = fn(**args) if fn else {}
        json_dump = json.dumps(result)

        results.append({
            "role": "tool",
            "content": json_dump,
            "tool_call_id": call.id
        })

    log_message("TOOL RESULTS:" + json_dump)

    return results