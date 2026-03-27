from dotenv import load_dotenv
import gradio as gr
import os
from agents.profile.profile import ProfileAgent
from utils.metrics import metrics
from utils.file_reader import read_file
from utils.logger import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)
logger.info("Starting the application..")

agent = ProfileAgent()

def load_css():
    return read_file("ui/style.css")

def load_script():
    return read_file("ui/script.js")

def load_header():
    return read_file("ui/header.html")

def get_profile_image():
    return "ui/profile.jpg"    

def make_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}

# -------------------------
# Chat functions
# -------------------------
 
def add_user_message(message, history):
    history.append(make_message("user", message))
    return "", history
 
def generate_response(history, request: gr.Request):
    message = history[-1]["content"][0]["text"]
    metrics.record_question()
 
    # Gradio's session_hash is stable for the lifetime of one browser tab —
    # every turn from the same visitor carries the same hash.
    # Fallback to empty string if unavailable (e.g. unit tests).
    session_id = getattr(request, "session_hash", "") or ""
 
    answer, followups = agent.chat(message, history, session_id=session_id)
    history.append(make_message("assistant", answer))
    metrics.summary()
    followups = (followups + ["", "", ""])[:3]
    updates = [gr.update(value=f, visible=bool(f)) for f in followups]
    return (history, *updates)
 
# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(
    title=f"Digital Avatar for {agent.name}",
) as demo:
 
    with gr.Row(elem_id="header-bar", equal_height=False):
        with gr.Column(scale=0, min_width=120):
            gr.Image(
                value=get_profile_image(),
                elem_id="profile-image",
                show_label=False,
                interactive=False
            )
 
        with gr.Column():
            gr.HTML(load_header())
 
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=480,
        autoscroll=False
    )
 
    msg = gr.Textbox(
        label="Question",
        placeholder=agent.chat_placeholder,
        elem_id="question-box",
        max_length=150
    )
 
    with gr.Row(elem_id="followups"):
        btns = [
            gr.Button(agent.followup_questions[i])
            for i in range(3)
        ]
 
    msg.submit(
        add_user_message,
        [msg, chatbot],
        [msg, chatbot],
    ).then(
        generate_response,
        [chatbot],
        [chatbot, *btns]
    )
 
    for btn in btns:
        btn.click(
            add_user_message,
            [btn, chatbot],
            [msg, chatbot],
        ).then(
            generate_response,
            [chatbot],
            [chatbot, *btns]
        )
 
    chatbot.value = [make_message("assistant", agent.welcome_message)]
 
# On Hugging Face Spaces, share=True is not supported (Spaces are already public)
is_hf_space = "HF_SPACE_ID" in os.environ
is_local = os.getenv("IS_LOCAL", "FALSE").strip().upper() == "TRUE"
share = not is_hf_space and not is_local

demo.launch(css=load_css(), js=load_script(), share=share)