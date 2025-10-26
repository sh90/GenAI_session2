from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from summarization.schemas import ChatSummary

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt_chat.j2"

def build_messages(chat_text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, chat_text=chat_text)
    return [{"role": "user", "content": prompt}]

def summarize_chat(text: str) -> ChatSummary:
    client = LLMClient()
    msgs = build_messages(text)
    return client.run_structured(msgs, ChatSummary, temperature=0.2)

if __name__ == "__main__":
    transcript = """Alice: shipping delay is from vendor X
    Bob: approve partial refund?
    Alice: yes, INR 200. Assign to Ravi by Friday."""
    res = summarize_chat(transcript)
    print(res.model_dump_json(indent=2))
