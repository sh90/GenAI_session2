from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from sentimentclassification.schemas import SentimentResult

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt.j2"

def build_messages(text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, user_text=text)
    return [{"role": "user", "content": prompt}]

def analyze(text: str) -> SentimentResult:
    client = LLMClient()  # reads PROVIDER, OPENAI_MODEL, OLLAMA_MODEL from env
    msgs = build_messages(text)
    return client.run_structured(msgs, SentimentResult, temperature=0.3)

if __name__ == "__main__":
    res = analyze('Support was polite but my issue isn\'t fixed yet.')
    print(res.model_dump_json(indent=2))
