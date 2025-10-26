from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from ner.schemas import NERResult

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt.j2"

def build_messages(text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, user_text=text)
    return [{"role": "user", "content": prompt}]

def extract_entities(text: str) -> NERResult:
    client = LLMClient()
    msgs = build_messages(text)
    return client.run_structured(msgs, NERResult, temperature=0.2)

if __name__ == "__main__":
    ex = "I'm Rohan from Acme Corp, in Bengaluru. Order #AB1234 arrived damaged."
    res = extract_entities(ex)
    print(res.model_dump_json(indent=2))
