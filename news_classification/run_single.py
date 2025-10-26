from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from news_classification.schemas import NewsTopic

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt_zero_shot.j2"  # switch to few_shot if needed

def build_messages(text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, news_text=text)
    return [{"role": "user", "content": prompt}]

def classify_news(text: str) -> NewsTopic:
    client = LLMClient()
    msgs = build_messages(text)
    return client.run_structured(msgs, NewsTopic, temperature=0.2)

if __name__ == "__main__":
    headline = "Open-source model outperforms rivals on language tasks"
    res = classify_news(headline)
    print(res.model_dump_json(indent=2))
