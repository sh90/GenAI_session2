from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from reasoning.schemas import MathReasoning

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt.j2"

def build_messages(problem: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, problem=problem)
    return [{"role": "user", "content": prompt}]

def solve(problem: str) -> MathReasoning:
    client = LLMClient()
    msgs = build_messages(problem)
    return client.run_structured(msgs, MathReasoning, temperature=0.2)

if __name__ == "__main__":
    r = solve("8x + 7 = -23")
    print(r.model_dump_json(indent=2))
