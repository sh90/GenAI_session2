from __future__ import annotations
import os, pandas as pd
from common.prompts import render_template
from common.llm import LLMClient
from sentimentclassification.schemas import SentimentResult

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt.j2"

def build_messages(text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, user_text=text)
    return [{"role": "user", "content": prompt}]

def run_batch(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv).head(n=5)
    client = LLMClient()
    out_rows = []

    for _, row in df.iterrows():
        text = row["review"]
        msgs = build_messages(text)
        result = client.run_structured(msgs, SentimentResult, temperature=0.3)
        out_rows.append(result.model_dump())

    df_out = pd.concat([df, pd.DataFrame(out_rows)], axis=1)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved → {output_csv}")

if __name__ == "__main__":
    in_csv = os.path.join(os.path.dirname(__file__), "data", "sentiments.csv")
    print(in_csv)
    out_csv = os.path.join(os.path.dirname(__file__), "data", "sentiments_out.csv")
    run_batch(in_csv, out_csv)
