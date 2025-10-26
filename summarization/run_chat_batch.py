from __future__ import annotations
import os, json, pandas as pd
from typing import List, Dict
from common.prompts import render_template
from common.llm import LLMClient
from summarization.schemas import ChatSummary

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt_chat.j2"

# ---- Config knobs ----
MAX_TRANSCRIPT_CHARS = 12000     # safety cap for very long chats
KEEP_LAST_TURNS = 80             # if too long, keep last N turns (recency bias)
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "data", "chat_transcripts_out.csv")
OUTPUT_JSONL = os.path.join(os.path.dirname(__file__), "data", "chat_transcripts.jsonl")

def load_chat_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"conversation_id": str, "turn": int, "speaker": str, "message": str})
    # basic sanity
    required = {"conversation_id", "turn", "speaker", "message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def build_transcript(df_conv: pd.DataFrame) -> str:
    """
    Format one conversation as a readable transcript that fits within model limits.
    We keep the last N turns if the text becomes too long.
    """
    df_conv = df_conv.sort_values("turn")
    # If too many turns, trim to last KEEP_LAST_TURNS
    if len(df_conv) > KEEP_LAST_TURNS:
        df_conv = df_conv.tail(KEEP_LAST_TURNS)

    # Speaker: message lines, no commas to keep things simple
    lines: List[str] = [f"{row.speaker}: {row.message}" for _, row in df_conv.iterrows()]
    transcript = "\n".join(lines)
    # Truncate by characters as a final guard
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
    return transcript

def summarize_one_conversation(text: str) -> ChatSummary:
    """
    Use provider-agnostic LLMClient to produce a structured ChatSummary.
    """
    client = LLMClient()  # reads env: PROVIDER, OPENAI_MODEL, OLLAMA_MODEL, OPENAI_API_KEY
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, chat_text=text)
    messages = [{"role": "user", "content": prompt}]
    result = client.run_structured(messages, ChatSummary, temperature=0.2)
    return result

def flatten_action_items(items: List[Dict]) -> str:
    """
    For CSV output: condense action items to a semicolon-separated string.
    Example: "Ravi: ship replacement by 2025-11-01; Meera: email invoice by null"
    """
    parts = []
    for it in items or []:
        owner = it.get("owner", "")
        task = it.get("task", "")
        due = it.get("due", "")
        parts.append(f"{owner}: {task} by {due}")
    return "; ".join(parts)

def run_batch_chat_summaries(input_csv: str, out_csv: str = OUTPUT_CSV, out_jsonl: str = OUTPUT_JSONL):
    df = load_chat_csv(input_csv)

    results = []
    jsonl_rows = []

    for conv_id, df_conv in df.groupby("conversation_id", sort=True):
        print(f"🗂️ Summarizing conversation {conv_id} (turns={len(df_conv)})")
        transcript = build_transcript(df_conv)
        summary = summarize_one_conversation(transcript)

        # store for CSV
        decisions_str = "; ".join(summary.decisions or [])
        action_items_str = flatten_action_items([ai.model_dump() if hasattr(ai, "model_dump") else ai for ai in summary.action_items])

        results.append({
            "conversation_id": conv_id,
            "turns": len(df_conv),
            "tl_dr": summary.tl_dr,
            "decisions": decisions_str,
            "action_items": action_items_str,
        })

        # store for JSONL (full fidelity)
        jsonl_rows.append({
            "conversation_id": conv_id,
            "summary": summary.model_dump(),
        })

    # write CSV
    out_df = pd.DataFrame(results).sort_values("conversation_id")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f" Wrote CSV → {out_csv}")

    # write JSONL
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f" Wrote JSONL → {out_jsonl}")

if __name__ == "__main__":
    # Example: place your chat CSV at data/chat_transcripts.csv
    input_csv = os.path.join(os.path.dirname(__file__), "data", "chat_transcripts.csv")
    run_batch_chat_summaries(input_csv)
