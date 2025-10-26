from __future__ import annotations
import os
from common.prompts import render_template
from common.llm import LLMClient
from summarization.schemas import ArticleSummary

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__))
TEMPLATE = "prompt_article.j2"

def build_messages(article_text: str):
    prompt = render_template(TEMPLATE_DIR, TEMPLATE, article_text=article_text)
    return [{"role": "user", "content": prompt}]

def summarize_article(text: str) -> ArticleSummary:
    client = LLMClient()
    msgs = build_messages(text)
    return client.run_structured(msgs, ArticleSummary, temperature=0.2)

if __name__ == "__main__":
    sample = """ 
    Ruben Amorim is too wise to get sucked into the wider talk about Manchester United.
    
    "You said it," he pointed out in response to a question about his team's improved form this month. "Three weeks."
    
    Let's wind back.
    
    Three weeks ago, United went into a game against Sunderland, a side that has just beaten Chelsea and are sitting very nicely in a Champions League berth, amid reports Amorim was at risk of losing his job if his side had lost.
    
    Senior club officials strongly rejected the notion privately before the game and minority owner Sir Jim Ratcliffe has since done so publicly.
    
    But that was the backdrop.
    
    United won, one of those routine home victories that were commonplace in the glory days under Sir Alex Ferguson. Last week, they beat Liverpool at Anfield for the first time since 2016.
    
    Now they have overcome Brighton, who have made an art form out of beating United in recent years.
    
    After 11 months of almost relentless negativity, Amorim is suddenly presiding over a success story. By the final whistle of the 4-2 win against Brighton, United were fourth, with a positive goal difference. In relative terms, these are heady days. It feels like a very big corner has been turned.
    
    Not so fast, says Amorim.
    
    "The team is playing so much better since we start this season compared to last," he said.
    
    "But you [journalist] said everything. It was three weeks ago. So, it can change in the next three weeks."
    """
    res = summarize_article(sample)
    print(res.model_dump_json(indent=2))
