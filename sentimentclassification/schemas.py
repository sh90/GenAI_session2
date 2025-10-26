from __future__ import annotations
from pydantic import BaseModel, Field, constr

Short140 = constr(strip_whitespace=True, max_length=140)

class SentimentResult(BaseModel):
    sentiment: str  # positive | neutral | negative
    evidence: Short140
