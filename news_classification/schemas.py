from __future__ import annotations
from pydantic import BaseModel, Field, constr

Short140 = constr(strip_whitespace=True, max_length=140)
class NewsTopic(BaseModel):
    topic: str  # constrained by taxonomy at prompt-time
    confidence: str  # low|medium|high
    evidence: Short140
