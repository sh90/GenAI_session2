from __future__ import annotations
from pydantic import BaseModel, Field, constr
from typing import List

ShortStr = constr(strip_whitespace=True, max_length=120)

class Step(BaseModel):
    explanation: ShortStr
    output: str

class MathReasoning(BaseModel):
    steps: List[Step] = Field(min_length=1)
    final_answer: str
