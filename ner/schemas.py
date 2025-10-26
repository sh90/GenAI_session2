from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, constr

Short140 = constr(strip_whitespace=True, max_length=140)

class KV(BaseModel):
    name: str

class Order(BaseModel):
    id: str

class Issue(BaseModel):
    label: str
    evidence: Short140

class NERResult(BaseModel):
    persons: List[KV] = Field(default_factory=list)
    organizations: List[KV] = Field(default_factory=list)
    locations: List[KV] = Field(default_factory=list)
    order_ids: List[Order] = Field(default_factory=list)
    issues: List[Issue] = Field(default_factory=list)
