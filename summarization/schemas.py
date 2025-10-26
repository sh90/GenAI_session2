from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, constr

Short120 = constr(strip_whitespace=True, max_length=120)

class Entity(BaseModel):
    name: str
    type: Literal["person","org","place","other"]

class ArticleSummary(BaseModel):
    summary: str
    key_points: List[Short120] = Field(default_factory=list, max_length=5)
    entities: List[Entity] = Field(default_factory=list)

class ActionItem(BaseModel):
    owner: str
    task: str
    due: Optional[str] = None  # ISO date or null

class ChatSummary(BaseModel):
    tl_dr: str
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
