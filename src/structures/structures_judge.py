"""Structures for LLM-as-a-judge scoring."""

from typing import Literal

from pydantic import BaseModel


class JudgeDimension(BaseModel):
    score: Literal[-1, 0, 1]
