"""Optional lightweight schemas for stable I/O.

No external deps required (dataclasses only).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class TokenPrediction:
    token: str
    start: int
    end: int
    label: str
    confidence: float

@dataclass(frozen=True)
class DocPrediction:
    doc_id: Optional[str]
    text_len: int
    tokens: List[TokenPrediction]
