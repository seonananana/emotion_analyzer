"""Sentence split utility preserving original char offsets.

Protocol rule:
- Split by one of: '.', '!', '?', '\n'
- DO NOT strip/trim sentences (offset must align with original text)
"""

from __future__ import annotations
from typing import List, Tuple

_DELIMS = {'.', '!', '?', '\n'}

def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (sent_text, sent_start, sent_end) with end exclusive.

    Notes:
    - Keeps delimiters attached to the sentence.
    - Skips empty spans (length==0), but does NOT strip whitespace within spans.
    """
    out: List[Tuple[str, int, int]] = []
    start = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch in _DELIMS:
            end = i + 1
            if end > start:
                out.append((text[start:end], start, end))
            start = end
    if start < n:
        out.append((text[start:n], start, n))
    return [(s, a, b) for (s, a, b) in out if b > a]
