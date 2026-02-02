"""Label definitions (single source of truth)."""

from __future__ import annotations

LABELS = ["O", "B-NEG_OOV", "I-NEG_OOV"]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# Padding / special tokens should be ignored in loss
IGNORE_INDEX = -100
