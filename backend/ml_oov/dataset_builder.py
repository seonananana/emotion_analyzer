"""Build token-level BIO dataset from gold OOV spans.

Input gold format (jsonl):
  {"id":"doc_0001","text":"...","oov_spans":[{"start":18,"end":26,"label":"NEG_OOV"}]}

Output features (jsonl) (one sentence per row):
  {
    "doc_id": "...",
    "sent_start": 0,
    "sent_end": 42,
    "text": "sentence text...",
    "input_ids": [...],
    "attention_mask": [...],
    "labels": [...],
    "offset_mapping": [[s,e], ...]
  }

Notes:
- Uses tokenizer offset_mapping to align gold spans to token indices.
- Special tokens offset (0,0) are labeled IGNORE_INDEX.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .labels import LABEL2ID, IGNORE_INDEX
from .sent_split import split_sentences

def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _token_overlaps_span(tok_s: int, tok_e: int, span_s: int, span_e: int) -> bool:
    return max(tok_s, span_s) < min(tok_e, span_e)

def build_features(
    gold_jsonl: Path,
    out_jsonl: Path,
    tokenizer_name_or_path: str,
    max_length: int = 256,
    doc_id_key: str = "id",
) -> None:
    """Convert gold jsonl into token BIO features jsonl."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    rows = _load_jsonl(gold_jsonl)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    bad_gold = 0
    written = 0

    with out_jsonl.open("w", encoding="utf-8") as w:
        for r in rows:
            doc_id = r.get(doc_id_key)
            text = r["text"]
            spans_raw = r.get("oov_spans", []) or []

            spans: List[Tuple[int,int]] = []
            for sp in spans_raw:
                s, e = int(sp["start"]), int(sp["end"])
                if 0 <= s <= e <= len(text):
                    spans.append((s, e))
                else:
                    bad_gold += 1

            for sent_text, sent_start, sent_end in split_sentences(text):
                rel_spans = [(gs, ge) for (gs, ge) in spans if max(gs, sent_start) < min(ge, sent_end)]

                enc = tok(
                    sent_text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                offsets = enc["offset_mapping"]

                labels = [LABEL2ID["O"]] * len(input_ids)

                for i, (os, oe) in enumerate(offsets):
                    if os == 0 and oe == 0:
                        labels[i] = IGNORE_INDEX
                        continue

                    tok_s = sent_start + int(os)
                    tok_e = sent_start + int(oe)

                    hit = None
                    for (gs, ge) in rel_spans:
                        if _token_overlaps_span(tok_s, tok_e, gs, ge):
                            hit = (gs, ge)
                            break
                    if hit is None:
                        continue

                    prev_same = False
                    if i > 0:
                        ps, pe = offsets[i - 1]
                        if not (ps == 0 and pe == 0):
                            prev_s = sent_start + int(ps)
                            prev_e = sent_start + int(pe)
                            if _token_overlaps_span(prev_s, prev_e, hit[0], hit[1]):
                                prev_same = True

                    labels[i] = LABEL2ID["I-NEG_OOV"] if prev_same else LABEL2ID["B-NEG_OOV"]

                feat = {
                    "doc_id": doc_id,
                    "sent_start": sent_start,
                    "sent_end": sent_end,
                    "text": sent_text,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "offset_mapping": [[int(a), int(b)] for (a, b) in offsets],
                }
                w.write(json.dumps(feat, ensure_ascii=False) + "\n")
                written += 1

    print(f"[build_features] wrote={written} bad_gold={bad_gold} -> {out_jsonl}")
