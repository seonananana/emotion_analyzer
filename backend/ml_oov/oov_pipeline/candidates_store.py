from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .types import CandidateRecord, CharSpan, LexiconSpan, TokenPredictions
from .filters_ko import keep_candidate


_HANGUL_RE = re.compile(r"[가-힣]")
_ONLY_HANGUL_RE = re.compile(r"^[가-힣]+$")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _context_window(text: str, start: int, end: int, win: int = 24) -> str:
    s = max(0, start - win)
    e = min(len(text), end + win)
    return text[s:e]


class JSONLCandidateStore:
    """
    CandidateRecord를 JSONL로 누적 저장하는 스토어.

    한 줄에 CandidateRecord 1개를 저장하며,
    key(표현)를 기준으로 count / avg_confidence / examples / timestamps를 누적 갱신한다.
    """

    def __init__(self, path: str, max_examples: int = 5):
        self.path = path
        self.max_examples = max_examples
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _load(self) -> Dict[str, CandidateRecord]:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return {}

        out: Dict[str, CandidateRecord] = {}
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = CandidateRecord(
                        key=d["key"],
                        text=d.get("text", d["key"]),
                        label=d.get("label", "CAND"),
                        count=int(d.get("count", 1)),
                        avg_confidence=float(d.get("avg_confidence", 0.0)),
                        examples=list(d.get("examples", [])),
                        first_seen_at=d.get("first_seen_at", _now_iso()),
                        last_seen_at=d.get("last_seen_at", _now_iso()),
                    )
                    out[rec.key] = rec
                except Exception as e:
                    # JSONL 한 줄이 깨져 있어도 전체를 죽이지 않도록 방어
                    raise ValueError(f"[JSONLCandidateStore] JSON decode failed at line {line_no}: {e}")
        return out

    def _dump(self, records: Dict[str, CandidateRecord]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for k in sorted(records.keys()):
                f.write(json.dumps(asdict(records[k]), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def _upsert(
        self,
        records: Dict[str, CandidateRecord],
        *,
        key: str,
        text: str,
        label: str,
        confidence: float,
        example: Optional[str],
    ) -> None:
        now = _now_iso()
        if key in records:
            rec = records[key]
            new_count = rec.count + 1
            # running average
            new_avg = (rec.avg_confidence * rec.count + confidence) / new_count
            examples = rec.examples[:]
            if example:
                if example not in examples:
                    examples.append(example)
                examples = examples[: self.max_examples]
            records[key] = CandidateRecord(
                key=rec.key,
                text=rec.text,
                label=rec.label,
                count=new_count,
                avg_confidence=float(new_avg),
                examples=examples,
                first_seen_at=rec.first_seen_at,
                last_seen_at=now,
            )
        else:
            examples = []
            if example:
                examples.append(example)
            records[key] = CandidateRecord(
                key=key,
                text=text,
                label=label,
                count=1,
                avg_confidence=float(confidence),
                examples=examples[: self.max_examples],
                first_seen_at=now,
                last_seen_at=now,
            )

    def update_from_spans(self, spans: List[CharSpan]) -> List[CandidateRecord]:
        """
        span 기반 후보 누적(기존 방식).
        key = span.text
        label = span.label
        """
        records = self._load()
        for sp in spans:
            key = sp.text.strip()
            if not key:
                continue
            self._upsert(
                records,
                key=key,
                text=sp.text,
                label=sp.label,
                confidence=float(sp.confidence),
                example=None,  # span은 example 굳이 안 넣음(원하면 text context로 확장 가능)
            )
        self._dump(records)
        return list(records.values())

    def update_from_tokens(
        self,
        preds: TokenPredictions,
        *,
        lexicon_spans: Optional[List[LexiconSpan]] = None,
        min_len: int = 2,
        token_conf_threshold: float = 0.0,
        only_hangul: bool = True,
    ) -> List[CandidateRecord]:
        records = self._load()
        text = preds.text
        lex = lexicon_spans or []

        for t in preds.tokens:
            s, e = int(t.start), int(t.end)
            if s < 0 or e > len(text) or s >= e:
                continue

            surface = text[s:e].strip()
            if not surface:
                continue
            if not keep_candidate(surface):
                 continue

            # 길이/문자 필터
            if len(surface) < min_len:
                continue
            if only_hangul and not _ONLY_HANGUL_RE.match(surface):
                continue
            if not _HANGUL_RE.search(surface):
                continue

            span = (s, e)

            # ✅ 1차: 위치 겹침으로 제외
            if any(_overlaps(span, (ls.start, ls.end)) for ls in lex):
                continue

            # ✅ 2차 안전장치: 문자열 기준으로도 제외
            # (예: 형태소/토크나이즈로 offset이 살짝 어긋나거나, surface가 변형된 경우 방어)
            if any(surface == ls.text or surface in ls.text or ls.text in surface for ls in lex):
                continue

            conf = float(t.confidence)
            if conf < token_conf_threshold:
                continue

            label = "CAND"
            ex = _context_window(text, s, e, win=24)

            self._upsert(
                records,
                key=surface,
                text=surface,
                label=label,
                confidence=conf,
                example=ex,
            )

        self._dump(records)
        return list(records.values())
