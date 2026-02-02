from __future__ import annotations

import os
import yaml
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from .types import CandidateRecord, CharSpan, LexiconSpan, TokenPredictions

# candidates_store.py에 이미 있을 수도 있는 util을 최소로 복제
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _context_window(text: str, start: int, end: int, win: int = 24) -> str:
    a = max(0, start - win)
    b = min(len(text), end + win)
    return text[a:b]

def _overlaps(a, b) -> bool:
    # a,b: (start,end)
    return not (a[1] <= b[0] or b[1] <= a[0])

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _is_bad_candidate_key(key: str) -> bool:
    key = (key or "").strip()
    if not key:
        return True

    # 개행/캐리지리턴 금지
    if ("\n" in key) or ("\r" in key):
        return True

    # 공백 포함(단어 합쳐짐/깨짐) 금지: "빡 멘붕"
    if any(ch.isspace() for ch in key):
        return True

    # 1글자: 기능어/잡음만 차단, 나머지는 허용(예: '빡'은 통과)
    deny_1 = {
        "고","에","을","를","은","는","이","가","도","만","과","와","로","의",
        "다","요","지","서","면","듯","걸","게",
        "타",  # 현타 -> 타 같은 케이스 차단
    }
    if len(key) == 1 and key in deny_1:
        return True

    # 2글자 대표 잡음 차단
    deny_2 = {"겠다"}  # 필요하면 추가
    if len(key) == 2 and key in deny_2:
        return True

    return False

class YAMLCandidateStore:
    """
    YAML 1파일에 dict 형태로 저장(매번 덮어쓰기).
    key -> CandidateRecord
    """

    def __init__(self, path: str, max_examples: int = 5):
        self.path = path
        self.max_examples = max_examples

    def _load(self) -> Dict[str, CandidateRecord]:
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        recs: Dict[str, CandidateRecord] = {}
        for k, v in data.items():
            recs[k] = CandidateRecord(
                key=v.get("key", k),
                text=v.get("text", k),
                label=v.get("label", "CAND"),
                count=int(v.get("count", 0)),
                avg_confidence=_safe_float(v.get("avg_confidence", 0.0)),
                examples=list(v.get("examples", []) or []),
                first_seen_at=v.get("first_seen_at", _now()),
                last_seen_at=v.get("last_seen_at", _now()),
            )
        return recs

    def _dump(self, recs: Dict[str, CandidateRecord]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        out = {k: asdict(v) for k, v in recs.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    def _upsert(
        self,
        recs: Dict[str, CandidateRecord],
        *,
        key: str,
        text: str,
        label: str,
        confidence: float,
        example: str,
    ) -> None:
        key = (key or "").strip()
       
        if _is_bad_candidate_key(key):
            return
       
        now = _now()
        if key not in recs:
            recs[key] = CandidateRecord(
                key=key,
                text=text,
                label=label,
                count=1,
                avg_confidence=float(confidence),
                examples=[example][: self.max_examples],
                first_seen_at=now,
                last_seen_at=now,
            )
            return

        r = recs[key]
        new_count = r.count + 1
        new_avg = (r.avg_confidence * r.count + float(confidence)) / new_count

        examples = list(r.examples)
        if example and example not in examples:
            examples.append(example)
        examples = examples[: self.max_examples]

        recs[key] = CandidateRecord(
            key=r.key,
            text=r.text,
            label=r.label,
            count=new_count,
            avg_confidence=float(new_avg),
            examples=examples,
            first_seen_at=r.first_seen_at,
            last_seen_at=now,
        )

    # ---------- public API (pipeline에서 쓰는 것들) ----------

    def update_from_spans(self, spans: List[CharSpan]) -> List[CandidateRecord]:
        recs = self._load()
        for s in spans:
            key = (s.text or "").strip()

            # ✅ 후보 키 정제/차단 (개행/1글자)
            if _is_bad_candidate_key(key):
                continue

            if not key:
                continue
            ex = _context_window(s.text, 0, len(s.text), win=0)
            self._upsert(
                recs,
                key=key,
                text=s.text,
                label=s.label,
                confidence=float(s.confidence),
                example=ex,
            )
        self._dump(recs)
        return list(recs.values())

    def update_from_tokens(
        self,
        preds: TokenPredictions,
        *,
        lexicon_spans: Optional[List[LexiconSpan]] = None,
        keep_candidate_fn=None,
        token_conf_threshold: float = 0.0,
        min_len: int = 2,
        max_len: int = 8,
        stopwords=None,
        **kwargs,
    ) -> List[CandidateRecord]:
        """
        YAML 후보 누적 저장.

        핵심:
        - label=='O'는 저장하지 않음 (조사/일반단어 폭주 방지)
        - min_len 기본 2
        - 기본 stopwords(조사/어미/구두점) 적용 + 외부 stopwords 합침
        """
        recs = self._load()
        text = preds.text
        lex = lexicon_spans or []

        # ✅ 기본 stopwords (최소)
        base_sw = {
            # 조사/어미/기능어(상위 폭주하는 것들)
            "고","에","을","를","은","는","이","가","도","만","과","와","로","으로","에게",
            "한","하","했","되","다","요","니다","습니다","았","었","던","게","지","서","면",
            "때","적","보다","같",
            # 구두점/기호
            ",",".","!","?","…","~","·","(",")","[","]","{","}","\"","'","`",
        }
        sw = set(base_sw)
        if stopwords:
            sw |= set(stopwords)

        for t in preds.tokens:
            # ✅ 라벨 필터: O는 스킵 (가장 중요)
            lab = str(getattr(t, "label", ""))
            if lab == "O":
                continue

            s, e = int(t.start), int(t.end)
            if s < 0 or e > len(text) or s >= e:
                continue

            surface = (text[s:e] or "").strip()
            if not surface:
                continue

            # ✅ 후보 키 정제/차단 (개행/1글자)
            if _is_bad_candidate_key(surface):
                continue
            # ✅ 단일 기호/숫자/영문 등 제거(최소 안전망)
            if all(ch.isdigit() for ch in surface):
                continue
            if all(("a" <= ch.lower() <= "z") for ch in surface):
                continue

            # ✅ stopwords
            if surface in sw:
                continue

            # ✅ 외부 필터(keep_candidate) 있으면 적용
            if keep_candidate_fn is not None and not keep_candidate_fn(surface):
                continue

            # ✅ 위치 겹침으로 lexicon 제외
            if any(_overlaps((s, e), (ls.start, ls.end)) for ls in lex):
                continue

            conf = float(t.confidence)
            if conf < float(token_conf_threshold):
                continue

            ex = _context_window(text, s, e, win=24)
            self._upsert(
                recs,
                key=surface,
                text=surface,
                label="CAND",
                confidence=conf,
                example=ex,
            )

        self._dump(recs)
        return list(recs.values())
