# backend/ml_oov/oov_pipeline/pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set
from types import SimpleNamespace
from pathlib import Path
import json
import re

from .yaml_candidates_store import YAMLCandidateStore
from .config import load_config
from .postprocess import apply_conf_threshold, nms_char_spans, remove_known_overlaps
from .span_recover import recover_spans
from .types import LexiconMatcher, LexiconSpan, PipelineResult, PredictTokens, CharSpan
from .filters_ko import keep_candidate
from .heuristic_extract import extract_heuristic_oov_candidates


def _wrap_token_dicts_as_objects(tokens: List[Dict[str, Any]]) -> List[Any]:
    """
    recover_spans()가 token.label 처럼 속성 접근을 사용하므로,
    predictor가 dict를 반환하면 SimpleNamespace로 변환해준다.
    """
    objs = []
    for t in tokens:
        tt = dict(t)
        if "conf" in tt and "confidence" not in tt:
            tt["confidence"] = tt["conf"]
        objs.append(SimpleNamespace(**tt))
    return objs


def _span_to_dict(s: Any) -> Dict[str, Any]:
    """JSON 직렬화용 최소 span dict"""
    return {
        "start": getattr(s, "start", None),
        "end": getattr(s, "end", None),
        "text": getattr(s, "text", None),
        "label": getattr(s, "label", None),
        "confidence": getattr(s, "confidence", None),
        "source": getattr(s, "source", None),
    }


# ---------------------------
# context-based OOV mining (refined)
# ---------------------------
_HANGUL_TOKEN = re.compile(r"[가-힣]{2,8}")  # 조금 여유

_CONTEXT_STOP = {
    "오늘", "어제", "내일",
    "너무", "진짜", "정말", "완전", "그냥", "약간",
    "했다", "한다", "됨", "된다", "하는", "있다", "없다",
    "왔다", "갔다", "된다", "같다",
}

_JOSA_SUFFIX = (
    "까지", "부터", "에서", "으로", "에게", "한테",
    "과", "와", "로", "에",
    "을", "를", "은", "는", "이", "가", "도", "만",
)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _strip_josa(tok: str) -> str:
    tok = (tok or "").strip()
    if len(tok) < 3:
        return tok
    for suf in _JOSA_SUFFIX:
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            return tok[: -len(suf)]
    return tok


def _mine_oov_from_context(
    text: str,
    *,
    anchors: List[Any],
    lexicon_spans: List[LexiconSpan],
    window: int = 12,
) -> List[CharSpan]:
    """
    anchor(사전/모델 span) 주변 ±window 글자 범위에서
    '사전에 없는 한글 토큰'을 OOV 후보(CharSpan)로 생성한다.

    개선점:
    - lexicon span 위치와 겹치면 제외
    - 기본 stopwords 제외
    - 조사 제거 normalize (현타가 -> 현타)
    """
    anchor_spans: List[Tuple[int, int]] = []
    anchor_texts: Set[str] = set()

    for a in anchors or []:
        s = getattr(a, "start", None)
        e = getattr(a, "end", None)
        t = getattr(a, "text", None)
        if isinstance(s, int) and isinstance(e, int) and s < e:
            anchor_spans.append((s, e))
        if isinstance(t, str) and t:
            anchor_texts.add(t)

    # ✅ 앵커가 없으면 context mining 불가
    if not anchor_spans:
        return []

    lex_texts: Set[str] = set(getattr(x, "text", "") for x in (lexicon_spans or []))

    out: List[CharSpan] = []
    seen_key: Set[Tuple[int, int, str]] = set()

    for (s, e) in anchor_spans:
        left = max(0, s - window)
        right = min(len(text), e + window)
        chunk = text[left:right]

        for m in _HANGUL_TOKEN.finditer(chunk):
            ts = left + m.start()
            te = left + m.end()
            raw = text[ts:te]

            if not raw or raw.strip() == "":
                continue

            # ✅ lexicon span과 위치 겹치면 제외
            if any(_overlaps(ts, te, ls.start, ls.end) for ls in (lexicon_spans or [])):
                continue

            # anchor 자체 텍스트면 제외
            if raw in anchor_texts:
                continue

            # 사전에 있는 단어면 제외 (텍스트 기준 2차)
            if raw in lex_texts:
                continue

            norm = _strip_josa(raw)
            if not norm:
                continue

            if norm in _CONTEXT_STOP:
                continue

            if len(norm) < 2:
                continue

            if norm != raw:
                te = ts + len(norm)

            key = (ts, te, norm)
            if key in seen_key:
                continue
            seen_key.add(key)

            out.append(
                CharSpan(
                    start=ts,
                    end=te,
                    text=norm,
                    label="CAND_OOV",
                    confidence=1.0,
                    source="context",
                )
            )

    return out


def run_oov_pipeline(
    text: str,
    *,
    predictor: PredictTokens,
    lexicon_matcher: Optional[LexiconMatcher] = None,
    lexicon_spans: Optional[List[LexiconSpan]] = None,
    target_labels: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """단일 문서에 대해 B 측 OOV 파이프라인을 실행한다."""

    cfg = load_config()
    if config_overrides:
        for k, v in config_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if not cfg.enabled:
        return PipelineResult(
            oov_spans=[],
            model_spans=[],
            lexicon_spans=[],
            debug={"enabled": False},
        )

    # 1) KNOWN span 확보 (사전)
    if lexicon_spans is None:
        if lexicon_matcher is None:
            lexicon_spans = []
        else:
            lexicon_spans = lexicon_matcher(text)

    # 2) 토큰 단위 예측
    preds_raw = predictor(text)

    if isinstance(preds_raw, list):
        if preds_raw and isinstance(preds_raw[0], dict):
            tokens_obj = _wrap_token_dicts_as_objects(preds_raw)
            preds = SimpleNamespace(text=text, tokens=tokens_obj)
            debug_pred_shape = "list[dict]"
        else:
            preds = SimpleNamespace(text=text, tokens=preds_raw)
            debug_pred_shape = "list[obj]"
    else:
        preds = preds_raw
        debug_pred_shape = type(preds_raw).__name__

    # 3) BIO → 문자 span 복원
    model_spans = recover_spans(
        preds,
        target_prefixes=target_labels,
        source="koelectra",
    )

    # 4) 후처리: NMS → 사전 제거 → confidence threshold
    model_spans_nms = nms_char_spans(model_spans)
    oov_spans = remove_known_overlaps(model_spans_nms, lexicon_spans)
    oov_spans = apply_conf_threshold(oov_spans, cfg.conf_threshold)

    # 5) context 기반 후보 생성 (앵커가 있어야 발동)
    context_window = getattr(cfg, "context_window", 12)
    context_cands = _mine_oov_from_context(
        text,
        anchors=(lexicon_spans or []) + (model_spans_nms or []),
        lexicon_spans=lexicon_spans or [],
        window=int(context_window),
    )

    if context_cands:
        existing_pos = {(getattr(s, "start", None), getattr(s, "end", None)) for s in oov_spans}
        for s in context_cands:
            pos = (s.start, s.end)
            if pos not in existing_pos:
                oov_spans.append(s)

    # ✅ CHANGED: heuristic은 "oov_spans가 비었을 때만"이 아니라,
    #            항상 돌리고(혹은 context/model이 약할 때), 중복/사전겹침 제거 후 합친다.
    heuristic_cands: List[CharSpan] = extract_heuristic_oov_candidates(
        text,
        lexicon_spans=lexicon_spans or [],
    )

    # heuristic 후보를 oov_spans에 합치되:
    # - lexicon span 위치 겹치면 제거(이중 안전)
    # - 이미 같은 위치 span 있으면 제거
    if heuristic_cands:
        existing_pos = {(getattr(s, "start", None), getattr(s, "end", None)) for s in oov_spans}
        for s in heuristic_cands:
            if any(_overlaps(s.start, s.end, ls.start, ls.end) for ls in (lexicon_spans or [])):
                continue
            pos = (s.start, s.end)
            if pos in existing_pos:
                continue
            oov_spans.append(s)
            existing_pos.add(pos)

    debug: Dict[str, Any] = {
        "enabled": True,
        "conf_threshold": cfg.conf_threshold,
        "predictor_return": debug_pred_shape,
        "counts": {
            "model_spans": len(model_spans),
            "model_spans_nms": len(model_spans_nms),
            "lexicon_spans": len(lexicon_spans),
            "oov_spans": len(oov_spans),
        },
        "context_window": int(context_window),
        "context_candidates_added": len(context_cands),
        "heuristic_candidates_added": len(heuristic_cands),
    }

    # 6) 후보 저장 (A안 구조)
    if cfg.write_candidates:
        store = YAMLCandidateStore(
            cfg.candidate_path,
            max_examples=cfg.max_examples_per_key,
        )
        debug["candidate_path"] = cfg.candidate_path
        debug["candidate_mode"] = "spans"

        # ✅ 저장 직전 최종 필터:
        # - keep_candidate
        # - lexicon 위치 겹침 제거(아주 마지막 안전망)
        filtered_oov: List[CharSpan] = []
        for s in oov_spans:
            txt = getattr(s, "text", "")
            if not keep_candidate(txt):
                continue
            if any(_overlaps(s.start, s.end, ls.start, ls.end) for ls in (lexicon_spans or [])):
                continue
            filtered_oov.append(s)

        debug["candidate_oov_before_filter"] = len(oov_spans)
        debug["candidate_oov_after_filter"] = len(filtered_oov)

        if filtered_oov:
            updated = store.update_from_spans(filtered_oov)
            debug["candidate_records"] = len(updated)
        else:
            debug["candidate_records"] = 0
            debug["candidate_reason"] = "no_oov_spans_after_filter"

            try:
                rejected_path = Path("backend/data/rejected_candidates.jsonl")
                rejected_path.parent.mkdir(parents=True, exist_ok=True)

                rec = {
                    "text": text,
                    "model_spans": [_span_to_dict(s) for s in model_spans_nms],
                    "lexicon_spans": [_span_to_dict(s) for s in lexicon_spans],
                    "reason": debug["candidate_reason"],
                    "candidate_oov_before_filter": debug["candidate_oov_before_filter"],
                    "candidate_oov_after_filter": debug["candidate_oov_after_filter"],
                }

                with rejected_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                debug["rejected_store_error"] = str(e)
    else:
        debug["candidate_path"] = None
        debug["candidate_mode"] = "disabled"

    return PipelineResult(
        oov_spans=oov_spans,
        model_spans=model_spans_nms,
        lexicon_spans=lexicon_spans,
        debug=debug,
    )
