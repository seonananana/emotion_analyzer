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
# NEW: context-based OOV mining
# ---------------------------
_HANGUL_TOKEN = re.compile(r"[가-힣]{2,6}")


def _mine_oov_from_context(
    text: str,
    *,
    anchors: List[Any],
    lexicon_spans: List[LexiconSpan],
    window: int = 12,
) -> List[CharSpan]:
    """
    anchor(사전/모델 span) 주변 ±window 글자 범위에서
    '사전에 없는 한글 토큰(2~6자)'을 OOV 후보 span으로 생성한다.

    - 목적: 모델이 못 잡는 신조어(현타/멘붕/개빡 등)를
            '부정 단서(짜증/불안 등)' 주변에서 후보로 끌어올리기.
    """
    # 1) anchor spans/texts 수집 (LexiconSpan 객체 기반)
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

    if not anchor_spans:
        return []

    # 2) 사전 텍스트 set (빠른 제외용)
    lex_texts: Set[str] = set(getattr(x, "text", "") for x in (lexicon_spans or []))

    # 3) 주변 윈도우에서 토큰 추출 → LexiconSpan으로 반환
    out: List[LexiconSpan] = []
    seen_pos: Set[Tuple[int, int]] = set()

    for (s, e) in anchor_spans:
        left = max(0, s - window)
        right = min(len(text), e + window)
        chunk = text[left:right]

        for m in _HANGUL_TOKEN.finditer(chunk):
            ts = left + m.start()
            te = left + m.end()
            tok = text[ts:te]

            # anchor 자체/사전 단어는 제외
            if tok in anchor_texts:
                continue
            if tok in lex_texts:
                continue

            key = (ts, te)
            if key in seen_pos:
                continue
            seen_pos.add(key)

            out.append(
                CharSpan(
                    start=ts,
                    end=te,
                    text=tok,
                    label="CAND_OOV",     # 임시 라벨
                    confidence=1.0,       # rule 기반이니 1.0로 두자
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

    # ✅ NEW: context 기반 후보 생성 (신조어/구어체 OOV를 끌어올림)
    # - anchor = (사전 span + 모델 span NMS)
    # - anchor 주변에서 사전에 없는 한글 토큰을 OOV 후보로 생성
    context_window = getattr(cfg, "context_window", 12)  # config에 없으면 12
    context_cands = _mine_oov_from_context(
        text,
        anchors=(lexicon_spans or []) + (model_spans_nms or []),
        lexicon_spans=lexicon_spans or [],
        window=int(context_window),
    )
    if context_cands:
        # 중복 방지(동일 위치)
        existing_pos = {(getattr(s, "start", None), getattr(s, "end", None)) for s in oov_spans}
        for s in context_cands:
            pos = (s.start, s.end)
            if pos not in existing_pos:
                oov_spans.append(s)

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
    }

    # 5) 후보 저장 (A안 구조)
    if cfg.write_candidates:
        store = YAMLCandidateStore(
            cfg.candidate_path,
            max_examples=cfg.max_examples_per_key,
        )
        debug["candidate_path"] = cfg.candidate_path
        debug["candidate_mode"] = "spans"

        filtered_oov = [s for s in oov_spans if keep_candidate(getattr(s, "text", ""))]

        debug["candidate_oov_before_filter"] = len(oov_spans)
        debug["candidate_oov_after_filter"] = len(filtered_oov)

        if filtered_oov:
            # ✅ 확정 OOV만 YAML에 저장
            updated = store.update_from_spans(filtered_oov)
            debug["candidate_records"] = len(updated)
        else:
            # ❌ 확정 OOV 없음 → rejected 후보로 누적
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
