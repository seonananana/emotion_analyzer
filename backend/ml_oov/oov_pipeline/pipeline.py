from __future__ import annotations

from typing import Any, Dict, List, Optional
from types import SimpleNamespace

from .yaml_candidates_store import YAMLCandidateStore
from .config import load_config
from .postprocess import apply_conf_threshold, nms_char_spans, remove_known_overlaps
from .span_recover import recover_spans
from .types import LexiconMatcher, LexiconSpan, PipelineResult, PredictTokens
from .filters_ko import keep_candidate


def _wrap_token_dicts_as_objects(tokens: List[Dict[str, Any]]) -> List[Any]:
    """
    recover_spans()가 token.label 처럼 속성 접근을 사용하므로,
    predictor가 dict를 반환하면 SimpleNamespace로 변환해준다.
    """
    objs = []
    for t in tokens:
        # 키가 없을 수 있으니 기본값도 보강
        # (recover_spans가 쓰는 필드: token/start/end/label/confidence 등)
        tt = dict(t)
        if "conf" in tt and "confidence" not in tt:
            tt["confidence"] = tt["conf"]
        objs.append(SimpleNamespace(**tt))
    return objs


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

    # 1) KNOWN span 확보
    if lexicon_spans is None:
        if lexicon_matcher is None:
            lexicon_spans = []
        else:
            lexicon_spans = lexicon_matcher(text)

    # 2) 토큰 단위 예측(A)
    preds_raw = predictor(text)

    # ✅ predictor 출력 형태 어댑팅
    # case A) list[dict] 또는 list[TokenObj] 형태
    if isinstance(preds_raw, list):
        # list 요소가 dict면 객체로 변환 (recover_spans 호환)
        if preds_raw and isinstance(preds_raw[0], dict):
            tokens_obj = _wrap_token_dicts_as_objects(preds_raw)  # type: ignore[arg-type]
            preds = SimpleNamespace(text=text, tokens=tokens_obj)
            preds_tokens_dict = preds_raw  # 저장(후보)용으로 원본 dict 유지
            debug_pred_shape = "list[dict]"
        else:
            # 이미 객체 리스트일 수도 있음
            preds = SimpleNamespace(text=text, tokens=preds_raw)
            preds_tokens_dict = None
            debug_pred_shape = "list[obj]"
    else:
        # case B) predictor가 (text/tokens 속성을 가진 객체) 반환
        preds = preds_raw
        preds_tokens_dict = None
        debug_pred_shape = type(preds_raw).__name__

    # 3) BIO → 문자 단위 span 복원
    model_spans = recover_spans(
        preds,
        target_prefixes=target_labels,
        source="koelectra",
    )

    # 4) 후처리 순서: NMS → KNOWN 제외 → confidence threshold
    model_spans_nms = nms_char_spans(model_spans)
    oov_spans = remove_known_overlaps(model_spans_nms, lexicon_spans)
    oov_spans = apply_conf_threshold(oov_spans, cfg.conf_threshold)

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
    }

    # 5) 후보 누적 저장
    if cfg.write_candidates:
        store = YAMLCandidateStore(cfg.candidate_path, max_examples=cfg.max_examples_per_key)
        debug["candidate_path"] = cfg.candidate_path

        # ✅ 최종 OOV span만 저장 (known 제거 + threshold 적용된 결과)
        filtered_oov = [s for s in oov_spans if keep_candidate(getattr(s, "text", ""))]

        debug["candidate_mode"] = "spans"
        debug["candidate_oov_before_filter"] = len(oov_spans)
        debug["candidate_oov_after_filter"] = len(filtered_oov)

        if filtered_oov:
            updated = store.update_from_spans(filtered_oov)
            debug["candidate_records"] = len(updated)
        else:
            debug["candidate_records"] = 0
            debug["candidate_reason"] = "no_oov_spans_after_filter"
    else:
        debug["candidate_path"] = None
        debug["candidate_mode"] = "disabled"

    return PipelineResult(
        oov_spans=oov_spans,
        model_spans=model_spans_nms,
        lexicon_spans=lexicon_spans,
        debug=debug,
    )
