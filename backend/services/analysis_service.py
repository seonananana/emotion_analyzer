from __future__ import annotations

import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import date
from typing import Any, Dict, Optional
from contextlib import contextmanager

from backend.domain.analyzer import NegativeEmotionAnalyzer
from backend.infra.output_repo import save_web_diary_yaml
from backend.services.legacy_bridge import run_legacy_pipeline_single_diary

from backend.ml_oov.oov_pipeline.adapters import make_mloov_predictor
from backend.ml_oov.oov_pipeline.pipeline import run_oov_pipeline
from backend.ml_oov.oov_pipeline.config import load_config

# lexicon matcher: 1) analyzer adapter 우선, 2) yaml matcher fallback
try:
    from backend.ml_oov.oov_pipeline.lexicon_adapter import analyzer_lexicon_matcher as _lexicon_matcher
except Exception:
    from backend.ml_oov.oov_pipeline.lexicon_matcher import build_lexicon_matcher
    _LEXICON_YAML = os.getenv("NEG_LEXICON_YAML", "backend/data/부정_감성_사전_완전판.yaml")
    _lexicon_matcher = build_lexicon_matcher(_LEXICON_YAML)


# =========================
# ✅ 운영 설정 (단일 표준: MLOOV_*)
# =========================
_CFG = load_config()

_MLOOV_CKPT = os.getenv("MLOOV_CKPT", "models/koelectra_oov/checkpoint-114")
_MLOOV_DEVICE = os.getenv("MLOOV_DEVICE", "cpu")
_MLOOV_SENT_SPLIT = os.getenv("MLOOV_SENTENCE_SPLIT", "1").strip().lower() not in ("0", "false", "no")

# ✅ predictor는 전역 1회 로드 (요청마다 로드 금지)
_MLOOV_PRED = make_mloov_predictor(
    ckpt_dir=_MLOOV_CKPT,
    device=_MLOOV_DEVICE,
    sentence_split=_MLOOV_SENT_SPLIT,
)

# 모듈 전역 analyzer
_analyzer = NegativeEmotionAnalyzer()


# =========================
# ✅ 후보 누적 파일락 (프로세스 내 + 프로세스 간)
# =========================
_process_lock = threading.Lock()

@contextmanager
def _file_lock(lock_path: str):
    """
    간단한 cross-platform 파일락.
    - Windows: msvcrt.locking
    - Unix: fcntl.flock
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")

    try:
        if os.name == "nt":
            import msvcrt
            # 1 byte 잠금
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _obj_to_dict(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if is_dataclass(x):
        return asdict(x)
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return x


def analyze_diary(
    name: str,
    age: int,
    gender: str,
    text: str,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    단일 일기 분석 서비스 진입점.
    - legacy 결과 유지
    - lexicon 상세 분석 유지
    - OOV 파이프라인 실행 + 후보 YAML 누적
    """

    # 1) 기존 파이프라인 결과
    result = run_legacy_pipeline_single_diary(
        name=name,
        age=age,
        gender=gender,
        text=text,
        use_llm=use_llm,
    )

    # 2) lexicon 기반 상세 분석
    neg_detail = _analyzer.analyze_text(text, student_name=name)
    text_comment = _analyzer.generate_text_analysis_comment(neg_detail, text)
    llm_summary = _analyzer.generate_negative_word_analysis_summary(neg_detail)

    # 3) OOV 파이프라인
    #    - 사전 known span 생성
    #    - 모델 span에서 known overlap 제거
    #    - 남은 것만 oov_spans
    try:
        # ✅ 후보 누적이 켜져 있으면, 동일 프로세스/다중 프로세스 동시성 방어
        lockfile = f"{_CFG.candidate_path}.lock"

        if _CFG.write_candidates:
            with _process_lock:
                with _file_lock(lockfile):
                    oov_res = run_oov_pipeline(
                        text=text,
                        predictor=_MLOOV_PRED,
                        lexicon_matcher=_lexicon_matcher,
                        config_overrides={
                            "enabled": _CFG.enabled,
                            "conf_threshold": _CFG.conf_threshold,
                            "write_candidates": _CFG.write_candidates,
                            "candidate_path": _CFG.candidate_path,
                            "max_examples_per_key": _CFG.max_examples_per_key,
                        },
                    )
        else:
            oov_res = run_oov_pipeline(
                text=text,
                predictor=_MLOOV_PRED,
                lexicon_matcher=_lexicon_matcher,
                config_overrides={
                    "enabled": _CFG.enabled,
                    "conf_threshold": _CFG.conf_threshold,
                    "write_candidates": _CFG.write_candidates,
                    "candidate_path": _CFG.candidate_path,
                    "max_examples_per_key": _CFG.max_examples_per_key,
                },
            )

        result["oov"] = {
            "ckpt": _MLOOV_CKPT,
            "threshold": _CFG.conf_threshold,
            "write_candidates": _CFG.write_candidates,
            "candidate_path": _CFG.candidate_path,
            "oov_spans": [_obj_to_dict(s) for s in (oov_res.oov_spans or [])],
            "model_spans": [_obj_to_dict(s) for s in (oov_res.model_spans or [])],
            "lexicon_spans": [_obj_to_dict(s) for s in (oov_res.lexicon_spans or [])],
            "debug": _obj_to_dict(oov_res.debug),
        }

    except Exception as e:
        result["oov_error"] = str(e)

    # 4) 메타 정보
    neg_detail = dict(neg_detail)
    neg_detail.setdefault("student_name", name)
    neg_detail.setdefault("analysis_date", date.today().isoformat())
    neg_detail.setdefault("text_length", len(text))

    yaml_payload = {
        "negative_word_analysis": neg_detail,
        "text_analysis_comment": text_comment,
        "llm_prompt_summary": llm_summary,
        "oov": result.get("oov", None),
        "oov_error": result.get("oov_error", None),
    }

    # 5) 웹 일기 YAML 저장
    try:
        save_web_diary_yaml(name=name, analysis_result=yaml_payload)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("YAML 저장 실패: %s", e)

    # 6) 프론트 result에도 유지
    result["negative_word_analysis"] = neg_detail
    result["text_analysis_comment"] = text_comment
    result["llm_prompt_summary"] = llm_summary

    return result
