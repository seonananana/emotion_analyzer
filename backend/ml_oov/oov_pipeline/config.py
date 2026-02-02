from __future__ import annotations

import os
from .types import PipelineConfig


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_config() -> PipelineConfig:
    """
    ✅ MLOOV 환경변수 기준으로 OOV 파이프라인 설정 로드 (단일 표준)

    환경변수
    - MLOOV_ENABLED (default: 1)
    - MLOOV_THRESHOLD (default: 0.30)
    - MLOOV_WRITE_CANDIDATES (default: 1)
    - MLOOV_CANDIDATE_PATH (default: backend/data/oov_candidates.yaml)
    - MLOOV_MAX_EXAMPLES (default: 5)
    """
    enabled = _env_bool("MLOOV_ENABLED", True)
    conf_threshold = _env_float("MLOOV_THRESHOLD", 0.30)
    write_candidates = _env_bool("MLOOV_WRITE_CANDIDATES", True)

    candidate_path = os.getenv(
        "MLOOV_CANDIDATE_PATH",
        "backend/data/oov_candidates.yaml",
    )

    try:
        max_examples = int(os.getenv("MLOOV_MAX_EXAMPLES", "5"))
    except ValueError:
        max_examples = 5

    return PipelineConfig(
        enabled=enabled,
        conf_threshold=conf_threshold,
        write_candidates=write_candidates,
        candidate_path=candidate_path,
        max_examples_per_key=max_examples,
    )
