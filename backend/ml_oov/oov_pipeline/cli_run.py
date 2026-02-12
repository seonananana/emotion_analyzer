#  cli_run.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .adapters import make_mloov_predictor
from .pipeline import run_oov_pipeline
from .lexicon_matcher import build_lexicon_matcher


# ✅ FastAPI(UI Quick Test)에서 import해서 쓸 predictor factory
def build_predictor(
    ckpt: str = "models/koelectra_oov/checkpoint-114",
    device: str = "cpu",
):
    """
    UI(/ui/oov_run)에서도 CLI와 동일한 predictor를 쓰기 위한 factory.
    routes_oov.py가 이 함수를 import해서 predictor(text) callable을 얻는다.
    """
    return make_mloov_predictor(ckpt_dir=ckpt, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--no_write", action="store_true", help="후보 파일 저장을 비활성화한다.")
    ap.add_argument("--ckpt", default="models/koelectra_oov/checkpoint-114", help="A 모델 체크포인트 디렉토리")
    ap.add_argument("--threshold", type=float, default=None, help="conf threshold override (예: 0.0)")
    ap.add_argument(
        "--lexicon",
        default="backend/data/부정_감성_사전_완전판.yaml",
        help="부정 감성 사전 YAML 경로(사전 제외 적용)",
    )
    args = ap.parse_args()

    predictor = make_mloov_predictor(ckpt_dir=args.ckpt, device="cpu")

    overrides = {}
    if args.no_write:
        overrides["write_candidates"] = False
    if args.threshold is not None:
        overrides["conf_threshold"] = float(args.threshold)

    LEXICON_YAML = args.lexicon
    lexicon_matcher = build_lexicon_matcher(LEXICON_YAML)

    res = run_oov_pipeline(
        args.text,
        predictor=predictor,
        lexicon_matcher=lexicon_matcher,
        config_overrides=overrides or None,
    )

    print(
        json.dumps(
            {
                "oov_spans": [asdict(s) for s in res.oov_spans],
                "model_spans": [asdict(s) for s in res.model_spans],
                "lexicon_spans": [asdict(s) for s in res.lexicon_spans],
                "debug": res.debug,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
