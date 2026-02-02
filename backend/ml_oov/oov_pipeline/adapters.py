from __future__ import annotations

from typing import Callable, List

from backend.ml_oov.predictor import OOVTokenPredictor
from .types import TokenPred, TokenPredictions


def make_mloov_predictor(
    ckpt_dir: str,
    device: str = "cpu",
    max_length: int = 256,
    sentence_split: bool = True,
) -> Callable[[str], TokenPredictions]:
    """
    ml_oov predictor -> oov_pipeline TokenPredictions 어댑터.

    ⚠️ 중요:
    - 라벨을 절대 바꾸지 말 것 (B-NEG_OOV / I-NEG_OOV 그대로 유지)
    - 파이프라인 span recover는 BIO + base label을 그대로 사용한다.
    """
    model = OOVTokenPredictor(
        ckpt_dir=ckpt_dir,
        device=device,
        max_length=max_length,
        sentence_split=sentence_split,
    )

    def _predict(text: str) -> TokenPredictions:
        raw = model.predict_tokens(text)

        toks: List[TokenPred] = []
        for d in raw:
            toks.append(
                TokenPred(
                    token=str(d.get("token", "")),
                    label=str(d.get("label", "O")),  # ✅ 라벨 변조 금지
                    confidence=float(d.get("confidence", d.get("conf", 0.0)) or 0.0),
                    start=int(d.get("start", 0) or 0),
                    end=int(d.get("end", 0) or 0),
                )
            )

        toks.sort(key=lambda x: (x.start, x.end))
        return TokenPredictions(text=text, tokens=toks, meta={"ckpt": ckpt_dir})

    return _predict
