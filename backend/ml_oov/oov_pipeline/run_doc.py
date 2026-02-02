from __future__ import annotations

import os
import glob
import pandas as pd
from typing import Optional

from backend.ml_oov.predictor import OOVTokenPredictor
from backend.ml_oov.oov_pipeline.pipeline import run_oov_pipeline
from backend.ml_oov.oov_pipeline.lexicon_matcher import build_lexicon_matcher

LEXICON_PATH = "backend/data/부정_감성_사전_완전판.yaml"

def _resolve_xlsx_path(xlsx_path: str) -> str:
    # 그대로 존재하면 OK
    if os.path.exists(xlsx_path):
        return xlsx_path

    # ./ 붙여서 재시도
    if not xlsx_path.startswith("./"):
        p2 = "./" + xlsx_path
        if os.path.exists(p2):
            return p2

    # backend/data/*.xlsx 안내
    candidates = glob.glob("backend/data/*.xlsx") + glob.glob("./backend/data/*.xlsx")
    msg = (
        f"XLSX 파일을 찾을 수 없음: {xlsx_path}\n"
        f"예: 너 파일은 backend/data/25-2-전(24).xlsx 처럼 있어야 함\n"
        f"backend/data 하위 XLSX 후보:\n  - " + "\n  - ".join(candidates[:50])
    )
    raise FileNotFoundError(msg)


def run_xlsx_oov(
    xlsx_path: str,
    text_col: Optional[str] = None,
    *,
    ckpt_dir: str = "models/koelectra_oov",
    device: str = "cpu",
    sentence_split: bool = True,
    conf_threshold: Optional[float] = None,
) -> None:
    """
    XLSX 전체를 읽어
    - 사전에 있는 부정어는 제외(known)
    - 사전에 없는 부정 OOV만 추출
    - oov_candidates.yaml에 누적 저장
    """
    xlsx_path = _resolve_xlsx_path(xlsx_path)

    if not os.path.exists(LEXICON_PATH):
        raise FileNotFoundError(f"lexicon 파일을 찾을 수 없음: {LEXICON_PATH}")

    predictor = OOVTokenPredictor(
        ckpt_dir=ckpt_dir,
        sentence_split=sentence_split,
        device=device,
    )
    lexicon_matcher = build_lexicon_matcher(LEXICON_PATH)

    df = pd.read_excel(xlsx_path)

    # 텍스트 컬럼 자동 추정
    if text_col is None:
        cand_cols = []
        for c in df.columns:
            if df[c].dtype == object:
                series = df[c].dropna().astype(str)
                if len(series) == 0:
                    continue
                avg_len = series.map(len).mean()
                cand_cols.append((avg_len, c))
        cand_cols.sort(reverse=True)
        if not cand_cols:
            raise RuntimeError("텍스트 컬럼을 찾지 못함. text_col을 수동 지정해줘.")
        text_col = cand_cols[0][1]

    print(f"[run_doc] xlsx = {xlsx_path}")
    print(f"[run_doc] text column = {text_col}")
    print(f"[run_doc] rows = {len(df)}")

    total_oov_spans = 0
    for idx, row in df.iterrows():
        text = str(row[text_col]) if row[text_col] is not None else ""
        text = text.strip()
        if not text or text.lower() == "nan":
            continue

        # 문서 split: 줄 단위
        for sent in text.splitlines():
            sent = sent.strip()
            if not sent:
                continue
            res = run_oov_pipeline(
                text=sent,
                predictor=predictor.predict_tokens,
                lexicon_matcher=lexicon_matcher,
                config_overrides=(
                    {"conf_threshold": conf_threshold}
                    if conf_threshold is not None
                    else None
                ),
            )
            total_oov_spans += len(res.oov_spans)

        if (idx + 1) % 50 == 0:
            print(f"[run_doc] processed {idx+1}/{len(df)} | total_oov_spans={total_oov_spans}")

    print("[run_doc] DONE total_oov_spans =", total_oov_spans)
    print("[run_doc] candidates:", os.path.exists("backend/data/oov_candidates.yaml"))