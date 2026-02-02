from __future__ import annotations

from typing import List

from backend.domain.analyzer import NegativeEmotionAnalyzer  # 경로가 다르면 여기만 수정
from backend.ml_oov.oov_pipeline.types import LexiconSpan


_analyzer_singleton = None


def get_analyzer() -> NegativeEmotionAnalyzer:
    global _analyzer_singleton
    if _analyzer_singleton is None:
        _analyzer_singleton = NegativeEmotionAnalyzer()
    return _analyzer_singleton


def analyzer_lexicon_matcher(text: str) -> List[LexiconSpan]:
    """
    기존 NegativeEmotionAnalyzer의 매칭 결과를
    B 파이프라인이 쓰는 LexiconSpan 리스트로 변환.
    """
    analyzer = get_analyzer()
    res = analyzer.analyze_text(text)

    spans: List[LexiconSpan] = []
    detailed = res.get("detailed_word_analysis", {}) or {}

    for original_word, info in detailed.items():
        for occ in info.get("forms", []) or []:
            start = int(occ["start"])
            end = int(occ["end"])
            # matched form을 그대로 text로 쓰는게 맞음(원문 substring이랑 일치)
            matched = text[start:end]
            spans.append(LexiconSpan(start=start, end=end, text=matched, source="lexicon"))

    spans.sort(key=lambda s: (s.start, s.end))
    return spans
