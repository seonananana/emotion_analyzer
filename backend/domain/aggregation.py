from __future__ import annotations
from typing import Dict, List, Any

def build_negative_ratio(result: Dict[str, Any]) -> float:
    """텍스트 길이에 대한 부정 단어 빈도 비율을 계산한다.

    예시 구현이므로, 프로젝트 정책에 맞게 공식은 자유롭게 조정해도 된다.
    부정 비율 계산 방식, 감정별 점수 가중치, 위험도 라벨 기준 등은 필요에 따라 변경 가능.
    """
    text_length = result.get("text_length", 0) or 0
    total_negative = result.get("total_negative_word_frequency", 0) or 0

    if text_length <= 0:
        return 0.0

    return total_negative / float(text_length)

def extract_top_keywords(result: Dict[str, Any], max_items: int = 5) -> List[str]:
    """빈도가 높은 부정 단어를 상위 N개까지 추출한다.

    반환 형식 예시: ["불안(3)", "외롭다(2)", ...]
    """
    word_freq: Dict[str, int] = result.get("word_frequency") or {}

    items = sorted(
        word_freq.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = items[:max_items]
    return [f"{word}({count})" for word, count in top]

def build_scores(result: Dict[str, Any]) -> Dict[str, float]:
    """감정 카테고리별 점수/비율 딕셔너리를 생성한다.

    기본적으로는 emotion_percentages 를 그대로 사용하되,
    필요하면 emotion_distribution 을 이용해 재계산해도 된다.
    """
    emotion_percentages: Dict[str, float] = result.get("emotion_percentages") or {}
    if emotion_percentages:
        # 이미 비율(%)이나 0~1 값으로 정규화되어 있다고 가정하고 그대로 반환
        return dict(emotion_percentages)

    # 백업: emotion_distribution 값의 합을 1로 정규화해서 비율로 만든다.
    emotion_distribution: Dict[str, float] = result.get("emotion_distribution") or {}
    total = sum(emotion_distribution.values())
    if total <= 0:
        return {k: 0.0 for k in emotion_distribution.keys()}

    return {k: v / total for k, v in emotion_distribution.items()}