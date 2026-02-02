from __future__ import annotations
from typing import Optional, Dict, Any
from backend.domain.analyzer import NegativeEmotionAnalyzer
from backend.domain.aggregation import (
    build_negative_ratio,
    build_scores,
    extract_top_keywords,
)
from backend.infra.llm_client import LLMError, summarize_diary


class StudentDataError(ValueError):
    """웹/입력 데이터 검증 실패용 예외."""


def _validate_input(name: str, text: str, age: Optional[int] = None) -> None:
    if not name or not name.strip():
        raise StudentDataError("이름을 입력해 주세요.")
    if not text or not text.strip():
        raise StudentDataError("분석할 텍스트를 입력해 주세요.")
    if age is not None and age <= 0:
        raise StudentDataError("나이는 1 이상이어야 합니다.")


def _build_fallback_summary(
    profile: Dict[str, Any],
    base_result: Dict[str, Any],
    negative_ratio: float,
    top_keywords: list[str],
) -> str:
    """LLM 을 사용하지 못할 때 간단 요약을 만드는 규칙 기반 버전."""
    parts: list[str] = []

    name = profile.get("name", "학생")
    parts.append(
        f"{name} 학생의 글에서 감지된 부정 단어 비율은 "
        f"{negative_ratio:.3f} 수준입니다."
    )

    if top_keywords:
        parts.append(f"주요 부정 키워드는 {', '.join(top_keywords)} 입니다.")

    emotion_percentages = base_result.get("emotion_percentages") or {}
    if emotion_percentages:
        sorted_emotions = sorted(
            emotion_percentages.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_emotion, score = sorted_emotions[0]
        parts.append(f"가장 두드러지는 감정은 '{top_emotion}' 입니다.")

    return " ".join(parts)

def run_single_diary_usecase(
    name: str,
    age: Optional[int],
    gender: Optional[str],
    precious_thing: Optional[str],
    text: str,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """일기/자기표현 글 한 편에 대한 전체 분석 유즈케이스."""
    # 1) 입력 검증
    _validate_input(name=name, text=text, age=age)

    # 2) 분석기 생성
    analyzer = NegativeEmotionAnalyzer()

    # 3) 규칙 기반 1차 분석
    base_result: Dict[str, Any] = analyzer.analyze_text(text)
    
    print("DEBUG word_frequency =", base_result.get("word_frequency"))
    print("DEBUG total_negative_words_found =", base_result.get("total_negative_words_found"))
    print("DEBUG total_negative_word_frequency =", base_result.get("total_negative_word_frequency"))

    # 4) 2차 지표 계산
    negative_ratio = build_negative_ratio(base_result)
    top_keywords = extract_top_keywords(base_result)
    scores = build_scores(base_result)

    profile: Dict[str, Any] = {
        "id": None,
        "name": name,
        "age": age,
        "gender": gender,
        "precious_thing": precious_thing,
    }

    # 5) (옵션) LLM 요약
    if use_llm:
        try:
            summary = summarize_diary(profile, base_result)
        except LLMError:
            summary = _build_fallback_summary(
                profile, base_result, negative_ratio, top_keywords
            )
    else:
        summary = _build_fallback_summary(
            profile, base_result, negative_ratio, top_keywords
        )

    return {
        "student": profile,
        "base_result": base_result,
        "negative_ratio": negative_ratio,
        "top_keywords": top_keywords,
        "scores": scores,
        "summary": summary,
    }