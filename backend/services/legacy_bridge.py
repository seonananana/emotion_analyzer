# backend/services/legacy_bridge.py
"""
웹 서비스(FastAPI)에서 사용할 단일 일기 분석 브릿지 모듈.

예전에는 레거시 스크립트:
- analyze_all_students.py
- analyze_with_llm.py

를 직접 import 해서 사용했지만,
현재 프로젝트 구조에서는 모두 usecase 계층으로 리팩터링되었기 때문에

    backend.usecases.analyze_single_diary.run_single_diary_usecase

를 호출하는 얇은 어댑터 역할만 수행한다.
"""

from __future__ import annotations
from typing import Any, Dict

from backend.exceptions import AnalysisError, StudentDataError
from backend.usecases.analyze_single_diary import (
    run_single_diary_usecase,
    StudentDataError as UsecaseStudentDataError,
)


def run_legacy_pipeline_single_diary(
    name: str,
    age: int,
    gender: str,
    text: str,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    웹에서 들어온 한 편의 일기를 분석하고, dict 형태로 반환한다.

    파이프라인:
    1) usecase.run_single_diary_usecase(...) 호출
       - 규칙 기반 분석
       - use_llm=True 인 경우 LLM 요약 시도, 실패 시 폴백 요약
       - use_llm=False 인 경우 규칙 기반 + 폴백 요약만 사용
    2) usecase 가 반환한 dict → 프론트에서 쓰기 좋은 dict 로 변환
    """
    try:
        # usecase 결과도 이제 Dict[str, Any] 형태
        summary: Dict[str, Any] = run_single_diary_usecase(
            name=name,
            age=age,
            gender=gender,
            precious_thing=None,  # 웹 폼에는 지금 이 필드가 없으므로 일단 None
            text=text,
            use_llm=use_llm,
        )
    except UsecaseStudentDataError as e:
        # usecase 내부의 StudentDataError 를
        # 웹 레이어에서 사용하는 StudentDataError 로 변환
        raise StudentDataError(str(e)) from e
    except NotImplementedError as e:
        # NegativeEmotionAnalyzer.analyze_text 가 아직 구현되지 않은 상태였을 때용
        raise AnalysisError(
            "규칙 기반 분석 로직이 아직 구현되지 않았습니다. "
            "domain.NegativeEmotionAnalyzer.analyze_text 를 먼저 구현해야 합니다."
        ) from e
    except Exception as e:  # pragma: no cover
        # 그 밖의 모든 예외는 AnalysisError 로 래핑
        raise AnalysisError(f"일기 분석 중 알 수 없는 오류가 발생했습니다: {e}") from e

    # usecase 가 반환한 dict 에서 부분 추출
    student = summary.get("student") or {}
    base = summary.get("base_result") or {}

    result: Dict[str, Any] = {
        "student": {
            "name": student.get("name"),
            "age": student.get("age"),
            "gender": student.get("gender"),
        },
        "summary": summary.get("summary"),
        "negative_ratio": summary.get("negative_ratio"),
        "top_keywords": summary.get("top_keywords"),
        "scores": summary.get("scores"),
        # 인지사항 출력 왼쪽 박스에 쓸 상세 정보
        "detail": {
            "text_length": base.get("text_length"),
            "negative_word_count": base.get("total_negative_words_found"),
            "negative_word_frequency": base.get("total_negative_word_frequency"),
        },
    }
    return result