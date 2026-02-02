# backend/app/routes_analysis.py
from __future__ import annotations
import logging
from typing import Any, Dict, Literal, Union
from fastapi import APIRouter
from pydantic import BaseModel, Field
from backend.services.analysis_service import analyze_diary
from backend.exceptions import StudentDataError, AnalysisError

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------
# 요청(Request) 스키마
# ---------------------------

class AnalyzeRequest(BaseModel):
    name: str = Field(..., description="학생 이름")
    age: int = Field(..., description="학생 나이")
    gender: str = Field(..., description="학생 성별 (예: '남', '여')")
    text: str = Field(..., description="분석할 일기 원문")
    use_llm: bool = Field(
        True,
        description="LLM 요약 사용 여부 (True: 규칙+LLM, False: 규칙+폴백요약)",
    )

# ---------------------------
# 응답(Response) 스키마
# ---------------------------

class StudentInfo(BaseModel):
    name: str
    age: int
    gender: str


class DetailInfo(BaseModel):
    text_length: int
    negative_word_count: int
    negative_word_frequency: int


class AnalyzeResult(BaseModel):
    """
    legacy_bridge.run_legacy_pipeline_single_diary() 가 반환하는
    dict 구조를 타입으로 표현한 모델.
    """
    student: StudentInfo
    summary: str
    negative_ratio: float | None = None
    top_keywords: Any
    scores: Dict[str, float] | None = None
    detail: DetailInfo


class AnalyzeSuccessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    use_llm: bool
    result: AnalyzeResult


class AnalyzeErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error_type: str
    message: str


# ---------------------------
# 라우트
# ---------------------------

@router.post(
    "/analyze",
    response_model=Union[AnalyzeSuccessResponse, AnalyzeErrorResponse],
)
async def analyze_diary_route(req: AnalyzeRequest):
    """
    단일 일기 분석 API.

    - 입력: 학생 정보 + 일기 텍스트 + use_llm 옵션
    - 출력: 규칙 기반 분석 결과 + (옵션) LLM 요약
    """
    try:
        result_dict = analyze_diary(
            name=req.name,
            age=req.age,
            gender=req.gender,
            text=req.text,
            use_llm=req.use_llm,
        )

        # result_dict 는 legacy_bridge 에서 만든 dict 그대로
        return AnalyzeSuccessResponse(
            use_llm=req.use_llm,
            result=result_dict,
        )

    except StudentDataError as e:
        logger.warning("학생 데이터 오류: %s", e)
        return AnalyzeErrorResponse(
            error_type="student_data_error",
            message=str(e),
        )

    except AnalysisError as e:
        logger.error("분석 오류: %s", e)
        return AnalyzeErrorResponse(
            error_type="analysis_error",
            message=str(e),
        )

    except Exception as e:
        logger.exception("예상치 못한 내부 오류")
        return AnalyzeErrorResponse(
            error_type="internal_error",
            message="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
