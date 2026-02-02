# # backend/domain/models.py
# # 추후 db 연동 시 수정 및 적용 필요
# # 1차 구현때 사용 X

# from typing import Dict, List, Optional
# from pydantic import BaseModel

# class Student(BaseModel):
#     id: Optional[int] = None
#     name: str
#     age: Optional[int] = None
#     gender: Optional[str] = None
#     precious_thing: Optional[str] = None

# class DiaryAnalysisResult(BaseModel):
#     """사전 기반 분석 결과 요약"""
#     emotion_scores: Dict[str, float]         # 감정별 점수
#     emotion_percentages: Dict[str, float]    # 감정별 비율(0~1 또는 0~100)
#     total_negative_words_found: int
#     total_negative_word_frequency: int
#     text_length: int
#     negative_ratio: float                    # 부정 문장 비율 등
#     top_keywords: List[str]                  # 많이 등장한 부정 단어

# class LlmAnalysisResult(BaseModel):
#     """LLM 심화 분석 결과"""
#     summary: str                             # 주요 요약
#     detailed_comment: str                    # 상세 피드백
#     risk_level: Optional[str] = None         # "low"/"medium"/"high" 등 (있으면)

# class FullAnalysisResult(BaseModel):
#     """사전 + LLM 통합 결과 (유스케이스/웹에서 사용)"""
#     student: Student
#     lexicon: DiaryAnalysisResult
#     llm: LlmAnalysisResult


# class NegativeLexiconEntry(BaseModel):
#     id: int
#     word: str
#     emotions: Dict[str, float]  # 예: {"불안": 2.0, "분노": 1.0}
#     total: float