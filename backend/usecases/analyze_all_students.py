# # backend/usecases/analyze_all_students.py
# # 학생 글 모음 파일에서 모든 학생의 부정적 경험을 분석하고 결과를 저장하는 유스케이스
# # 실제로는 사용하지 않음(analyzer.py에서 학생 글 모음 파일을 직접 처리함)
# # 1차 구현때 사용 X

# from typing import Any, Dict, List, Optional
# from backend.domain.analyzer import NegativeEmotionAnalyzer
# from backend.infra.paths import get_lexicon_path
# from backend.infra.student_repo import load_all_students
# from backend.infra.output_repo import save_student_analysis_yaml

# def load_all_students(file_path: str) -> List[Dict[str, Any]]:
#     """학생 글 모음 파일에서 모든 학생 데이터를 추출"""
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = yaml.safe_load(file)
#     return data.get("student_negative_experiences", {}).get("students", [])

# def analyze_student(
#     analyzer: NegativeEmotionAnalyzer, 
#     student_data: Dict[str, Any]
# ) -> Dict[str, Any]:
#     """개별 학생 분석"""
#     student_id = student_data.get("id")
#     student_name = student_data.get("name", "")
#     precious_thing = student_data.get("precious_thing", "")
#     text = student_data.get("negative_experience", "")

#     if not text:
#         print(f"⚠️  학생 ID {student_id} ({student_name}): 텍스트가 없습니다.")
#         return None

#     # 분석 실행
#     result = analyzer.analyze_text(text, student_name)

#     # 소중한 것 정보 추가
#     result["precious_thing"] = precious_thing
#     result["student_id"] = student_id

#     # 코멘트 생성 (규칙 기반 - 전반적 특성과 감정 분포 특성만)
#     comment = analyzer.generate_text_analysis_comment(result, text)

#     # LLM 프롬프트용 요약 생성
#     analysis_summary = analyzer.generate_negative_word_analysis_summary(result)

#     return {
#         "negative_word_analysis": result,
#         "text_analysis_comment": comment,
#         "llm_prompt_summary": analysis_summary,
#     }