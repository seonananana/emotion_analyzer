# backend/infra/student_repo.py
from pathlib import Path
import yaml
from typing import Any, Dict, List
from backend.infra.paths import STUDENT_DATA_PATH

def load_all_students() -> List[Dict[str, Any]]:
    """학생 글 모음 파일에서 모든 학생 데이터를 추출"""
    with open(STUDENT_DATA_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data.get("student_negative_experiences", {}).get("students", [])

def load_student_text_mapping() -> Dict[int, str]:
    """학생 ID와 원본 텍스트 매핑"""

    with open(STUDENT_DATA_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    text_mapping = {} # 학생 ID -> 원본 텍스트 매핑
    for student in data.get("student_negative_experiences", {}).get("students", []): # 학생별 반복
        student_id = student.get("id")
        text = student.get("negative_experience", "") # 원본 텍스트
        if student_id and text: # id, 부정상황 있는 경우 추가
            text_mapping[student_id] = text

    return text_mapping
