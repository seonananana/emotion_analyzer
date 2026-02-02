# backend/infra/output_repo.py
from pathlib import Path
from typing import Dict, Any
from datetime import datetime  
import re

from backend.infra.paths import ensure_output_dir
from backend.infra.yaml_io import save_yaml

def save_student_analysis_yaml(student_id: int, student_name: str, analysis_result: Dict[str, Any]) -> Path:
    """
    ✅ analyze_all_students.py 의 main() 안에서
       student_{id:03d}_{name}_분석결과.yaml 저장하던 코드를 분리.
    """
    output_dir = ensure_output_dir()
    filename = f"student_{student_id:03d}_{student_name}_분석결과.yaml"
    path = output_dir / filename
    save_yaml(path, analysis_result)
    return path


def _slugify_name(name: str) -> str:
    """파일명에 안전하게 쓸 수 있도록 이름 정리."""
    if not name:
        return "학생"
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^\w\-가-힣]", "", s)
    return s or "학생"


def save_web_diary_yaml(
    name: str,
    analysis_result: Dict[str, Any],
) -> Path:
    """
    웹에서 단일 일기 분석할 때 결과를 YAML로 저장.

    예: output/web_20241128_143530_김희연_분석결과.yaml
    """
    output_dir = ensure_output_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _slugify_name(name)
    filename = f"web_{ts}_{safe_name}_분석결과.yaml"
    path = output_dir / filename
    save_yaml(path, analysis_result)
    return path