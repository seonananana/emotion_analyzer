# backend/infra/yaml_io.py
from pathlib import Path
from typing import Any
import yaml

def load_yaml(path: Path) -> Any:
    """YAML 파일을 로드하여 파이썬 객체로 반환한다."""
    if not path.exists():
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, data: Any) -> None:
    """파이썬 객체를 YAML 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)