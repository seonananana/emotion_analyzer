# backend/core/config.py
from pathlib import Path
import os

from dotenv import load_dotenv  # requirements.txt에 python-dotenv 있음 가정

# 프로젝트 루트 디렉토리 (emotion_analyzer/)
BASE_DIR = Path(__file__).resolve().parents[2]

# 프론트엔드 템플릿 / 정적 파일 경로
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"

# .env 로딩
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# 로그 레벨 (.env의 LOG_LEVEL로 조절, 기본 INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS 설정
# - .env 에 CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:8000" 처럼 넣으면 그 값 사용
# - 없으면 기본으로 전부 허용(["*"])
_cors_raw = os.getenv("CORS_ORIGINS", "")
if _cors_raw:
    CORS_ORIGINS = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    CORS_ORIGINS = ["*"]
