# backend/infra/prompts.py
from functools import lru_cache
from pathlib import Path
from backend.infra.paths import PROMPTS_SYSTEM_PATH, PROMPTS_USER_PATH
from backend.exceptions import ConfigError  # 이미 있는 예외 타입 쓴다고 가정

@lru_cache
def load_system_prompt() -> str:
    """NA-system-prompt-for-LM.txt 내용을 읽어 시스템 프롬프트로 반환."""
    system_file = PROMPTS_SYSTEM_PATH
    if not system_file.exists():
        raise ConfigError("system 프롬프트 파일이 없습니다.")
    return system_file.read_text(encoding="utf-8").strip()


@lru_cache
def load_user_prompt() -> str:
    """NA-user-prompt-for-LM.txt 내용을 읽어 유저 프롬프트 템플릿으로 반환."""
    user_file = PROMPTS_USER_PATH
    if not user_file.exists():
        raise ConfigError("user 프롬프트 파일이 없습니다.")
    return user_file.read_text(encoding="utf-8").strip()