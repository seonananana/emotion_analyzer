# backend/infra/paths.py
from pathlib import Path
from backend.core.config import BASE_DIR

DATA_DIR   = BASE_DIR / "backend" / "data"
CONF_DIR   = BASE_DIR / "backend" / "conf"
OUTPUT_DIR = BASE_DIR / "output"

NEGATIVE_LEXICON_PATH = DATA_DIR / "부정_감성_사전_완전판.yaml"
STUDENT_DATA_PATH     = DATA_DIR / "학생_부정경험_글모음.yaml"

PROMPTS_SYSTEM_PATH   = CONF_DIR / "instruct" / "NA-system-prompt-for-LM.txt"
PROMPTS_USER_PATH     = CONF_DIR / "instruct" / "NA-user-prompt-for-LM.txt"

KEY_PATH = CONF_DIR / "key-togetherai.txt"

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR