# scripts/dump_known_variations.py
from __future__ import annotations

import json
from pathlib import Path

def main():
    out_path = Path("backend/data/known_variations.json")

    # ✅ 여기서만 analyzer import (UI/엑셀 스크립트는 import 안 함)
    try:
        from backend.domain.analyzer import NegativeEmotionAnalyzer
    except Exception as e:
        raise SystemExit(f"[FATAL] cannot import NegativeEmotionAnalyzer: {e}")

    an = NegativeEmotionAnalyzer()
    vars_map = getattr(an, "word_variations", None)
    if not isinstance(vars_map, dict) or not vars_map:
        raise SystemExit("[FATAL] analyzer.word_variations is empty or invalid")

    # variation 폼만 저장 (원형은 값에 있지만 우리는 제외용 set만 필요)
    variations = sorted({k.strip() for k in vars_map.keys() if isinstance(k, str) and k.strip()})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"count": len(variations), "variations": variations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] wrote {len(variations):,} variations -> {out_path}")

if __name__ == "__main__":
    main()
