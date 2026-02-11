# scripts/extract_oov_from_excel.py
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import pandas as pd
import yaml

from backend.ml_oov.oov_pipeline.lexicon_matcher import build_lexicon_matcher

# ====== 설정 ======
LEXICON_YAML = "backend/data/부정_감성_사전_완전판.yaml"
OUT_YAML = "backend/data/oov_candidates.yaml"

# ====== 부정 패턴 ======
NEG_PATTERNS = [
    r"멘붕", r"현타", r"빡침", r"빡치", r"열받", r"킹받",
    r"개빡", r"개같", r"존나", r"ㅈㄴ", r"좆같", r"씨발", r"시발",
    r"미치겠", r"죽겠", r"토나오", r"숨막", r"답답", r"짜증",
    r"불안", r"우울", r"공포", r"무섭", r"싫어", r"힘들", r"괴롭",
]
NEG_RE = re.compile("|".join(NEG_PATTERNS))

# 토큰 후보 추출
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣ㄱ-ㅎㅏ-ㅣ]{2,}")

# ====== STOPWORDS ======
STOPWORDS = {
    "그냥","진짜","너무","완전","계속","정말","근데","그래서","하지만","그리고",
    "오늘","내일","어제","요즘","사람","때문","정도","이런","저런","그런",
    "있는","없는","같다","했다","한다","되다","하다","것","수","좀",
    "동안","그동안","사이","때","경우","상황","문제",
}

# ====== PATCH B: 노이즈 컷 규칙 ======

# 문법적 보조동사/기능어 계열 컷(너무 흔해서 OOV 후보로 의미 없음)
AUX_PREFIXES = (
    "못하","못했","못한","못할","못해","못하는","못했습니다","못했습니",
    "안되","안돼","안되었","안됐","안되면","안되는","안된",
)

# 의미 없는 접미/조사
BAD_SUFFIXES = (
    "에서","에게","으로","까지","부터","처럼","같이","조차","마저",
    "라도","때문","동안","그동안",
    "안에","안으로","밖에",
)

# 중립 코어(걸리면 바로 제외)
NEUTRAL_CORE = {
    "동안","그동안","사이","때","경우","상황","문제","시간",
    "사람","생각","마음","정도","부분",
    "안","안에",
    "안전","안전한","안정","안정적",
    "제안","편안","집안",  # ✅ “안” 때문에 자주 튀는 것들
}

BAD_REGEXES = [
    re.compile(r"^못.{0,6}$"),
    re.compile(r"^안.{0,6}$"),
    re.compile(r"^[0-9]+$"),
]

# ====== 정규화 ======
def _normalize(t: str) -> str:
    t = (t or "").strip()
    t = t.replace("\u00a0", "")  # NBSP
    t = re.sub(r"\s+", "", t)
    return t

def normalize_for_compare(t: str) -> str:
    """
    저장 키를 좀 더 ‘의미 단위’로 통합하려는 정규화.
    - 공백 제거
    - 너무 공격적인 어미/활용 일부 제거(파편 후보 줄이기)
    """
    t = _normalize(t)

    for suf in (
        "이었다","였다","였다가",
        "었다","았다","었","았",
        "습니다","어요","에요",
        "서","고",
        "는","은","을","게","던","기에",
        "했다","했","하는","한","할",
        "지","지만","니까","면","으면",
    ):
        if t.endswith(suf) and len(t) > len(suf) + 1:
            t = t[:-len(suf)]
            break

    return t

# (사용 편의) _normalize를 normalize_for_compare로 쓰고 싶으면 아래 한 줄만 바꾸면 됨
# _normalize = normalize_for_compare

# ====== 사전 변형 로딩 ======
def build_known_variations_set() -> Set[str]:
    """
    ✅ 가장 안전한 방법:
    lexicon_adapter.get_analyzer()는 이미 파이프라인에서 쓰는 singleton이라
    여기서 가져오면 circular import를 피할 수 있음.
    """
    try:
        from backend.ml_oov.oov_pipeline.lexicon_adapter import get_analyzer  # ✅ circular 회피 포인트

        an = get_analyzer()
        vars_map = getattr(an, "word_variations", None)
        if isinstance(vars_map, dict) and vars_map:
            return { normalize_for_compare(k) for k in vars_map.keys() if isinstance(k, str) and k.strip() }

        print("[WARN] get_analyzer().word_variations is empty; fallback to yaml-word scan")
    except Exception as e:
        print(f"[WARN] analyzer singleton import failed; fallback to yaml-word scan: {e}")

    # fallback: YAML에서 word 긁어온 뒤 최소 변형 생성
    words = _load_lexicon_words_basic(LEXICON_YAML)
    return _fallback_make_variations(words)

def _load_lexicon_words_basic(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    words: List[str] = []

    # 1) {"negative_emotion_lexicon": {"entries":[{"word":...}]}} 형태 우선
    try:
        for e in data["negative_emotion_lexicon"]["entries"]:
            w = e.get("word")
            if isinstance(w, str) and w.strip():
                words.append(w.strip())
        if words:
            return words
    except Exception:
        pass

    # 2) generic walk
    def walk(x):
        if isinstance(x, dict):
            w = x.get("word")
            if isinstance(w, str) and w.strip():
                words.append(w.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(data)
    return list(dict.fromkeys(words))

def _fallback_make_variations(words: List[str]) -> Set[str]:
    endings = [
        "다","요","어","아","서","고","지","게","는","은","을",
        "었","았","었어","았어","었어요","았어요",
        "던","었던","았던","는데","지만","니까","면",
        "해","해서","했","했다","했어요",
    ]
    out: Set[str] = set()
    for w in words:
        w0 = _normalize(w)
        if not w0:
            continue
        out.add(w0)
        if w0.endswith("다") and len(w0) >= 3:
            stem = w0[:-1]
            out.add(stem)
            for e in endings:
                out.add(stem + e)
    return out

# ====== 마스킹 ======
def mask_known_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    if not spans:
        return text
    spans = sorted(spans)
    out: List[str] = []
    cur = 0
    for s, e in spans:
        out.append(text[cur:s])
        out.append(" " * (e - s))
        cur = e
    out.append(text[cur:])
    return "".join(out)

# ====== 후보 추출 ======
def extract_candidates_from_text(text: str) -> List[str]:
    cands: List[str] = []

    # (A) 패턴 매칭
    for m in NEG_RE.finditer(text):
        cands.append(m.group(0))

    # (B) 토큰 기반
    toks = TOKEN_RE.findall(text)
    for t in toks:
        tt = _normalize(t)
        if len(tt) < 2 or tt in STOPWORDS:
            continue

        # 부정 패턴이 토큰 안에 포함되면 후보
        if NEG_RE.search(tt):
            cands.append(tt)
            continue

        # ✅ “안”은 포함이 아니라 접두일 때만 (편안/제안/집안 제거)
        if tt.startswith(("안", "못")):
            cands.append(tt)
            continue

        # 기타 부정 힌트(‘안’ 제외)
        if any(x in tt for x in ("싫", "힘들", "무섭", "불안", "우울", "답답", "짜증", "괴롭")):
            cands.append(tt)

    return cands

def load_existing_yaml(path: Path) -> Dict:
    if not path.exists():
        return {"records": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"records": []}

def upsert_records(data: Dict, freq: Counter, examples: Dict[str, List[str]], *, min_count: int):
    recs = data.get("records", [])
    idx = {r.get("key"): r for r in recs if isinstance(r, dict) and r.get("key")}

    for k, cnt in freq.most_common():
        if cnt < min_count:
            continue
        ex = examples.get(k, [])[:5]
        if k in idx:
            idx[k]["count"] = int(idx[k].get("count", 0)) + int(cnt)
            merged = list(dict.fromkeys((idx[k].get("examples") or []) + ex))
            idx[k]["examples"] = merged[:10]
        else:
            idx[k] = {"key": k, "count": int(cnt), "examples": ex, "source": "heuristic"}

    data["records"] = sorted(idx.values(), key=lambda r: int(r.get("count", 0)), reverse=True)

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min_count", type=int, default=2)
    args = ap.parse_args()

    xlsx = Path(args.input)
    assert xlsx.exists(), f"not found: {xlsx}"
    assert xlsx.is_file(), f"--input must be a file, got: {xlsx} (did you pass a directory?)"

    matcher = build_lexicon_matcher(LEXICON_YAML)
    known_vars = build_known_variations_set()
    print(f"[INFO] known_variations size = {len(known_vars):,}")

    df_or_dict = pd.read_excel(xlsx, sheet_name=args.sheet, engine="openpyxl")
    if isinstance(df_or_dict, dict):
        sheet = list(df_or_dict.keys())[0]
        df = df_or_dict[sheet]
        print(f"[INFO] sheet not provided. Using first sheet: {sheet}")
    else:
        df = df_or_dict

    df.columns = [str(c) for c in df.columns]

    text_col = args.text_col
    if text_col is None:
        cand = [c for c in df.columns if "부정" in c and ("경험" in c or "응답" in c)]
        text_col = cand[0] if cand else df.columns[0]
        print(f"[INFO] auto text_col = {text_col}")

    rows = df[text_col].dropna().astype(str).tolist()
    if args.limit:
        rows = rows[:args.limit]

    freq = Counter()
    ex: Dict[str, List[str]] = defaultdict(list)

    for text in rows:
        # 1) 사전 매칭 span 마스킹
        lex_spans = matcher(text)
        masked = mask_known_spans(text, [(s.start, s.end) for s in lex_spans])

        # 2) 후보 추출
        cands = extract_candidates_from_text(masked)

    for k in cands:
        kk_raw = _normalize(k)
        kk = normalize_for_compare(kk_raw)

        if not kk or kk in STOPWORDS:
            continue

        # PATCH B: 노이즈 컷 먼저 (빠르게 제거)
        if kk in NEUTRAL_CORE:
            continue
        if kk.startswith(AUX_PREFIXES):
            continue
        if kk.endswith(BAD_SUFFIXES):
            continue
        if any(rx.match(kk) for rx in BAD_REGEXES):
            continue

        # 1) matcher 기준 “완전일치”만 제거 (원형/정확히 사전 표현인 경우)
        m2 = matcher(kk_raw)
        if any(s.start == 0 and s.end == len(kk_raw) for s in m2):
            continue

        # 2) known variations (활용형/변형) 제거
        if kk in known_vars:
            continue

        freq[kk] += 1
        if len(ex[kk]) < 10:
            ex[kk].append(text[:200])

    out_path = Path(OUT_YAML)
    data = load_existing_yaml(out_path)
    upsert_records(data, freq, ex, min_count=args.min_count)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("===== SUMMARY =====")
    print("rows_processed:", len(rows))
    print("unique_candidates:", len(freq))
    print("top10:", freq.most_common(10))
    print("saved:", str(out_path))

if __name__ == "__main__":
    main()
