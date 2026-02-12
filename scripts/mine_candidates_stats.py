# scripts/mine_candidates_stats.py
from __future__ import annotations

import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import yaml

# ---- 기본 토큰화/정규화 ----
HANGUL_TOKEN = re.compile(r"[가-힣]{2,8}")

JOSA_SUFFIX = (
    "까지","부터","에서","으로","에게","한테","과","와","로","에",
    "을","를","은","는","이","가","도","만",
)

STOP = {
    "오늘","어제","내일","너무","진짜","정말","완전","그냥","약간",
    "했다","한다","하는","있다","없다","됐다","된다","같다","왔다","갔다",
    "그리고","하지만","그래서","또","그런데","때문","정도",
}

def strip_josa(tok: str) -> str:
    tok = tok.strip()
    if len(tok) < 3:
        return tok
    for suf in JOSA_SUFFIX:
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            return tok[:-len(suf)]
    return tok

def tokenize(text: str):
    for m in HANGUL_TOKEN.finditer(text):
        t = strip_josa(m.group())
        if len(t) < 2:
            continue
        if t in STOP:
            continue
        yield t

def load_lexicon_words(lex_yaml: str) -> set[str]:
    p = Path(lex_yaml)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    words = set()

    def walk(x):
        if isinstance(x, dict):
            w = x.get("word")
            if isinstance(w, str) and w.strip():
                words.add(w.strip())
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(data)
    return words

def read_texts_from_excel_like_files(paths: list[str]) -> list[str]:
    """
    여기만 네 프로젝트 엑셀 파서에 맞춰 교체하면 됨.
    일단은 'txt 파일/ yaml / jsonl' 같은 걸 대비해서 최소 안전 구현.
    엑셀은 너가 이미 scripts/extract_oov_from_excel.py 만들었으니 그걸 import해서 써도 됨.
    """
    texts = []
    for fp in paths:
        p = Path(fp)
        if not p.exists():
            continue
        if p.suffix.lower() in [".yaml", ".yml"]:
            obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            # 흔한 케이스: {"items":[{"text":...}, ...]} / ["...", "..."]
            if isinstance(obj, list):
                for it in obj:
                    if isinstance(it, str):
                        texts.append(it)
            elif isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        texts.append(v)
        elif p.suffix.lower() == ".jsonl":
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                # 최소: {"text":"..."}
                if '"text"' in ln:
                    try:
                        import json
                        texts.append(json.loads(ln).get("text",""))
                    except Exception:
                        pass
        else:
            # 엑셀은 여기서 네 기존 스크립트/로더로 교체 권장
            # 지금은 placeholder: 파일을 그냥 읽는 fallback
            try:
                texts.append(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return [t for t in texts if isinstance(t, str) and t.strip()]

def log_odds_ratio(count_pos: int, total_pos: int, count_neg: int, total_neg: int, alpha: float = 0.5) -> float:
    # add-alpha smoothing
    import math
    p = (count_pos + alpha) / (total_pos + alpha * 2.0)
    q = (count_neg + alpha) / (total_neg + alpha * 2.0)
    return math.log(p / q)

def dump_to_oov_candidates_yaml(out_path: str, ranked, examples_map, label="CAND_STATS", topk=300):
    records = []
    now = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    for w, score, cp, cn in ranked[:topk]:
        exs = examples_map.get(w, [])[:3]
        records.append({
            "key": w,
            "text": w,
            "label": label,
            "count": int(cp),
            "avg_confidence": float(score),  # score 저장(의미: 통계 점수)
            "examples": exs,
            "first_seen_at": now,
            "last_seen_at": now,
            "source": "stats",
            "neg_count": int(cp),
            "bg_count": int(cn),
        })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(yaml.safe_dump({"records": records}, allow_unicode=True, sort_keys=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="backend/data/부정_감성_사전_완전판.yaml")
    ap.add_argument("--out", default="backend/data/oov_candidates.yaml")
    ap.add_argument("--topk", type=int, default=300)

    # ✅ 네가 말한 데이터 소스 2개
    ap.add_argument("--neg", nargs="*", default=[
        "backend/data/25-2-전(24).xlsx",
        "backend/data/부정감성 파일럿 응답_학생 글만 2.xlsx",
    ])

    # ✅ 대조군이 없으면 자동 구성(neg 중 lexicon hit 거의 없는 문서로)
    ap.add_argument("--bg", nargs="*", default=[])

    args = ap.parse_args()

    lex_words = load_lexicon_words(args.lexicon)

    neg_texts = read_texts_from_excel_like_files(args.neg)

    # 대조군 처리
    bg_texts = read_texts_from_excel_like_files(args.bg) if args.bg else []
    if not bg_texts:
        # 임시 대조군: neg 중에서 lexicon 단어가 거의 안 나오는 문서
        # (lexicon 기반 "부정성이 약한 문서"를 background로 삼음)
        tmp = []
        for t in neg_texts:
            hits = 0
            for w in tokenize(t):
                if w in lex_words:
                    hits += 1
                if hits >= 2:
                    break
            if hits == 0:
                tmp.append(t)
        # 너무 적으면 그냥 일부 샘플로 채움
        bg_texts = tmp if len(tmp) >= 50 else neg_texts[:200]

    # 카운트
    pos = Counter()
    neg = Counter()
    ex_map = defaultdict(list)

    for t in neg_texts:
        toks = [w for w in tokenize(t) if w not in lex_words]
        for w in toks:
            pos[w] += 1
            if len(ex_map[w]) < 5:
                ex_map[w].append(w)  # 최소 예시(원하면 문맥으로 교체)

    for t in bg_texts:
        toks = [w for w in tokenize(t) if w not in lex_words]
        for w in toks:
            neg[w] += 1

    total_pos = sum(pos.values()) or 1
    total_neg = sum(neg.values()) or 1

    ranked = []
    for w, cp in pos.items():
        # 너무 희귀하면 제외(노이즈)
        if cp < 3:
            continue
        cn = neg.get(w, 0)
        score = log_odds_ratio(cp, total_pos, cn, total_neg, alpha=0.5)
        ranked.append((w, score, cp, cn))

    ranked.sort(key=lambda x: x[1], reverse=True)

    dump_to_oov_candidates_yaml(args.out, ranked, ex_map, topk=args.topk)

    print(f"[OK] wrote {min(args.topk, len(ranked))} records -> {args.out}")

if __name__ == "__main__":
    main()
