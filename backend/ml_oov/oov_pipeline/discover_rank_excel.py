# backend/ml_oov/oov_pipeline/discover_rank_excel.py
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, DefaultDict
from collections import Counter, defaultdict

import pandas as pd
import yaml

from backend.ml_oov.oov_pipeline.config import load_config
from backend.ml_oov.oov_pipeline.lexicon_adapter import analyzer_lexicon_matcher
from backend.ml_oov.oov_pipeline.filters_ko import keep_candidate
from backend.ml_oov.oov_pipeline.yaml_candidates_store import YAMLCandidateStore
from backend.ml_oov.oov_pipeline.types import CandidateRecord


# ============================================================
# 강한 후보 필터 + ngram 유틸
# ============================================================

_ONLY_HANGUL = re.compile(r"^[가-힣]+$")
_PARTICLE_LIKE = re.compile(
    r"^(은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|까지|부터|에게|한테|께|의|랑|이나|나|며|고|서)$"
)

STOPWORDS = {
    "나", "너", "우리", "저", "그", "이", "것", "거", "수", "때", "당시",
    "그리고", "하지만", "그래서", "또", "또는",
    "정말", "그냥", "약간", "조금", "괜히", "오히려",
    "마음", "느낌", "모습",

    # 파일럿 데이터 잡음(원하면 제거 가능)
    "수학", "양양이",
    "친척", "언니", "미래", "이별", "성적", "잠시", "무엇", "머리",
}

# --- 활용형/표제어 약한 접기(유니그램용) ---
_CANON_MAP_PREFIX = (
    ("못했", "못하"),
    ("못한", "못하"),
    ("못할", "못하"),
    ("못해", "못하"),
    ("못하", "못하"),
    ("아픈", "아프"),
    ("아팠", "아프"),
    ("아프", "아프"),
    ("어려운", "어렵"),
    ("어려웠", "어렵"),
    ("어렵", "어렵"),
    ("사소한", "사소"),
    ("사소했", "사소"),
    ("사소", "사소"),
)

_CANON_SUFFIXES = (
    "습니다", "했습니다", "합니다",
    "했어요", "했어", "했다", "했", "해",
    "었던", "았던", "었다", "았다", "었", "았",
    "하는", "한", "할", "됨", "된다", "되",
    "게", "고", "지", "서", "요",
    "은", "는", "을",
)

def canonicalize(tok: str) -> str:
    t = (tok or "").strip()
    if not t:
        return ""
    for pre, base in _CANON_MAP_PREFIX:
        if t.startswith(pre):
            return base
    for suf in _CANON_SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf) + 1:
            return t[:-len(suf)]
    return t


def is_bad_candidate(tok: str) -> bool:
    if not tok:
        return True
    t = tok.strip()
    if len(t) <= 1:
        return True
    if t in STOPWORDS:
        return True
    if _PARTICLE_LIKE.match(t):
        return True
    if not _ONLY_HANGUL.match(t):
        return True
    return False


def make_ngrams(tokens: List[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


# ============================================================
# sentence split (weak)
# ============================================================

_SENT_SPLIT = re.compile(r"[\n\r]+|(?<=[\.\!\?…])\s+|(?<=다)\.\s*|(?<=요)\.\s*")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= 2]


# ============================================================
# tokenization + normalize
# ============================================================

_EDGE_NOISE = re.compile(r"^[^\w가-힣]+|[^\w가-힣]+$", flags=re.UNICODE)
_WS_TOKEN = re.compile(r"[가-힣]+|[^\s가-힣]+")  # 한글 덩어리 + 기타 덩어리

_SAVE_JOSA_MIN = (
    "까지", "부터", "에서", "으로", "에게", "한테",
    "과", "와", "로", "에",
    "을", "를", "은", "는", "이", "가", "도", "만",
    "요",
)

def normalize_candidate_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _EDGE_NOISE.sub("", s)
    s = re.sub(r"\s+", "", s)

    # 아주 얕은 조사 제거(1회)
    for suf in _SAVE_JOSA_MIN:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
            break
    return s


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ============================================================
# lexicon words load (robust) + variations
# ============================================================

def _load_words_anywhere_from_yaml(path: str) -> Dict[str, str]:
    """
    YAML 구조와 무관하게 'word' 키의 문자열을 전부 수집.
    return: {word: word}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    words: Dict[str, str] = {}

    def walk(x):
        if isinstance(x, dict):
            w = x.get("word")
            if isinstance(w, str) and w.strip():
                ww = w.strip()
                words[ww] = ww
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(data)
    return words


def create_word_variations(base_words: List[str]) -> Dict[str, str]:
    """
    base_word -> variations 생성(간단 버전)
    return: variation_form -> base_word
    """
    variations: Dict[str, str] = {}

    for word in base_words:
        if not word:
            continue
        word = word.strip()
        if not word:
            continue

        variations[word] = word

        # 동사/형용사 기본 활용 (…다)
        if word.endswith("다") and len(word) >= 2:
            stem = word[:-1]
            forms = [
                stem + "어", stem + "아",
                stem + "었", stem + "았",
                stem + "고", stem + "지",
                stem + "는", stem + "은", stem + "을",
                stem + "게", stem + "면",
                stem + "어서", stem + "아서",
                stem + "었어", stem + "았어",
                stem + "어요", stem + "아요",
                stem + "었습니다", stem + "았습니다",
                stem + "다",
            ]

            # 하다 계열
            if stem.endswith("하") and len(stem) >= 1:
                ha = stem[:-1]
                forms += [
                    ha + "해", ha + "해서",
                    ha + "했", ha + "했다",
                    ha + "했어", ha + "했어요",
                    ha + "합니다", ha + "했습니다",
                    ha + "한", ha + "할",
                ]

            for f in forms:
                if f and f not in variations:
                    variations[f] = word

    return variations


def build_variations_from_lexicon(lexicon_path: str) -> Dict[str, str]:
    base_map = _load_words_anywhere_from_yaml(lexicon_path)
    base_words = list(base_map.keys())
    return create_word_variations(base_words)


# ============================================================
# token extraction for ranking (unigram 후보)
# ============================================================

def extract_oov_tokens(text: str, *, variations: Dict[str, str]) -> List[str]:
    """
    unigram 후보(저장용):
    - 사전/variations 제외
    - keep_candidate + 강한 필터
    - 활용형 접기 + 흔한 베이스 컷
    """
    raw = _WS_TOKEN.findall(text)
    out: List[str] = []

    for t in raw:
        t = (t or "").strip()
        if not t:
            continue

        # 1글자 부정어는 unigram 후보로는 쓰지 않음(ngram에서만 활용)
        if t in ("안", "못"):
            continue

        norm = normalize_candidate_text(t)
        if not norm:
            continue

        if not keep_candidate(norm):
            continue
        if norm in variations:
            continue

        base = canonicalize(norm)
        if not base:
            continue

        # 너무 흔한 베이스 컷(원하면 조절)
        if base.startswith(("못", "힘들", "어렵", "아프", "사소")):
            continue

        if not keep_candidate(base):
            continue
        if is_bad_candidate(base):
            continue
        if base in variations:
            continue

        out.append(base)

    return out


# ============================================================
# token extraction for ngram phrases (keep 안/못)
# ============================================================

NGRAM_STOP = {
    "마음", "생각", "느낌", "사람", "서로", "정도", "이상", "가끔", "갑자기",
    "그냥", "정말", "조금", "약간", "여전히", "쉽게", "수도",
    "나는", "내가", "너무", "모든", "것", "것을", "사람들", "다른", "무엇", "말", "듣고", "잠시",
    "미래", "성적", "이별",
    "친척", "언니", "머리",
}

# ✅ '못/안' 뒤에 올 때 "용언 느낌"일 때만 분해 허용
_NEG_SPLIT_ALLOW = re.compile(
    r"^(하|해|했|할|하고|해서|한다|하면|했었|되|돼|됐|될|된다|되면|됨|되는데|되지만|되었|되어)"
)

def extract_phrase_tokens(text: str) -> List[str]:
    """
    ngram(표현)용 토큰:
    - '안','못' 1글자 부정어 살림
    - 붙어쓴 부정어 분해: 안돼/못해 -> 안 + 돼, 못 + 해 (단, rest가 용언일 때만)
    - 그 외는 normalize + 필터
    - variations 제외/표제어 접기 X (표현 보존)
    """
    raw = _WS_TOKEN.findall(text)
    toks: List[str] = []

    for t in raw:
        t = (t or "").strip()
        if not t:
            continue

        if t in ("안", "못"):
            toks.append(t)
            continue

        norm = normalize_candidate_text(t)
        if not norm:
            continue

        # ✅ 붙어쓴 부정어 분해(용언일 때만)
        if norm.startswith("안") and len(norm) >= 2:
            rest = norm[1:]
            if rest and _ONLY_HANGUL.match(rest) and _NEG_SPLIT_ALLOW.match(rest):
                toks.append("안")
                toks.append(rest)
                continue

        if norm.startswith("못") and len(norm) >= 2:
            rest = norm[1:]
            if rest and _ONLY_HANGUL.match(rest) and _NEG_SPLIT_ALLOW.match(rest):
                toks.append("못")
                toks.append(rest)
                continue

        if norm in NGRAM_STOP:
            continue

        if is_bad_candidate(norm):
            continue

        toks.append(norm)

    return toks


# ============================================================
# A2: anchor-window slicing
# ============================================================

def slice_by_anchor_window(sent: str, spans: list, window: int) -> str:
    if window <= 0 or not spans:
        return sent

    ranges: List[Tuple[int, int]] = []
    L = len(sent)

    for sp in spans:
        s = getattr(sp, "start", None)
        e = getattr(sp, "end", None)
        if not isinstance(s, int) or not isinstance(e, int) or s >= e:
            continue
        left = max(0, s - window)
        right = min(L, e + window)
        ranges.append((left, right))

    if not ranges:
        return sent

    ranges.sort()
    merged: List[Tuple[int, int]] = []
    cs, ce = ranges[0]
    for s, e in ranges[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    return " ".join(sent[s:e] for s, e in merged if s < e)


# ============================================================
# negation ngram canonicalization (안되다/못하다)
# ============================================================

def _canon_neg_token(x: str) -> str:
    if x.startswith(("됐", "돼", "되")):
        return "되"
    if x.startswith(("했", "해", "하", "하고", "해서")):
        return "하"
    return x

def canonicalize_negation_ngram(ng: str) -> str:
    """
    부정어 표현을 하나의 대표키로 강하게 접는다.
    - "못 (하 계열)" 포함 → "못하다"
    - "안 (되 계열)" 포함 → "안되다"
    """
    parts = ng.split()
    if not parts:
        return ng

    if "못" in parts:
        i = parts.index("못")
        if i + 1 < len(parts):
            v = _canon_neg_token(parts[i + 1])
            if v == "하":
                return "못하다"

    if "안" in parts:
        i = parts.index("안")
        if i + 1 < len(parts):
            v = _canon_neg_token(parts[i + 1])
            if v == "되":
                return "안되다"

    return ng


# ============================================================
# YAML 저장 (negation 후보 누적)
# ============================================================

def _merge_examples(old: List[str], new: List[str], max_n: int) -> List[str]:
    out = list(old or [])
    for ex in new:
        if ex and ex not in out:
            out.append(ex)
        if len(out) >= max_n:
            break
    return out[:max_n]


def save_negation_to_yaml(
    *,
    out_path: str,
    label: str,
    neg_ranked: List[Tuple[str, int, int, float]],
    examples_map: Dict[str, List[str]],
    max_examples: int,
) -> None:
    store = YAMLCandidateStore(out_path, max_examples=max_examples)
    recs: Dict[str, CandidateRecord] = store._load()  # type: ignore[attr-defined]
    now = _now()

    for key, neg_f, _, score in neg_ranked:
        exs = (examples_map.get(key) or [])[:max_examples]

        if key not in recs:
            recs[key] = CandidateRecord(
                key=key,
                text=key,
                label=label,
                count=int(neg_f),
                avg_confidence=float(score),
                examples=exs,
                first_seen_at=now,
                last_seen_at=now,
            )
            continue

        r = recs[key]
        new_count = int(r.count) + int(neg_f)
        new_avg = (float(r.avg_confidence) * float(r.count) + float(score) * float(neg_f)) / float(new_count)

        recs[key] = CandidateRecord(
            key=r.key,
            text=r.text,
            label=r.label or label,
            count=new_count,
            avg_confidence=float(new_avg),
            examples=_merge_examples(r.examples, exs, max_examples),
            first_seen_at=r.first_seen_at,
            last_seen_at=now,
        )

    store._dump(recs)  # type: ignore[attr-defined]


# ============================================================
# main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--min_freq", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=2.0)  # smoothing
    ap.add_argument("--anchor_window", type=int, default=15)
    ap.add_argument("--label", default="STAT_RANK_A2")
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--out", default=None)  # ✅ YAML 저장 경로
    args = ap.parse_args()

    cfg = load_config()

    lexicon_path = (
        args.lexicon
        or getattr(cfg, "lexicon_path", None)
        or "backend/data/부정_감성_사전_완전판.yaml"
    )
    out_path = args.out or getattr(cfg, "candidate_path", "backend/data/oov_candidates.yaml")

    variations = build_variations_from_lexicon(lexicon_path)
    if not variations:
        raise RuntimeError(f"variations empty: {lexicon_path}")

    # load excel
    if args.sheet:
        df = pd.read_excel(args.excel, sheet_name=args.sheet)
        sheet_used = args.sheet
    else:
        df = pd.read_excel(args.excel)
        sheet_used = "(first)"

    if args.text_col not in df.columns:
        raise ValueError(f"no text column: {args.text_col}\ncols={list(df.columns)}")

    texts = df[args.text_col].dropna().astype(str).tolist()

    neg_counter = Counter()
    ctrl_counter = Counter()

    # ✅ 안/못 포함 ngram 전용 카운터(NEG-like에서만 수집)
    neg_ng_counter = Counter()
    examples_map: DefaultDict[str, List[str]] = defaultdict(list)

    neg_sent_n = 0
    ctrl_sent_n = 0

    for doc in texts:
        sents = split_sentences(doc)

        for sent in sents:
            spans = analyzer_lexicon_matcher(sent) or []
            is_neg_like = len(spans) > 0

            if is_neg_like:
                neg_sent_n += 1
                view = slice_by_anchor_window(sent, spans, args.anchor_window)
                phrase_view = slice_by_anchor_window(sent, spans, max(int(args.anchor_window), 60))
            else:
                ctrl_sent_n += 1
                view = sent
                phrase_view = sent

            # unigram 후보 (NEG vs CTRL)
            toks = extract_oov_tokens(view, variations=variations)
            if toks:
                if is_neg_like:
                    neg_counter.update(toks)
                else:
                    ctrl_counter.update(toks)

            # ngram 후보(안/못 살리기)
            phrase_toks = extract_phrase_tokens(phrase_view)
            phrase_toks = [t for t in phrase_toks if (t in ("안", "못") or len(t) >= 2)]

            for n in (2, 3):
                for ng in make_ngrams(phrase_toks, n):
                    parts = ng.split()
                    if any(is_bad_candidate(p) for p in parts if p not in ("안", "못")):
                        continue

                    # 전체 ranked에 포함(원래 로직)
                    if is_neg_like:
                        neg_counter[ng] += 1
                    else:
                        ctrl_counter[ng] += 1
                    if "잘 못" in ng or "너무 못" in ng or "매우 못" in ng or "정말 못" in ng:
                         continue
                    # ✅ negation-ngrams는 NEG-like에서만 모은다 (CTRL 비교 X)
                    if is_neg_like and ("안 " in ng or "못 " in ng):
                        key = canonicalize_negation_ngram(ng)
                        neg_ng_counter[key] += 1
                        max_ex = getattr(cfg, "max_examples_per_key", 5)
                        if sent not in examples_map[key]:
                            examples_map[key].append(sent)
                            if len(examples_map[key]) > max_ex:
                                examples_map[key] = examples_map[key][:max_ex]

    ranked: List[Tuple[str, int, int, float]] = []
    alpha = float(args.alpha)

    for w, nf in neg_counter.items():
        if nf < int(args.min_freq):
            continue
        cf = int(ctrl_counter.get(w, 0))

        # ctrl=0 과대점수 방지
        if cf == 0 and nf < 8:
            continue

        score = math.log((nf + alpha) / (cf + alpha))
        ranked.append((w, int(nf), int(cf), float(score)))

    ranked.sort(key=lambda x: (x[3], x[1]), reverse=True)
    top = ranked[: int(args.topk)]

    print(f"[OK] excel={args.excel} (sheet={sheet_used})")
    print(f"[OK] text_col={args.text_col}")
    print(f"[OK] lexicon={lexicon_path} variations={len(variations)}")
    print(f"[OK] neg_like_sentences={neg_sent_n}, ctrl_like_sentences={ctrl_sent_n}")
    print(f"[OK] anchor_window={int(args.anchor_window)}")
    print(f"[OK] candidates_considered={len(ranked)}, saved_topk={len(top)}")
    print("[TOP 20 preview]")
    for row in top[:20]:
        print(row)

    # ✅ 안/못 ngram 전용 랭킹: NEG-like에서 얼마나 자주 나오나
    neg_ranked: List[Tuple[str, int, int, float]] = []
    for w, nf in neg_ng_counter.items():
        if nf < 2:
            continue
        score = math.log((nf + alpha) / (0 + alpha))
        neg_ranked.append((w, int(nf), 0, float(score)))

    neg_ranked.sort(key=lambda x: (x[3], x[1]), reverse=True)

    print("[TOP 30 negation-ngrams] word, neg_freq, ctrl_freq, score")
    if neg_ranked:
        for row in neg_ranked[:30]:
            print(row)
    else:
        print("(none)")

    # ✅ YAML 저장(negation 후보 누적)
    save_negation_to_yaml(
        out_path=out_path,
        label=str(args.label) + "_NEGATION",
        neg_ranked=neg_ranked,
        examples_map=examples_map,
        max_examples=getattr(cfg, "max_examples_per_key", 5),
    )
    print(f"[OK] saved negation candidates -> {out_path}")


if __name__ == "__main__":
    main()
