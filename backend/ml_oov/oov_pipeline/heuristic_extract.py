# backend/ml_oov/oov_pipeline/heuristic_extract.py
from __future__ import annotations

import re
from typing import List, Iterable, Tuple

from .types import CharSpan, LexiconSpan

# 단순 토큰(한글/자모 혼합 포함) 잡기
_RE_TOKEN = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]{2,20}")

# 너무 흔한 일반어는 후보에서 제외(필요하면 계속 추가)
_STOP = {
    "오늘", "너무", "진짜", "정말", "그냥", "근데", "그래서", "그리고", "하지만",
    "나는", "내가", "우리", "너", "너가", "저", "제가", "그", "이", "저런", "이런",
    "했다", "한다", "됐다", "된다", "왔다", "가다", "하다", "되다", "있다", "없다",
    "같다", "거다", "것", "수", "좀", "막", "완전",
}

# “부정 가능성”을 넓게 잡는 약한 씨앗(리스트 고정이 아니라, 신호 트리거 역할)
# (씨앗은 적게 두고, 변형/붙임/띄어쓰기 깨짐을 포괄하도록 규칙으로 확장)
_NEG_SEEDS = {
    "빡", "열받", "멘붕", "현타", "멘탈", "노답", "답없", "빡침", "빡치",
    "짜증", "불쾌", "우울", "불안", "무섭", "겁", "싫",
    "미치", "죽겠", "최악", "혐오", "극혐",
}

# 자모 욕설/축약(ㅅㅂ, ㅈㄴ, ㅈ같 등) 포괄 신호
_RE_JAMO_SWEAR = re.compile(r"(ㅅㅂ|ㅈㄴ|ㅈ같|ㅂㅅ|ㄱㅅㄲ|ㅅㄲ|ㅈㄹ|ㅁㅊ)")
# 완성형 욕설/강한 비속어 일부(너무 빡빡하게 막지 말고 “신호”로만 사용)
_RE_KOR_SWEAR = re.compile(r"(시발|씨발|좆|존나|개새|병신|미친)")

def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])

def _is_jamo_mixed(token: str) -> bool:
    # ㄱ-ㅎㅏ-ㅣ 포함이면 자모/비표준어 가능성
    return any("ㄱ" <= ch <= "ㅎ" or "ㅏ" <= ch <= "ㅣ" for ch in token)

def _has_repeat(token: str) -> bool:
    # 같은 글자 반복(ㅋㅋㅋ/ㅠㅠ/아아아/빡쳐어)
    return bool(re.search(r"(.)\1\1", token))

def _neg_signal(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    if t in _STOP:
        return False

    # 1) 자모 욕설/축약
    if _RE_JAMO_SWEAR.search(t):
        return True

    # 2) 완성형 욕설 신호
    if _RE_KOR_SWEAR.search(t):
        return True

    # 3) “개 + 형용/동사” 형태 (개빡쳤다/개힘들다 등)
    if t.startswith("개") and len(t) >= 3:
        return True

    # 4) 자모 섞임(비표준/축약) + 길이 조건
    if _is_jamo_mixed(t) and 2 <= len(t) <= 12:
        return True

    # 5) 반복 문자(감정 과장)
    if _has_repeat(t):
        return True

    # 6) 씨앗 어근 포함(변형 폭넓게)
    for s in _NEG_SEEDS:
        if s in t:
            return True

    return False

def extract_heuristic_oov_candidates(
    text: str,
    *,
    lexicon_spans: List[LexiconSpan],
    min_len: int = 2,
    max_len: int = 12,
) -> List[CharSpan]:
    """
    앵커(모델/사전) 없이도 텍스트에서 '부정 가능' 후보를 넓게 추출.
    단, lexicon_spans와 겹치는 건 제외(사전에 있는 부정어는 후보로 안 쌓이게).
    """
    out: List[CharSpan] = []
    lex_ranges = [(int(s.start), int(s.end)) for s in (lexicon_spans or [])]

    for m in _RE_TOKEN.finditer(text):
        s, e = m.start(), m.end()
        tok = m.group().strip()

        if not (min_len <= len(tok) <= max_len):
            continue
        if tok in _STOP:
            continue

        # 사전과 겹치면 제외
        if any(_overlaps((s, e), lr) for lr in lex_ranges):
            continue

        if not _neg_signal(tok):
            continue

        out.append(
            CharSpan(
                start=s,
                end=e,
                text=tok,
                label="CAND_OOV",
                confidence=1.0,
                source="heuristic",
            )
        )

    return out
