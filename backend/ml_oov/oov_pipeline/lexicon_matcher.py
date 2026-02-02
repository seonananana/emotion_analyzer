from __future__ import annotations

from typing import Callable, List, Set, Any
import re
import yaml

from .types import LexiconSpan

# 간단 어미 제거(파생형 확장용)
_SUFFIXES = ("하다", "되다", "시키다", "스럽다", "나다")
_ONLY_HANGUL_OR_SPACE = re.compile(r"^[가-힣 ]+$")


def _collect_all_strings(data: Any) -> List[str]:
    """
    YAML 구조와 무관하게 '모든 문자열'을 수집한다.
    - dict의 key/value
    - list의 item
    """
    out: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
            return

        if isinstance(x, list):
            for v in x:
                walk(v)
            return

        # 그 외 타입은 무시

    walk(data)

    # dedup(순서 유지)
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _normalize_word(w: str) -> str:
    # 공백 여러 개를 하나로, 양끝 공백 제거
    w = re.sub(r"\s+", " ", w.strip())
    return w


def _is_lexicon_word(w: str, *, min_len: int = 2) -> bool:
    """
    lexicon 후보 단어 필터:
    - 최소 2글자 (너 목표상 1글자 '화' 같은 건 제외하는 게 안전)
    - 한글/공백만 허용 (숫자/영문/기호 제거)
    """
    w = _normalize_word(w)
    if len(w) < min_len:
        return False
    if "\n" in w or "\r" in w:
        return False
    if not _ONLY_HANGUL_OR_SPACE.match(w):
        return False
    return True


def _expand_derivations(words: List[str], *, min_stem_len: int = 2) -> List[str]:
    """
    사전에 '불안하다', '짜증나다'만 있어도
    원문에 '불안', '짜증'이 나오면 KNOWN으로 처리하도록 파생형을 확장한다.
    """
    expanded: Set[str] = set()

    for w in words:
        w = _normalize_word(w)
        if not w:
            continue

        expanded.add(w)

        # 공백 포함 표현은 위험하니 그대로만(확장 X)
        if " " in w:
            continue

        # 한글만 파생형 확장
        if not _ONLY_HANGUL_OR_SPACE.match(w):
            continue

        for suf in _SUFFIXES:
            if w.endswith(suf):
                stem = w[: -len(suf)]
                if len(stem) >= min_stem_len:
                    expanded.add(stem)

    # 긴 단어부터 매칭 (겹침 줄이기)
    return sorted(expanded, key=len, reverse=True)


def build_lexicon_matcher(
    lexicon_yaml_path: str,
    *,
    min_len: int = 2,
    min_stem_len: int = 2,
) -> Callable[[str], List[LexiconSpan]]:
    """
    부정 감성 사전 YAML을 로드해서,
    텍스트에서 KNOWN(lexicon) span을 찾아 LexiconSpan 리스트로 반환하는 matcher를 만든다.
    """
    with open(lexicon_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 1) YAML 어디에 있든 문자열 전부 수집
    raw_strings = _collect_all_strings(data)

    # 2) lexicon 단어 후보만 필터
    base_words: List[str] = []
    for s in raw_strings:
        s = _normalize_word(s)
        if _is_lexicon_word(s, min_len=min_len):
            base_words.append(s)

    # dedup 유지
    seen = set()
    base_words_uniq: List[str] = []
    for w in base_words:
        if w not in seen:
            seen.add(w)
            base_words_uniq.append(w)

    # 3) 파생형 확장(하다/나다 등 제거)
    words = _expand_derivations(base_words_uniq, min_stem_len=min_stem_len)

    def matcher(text: str) -> List[LexiconSpan]:
        spans: List[LexiconSpan] = []
        for w in words:
            start = 0
            while True:
                i = text.find(w, start)
                if i == -1:
                    break
                j = i + len(w)
                spans.append(LexiconSpan(start=i, end=j, text=text[i:j], source="lexicon"))
                start = i + 1
        spans.sort(key=lambda s: (s.start, s.end))
        return spans

    return matcher


# ✅ 모듈 레벨에서 matcher 만들어 쓰고 싶으면 이렇게만 두면 됨
LEXICON_YAML = "backend/data/부정_감성_사전_완전판.yaml"
lexicon_matcher = build_lexicon_matcher(LEXICON_YAML, min_len=2, min_stem_len=2)
