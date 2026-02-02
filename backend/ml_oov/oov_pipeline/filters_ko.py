from __future__ import annotations

import re

# ✅ 최소 stopwords: 자주 나오는데 OOV 후보로 쓸모 거의 없는 것만
STOPWORDS = {
    "오늘", "어제", "내일", "지금", "방금",
    "정말", "진짜", "그냥", "너무", "조금", "좀", "약간",
    "그리고", "그래서", "하지만", "그런데", "또",
    "아마", "일단", "사실",
}

# ✅ 최소 기능형(조사/어미/접속 등): 후보에 자주 섞이는 2~3글자 위주
FUNCTION_FORMS = {
    "해서", "하고", "인데", "지만", "니까", "네요", "어요", "예요",
    "했다", "했어", "한다", "해요", "돼요", "됐다", "되는", "되어",
    "같아", "같다", "같은",
}

_ONLY_HANGUL = re.compile(r"^[가-힣]+$")


def keep_candidate(surface: str) -> bool:
    """
    True면 후보로 유지, False면 버림.
    - 최소한의 잡음 제거만 수행
    """
    if surface is None:
        return False

    s = surface.strip()
    if not s:
        return False

    # 길이 제한 (너무 짧거나 너무 길면 잡음 가능성 큼)
    if len(s) < 2 or len(s) > 8:
        return False

    # 한글만 (숫자/영문/기호 섞인 건 일단 버림)
    if not _ONLY_HANGUL.match(s):
        return False

    # stopwords / 기능형 제거
    if s in STOPWORDS:
        return False
    if s in FUNCTION_FORMS:
        return False

    # 2~3글자에서 특히 많이 끼는 패턴(최소)
    if len(s) <= 3 and (s.endswith("고") or s.endswith("서") or s.endswith("지") or s.endswith("면")):
        return False

    return True
