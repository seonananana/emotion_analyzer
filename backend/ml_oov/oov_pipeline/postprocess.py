from __future__ import annotations

from typing import List
from .types import CharSpan, LexiconSpan


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """두 문자 범위가 한 글자 이상 겹치는지 여부를 반환한다."""
    return not (a_end <= b_start or b_end <= a_start)


def nms_char_spans(spans: List[CharSpan]) -> List[CharSpan]:
    """겹치는 문자(span)에 대해 Greedy 방식의 NMS를 적용한다.

    우선순위 규칙:
    1) confidence가 높은 span 우선
    2) confidence가 같으면 길이가 긴 span 우선

    겹침 기준은 문자(char) 기준으로 한 글자라도 겹치면 overlap으로 판단한다.
    """

    sorted_spans = sorted(
        spans,
        key=lambda s: (s.confidence, s.length()),
        reverse=True,
    )
    kept: List[CharSpan] = []
    for s in sorted_spans:
        if any(_overlaps(s.start, s.end, k.start, k.end) for k in kept):
            continue
        kept.append(s)

    # 가독성을 위해 원문 위치(start, end) 기준으로 다시 정렬
    return sorted(kept, key=lambda s: (s.start, s.end))


def remove_known_overlaps(
    spans: List[CharSpan],
    known: List[LexiconSpan],
) -> List[CharSpan]:
    """사전(lexicon) 매칭 KNOWN span과 겹치는 span을 제거한다.

    - KNOWN span과 문자 기준으로 한 글자라도 겹치면 제거 대상이다.
    - 남은 span만 OOV 후보로 유지된다.
    """

    out: List[CharSpan] = []
    for s in spans:
        if any(_overlaps(s.start, s.end, k.start, k.end) for k in known):
            continue
        out.append(s)
    return out


def apply_conf_threshold(
    spans: List[CharSpan],
    threshold: float,
) -> List[CharSpan]:
    """confidence 임계값(threshold) 미만의 span을 제거한다."""
    return [s for s in spans if s.confidence >= threshold]