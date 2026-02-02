from __future__ import annotations

from typing import List, Optional

from .types import CharSpan, TokenPredictions


def _is_b(label: str) -> bool:
    """BIO 라벨이 B-로 시작하는지 여부를 반환한다."""
    return label.startswith("B-")


def _is_i(label: str) -> bool:
    """BIO 라벨이 I-로 시작하는지 여부를 반환한다."""
    return label.startswith("I-")


def _base(label: str) -> str:
    """BIO 라벨에서 base 라벨을 추출한다.

    예:
    - B-NEG -> NEG
    - I-NEG -> NEG
    - O     -> O
    """
    if "-" not in label:
        return label
    return label.split("-", 1)[1]


def recover_spans(
    preds: TokenPredictions,
    *,
    target_prefixes: Optional[List[str]] = None,
    source: str = "koelectra",
) -> List[CharSpan]:
    """토큰 단위 BIO 예측 결과를 문자(char) 단위 span으로 복원한다.

    Parameters
    ----------
    preds:
        원문 텍스트와 토큰 단위 예측 결과(TokenPredictions).
        각 토큰은 원문 기준 char offset(start/end)을 포함해야 한다.
    target_prefixes:
        지정 시, base 라벨(예: B-NEG/I-NEG의 NEG)이 이 목록에 포함된 경우만 span으로 복원한다.
        None이면 O를 제외한 모든 라벨을 대상으로 한다.
    source:
        생성된 CharSpan에 기록할 source 값 (예: 'koelectra').

    규칙 (스펙 준수)
    ----------------
    - B-X는 새로운 span의 시작이다.
    - I-X는 기존 span을 이어간다.
    - 선행 B-X 없이 I-X가 등장하면, 새로운 B-X로 간주하여 span을 시작한다.
    - span의 confidence는 포함된 토큰 confidence의 평균값이다.
    - span의 text는 반드시 원문 substring `text[start:end]`와 일치해야 한다.
    """

    text = preds.text
    tokens = preds.tokens

    spans: List[CharSpan] = []

    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_label: Optional[str] = None  # base 라벨(X)
    cur_confs: List[float] = []

    def flush() -> None:
        """현재 누적 중인 span을 종료하고 결과 리스트에 추가한다."""
        nonlocal cur_start, cur_end, cur_label, cur_confs
        if cur_start is None or cur_end is None or cur_label is None:
            cur_start = cur_end = cur_label = None
            cur_confs = []
            return
        if cur_end <= cur_start:
            cur_start = cur_end = cur_label = None
            cur_confs = []
            return
        span_text = text[cur_start:cur_end]
        if span_text == "":
            cur_start = cur_end = cur_label = None
            cur_confs = []
            return
        conf = sum(cur_confs) / max(1, len(cur_confs))
        spans.append(
            CharSpan(
                start=cur_start,
                end=cur_end,
                text=span_text,
                label=cur_label,
                confidence=float(conf),
                source=source,
            )
        )
        cur_start = cur_end = cur_label = None
        cur_confs = []

    for tp in tokens:
        lbl = tp.label

        # O 라벨 또는 빈 라벨은 span 경계로 처리
        if lbl == "O" or lbl == "":
            flush()
            continue

        base = _base(lbl)

        # 관심 대상 라벨이 아니면 span 경계로 처리
        if target_prefixes is not None and base not in target_prefixes:
            flush()
            continue

        if _is_b(lbl):
            # 새로운 span 시작
            flush()
            cur_start = tp.start
            cur_end = tp.end
            cur_label = base
            cur_confs = [tp.confidence]
            continue

        if _is_i(lbl):
            # 활성 span이 없으면 B로 간주 (스펙 규칙)
            if cur_start is None or cur_label is None:
                flush()
                cur_start = tp.start
                cur_end = tp.end
                cur_label = base
                cur_confs = [tp.confidence]
                continue

            # I-* 도중 라벨이 바뀌면 기존 span 종료 후 새 span 시작
            if cur_label != base:
                flush()
                cur_start = tp.start
                cur_end = tp.end
                cur_label = base
                cur_confs = [tp.confidence]
                continue

            # 정상적인 I-* 연속
            cur_end = tp.end
            cur_confs.append(tp.confidence)
            continue

        # 알 수 없는 라벨 형식은 경계로 처리
        flush()

    # 마지막 span 처리
    flush()

    # 최종 안전 검사: span text가 원문과 정확히 일치하는 경우만 유지
    cleaned: List[CharSpan] = []
    for s in spans:
        if 0 <= s.start < s.end <= len(text) and text[s.start:s.end] == s.text:
            cleaned.append(s)

    return cleaned