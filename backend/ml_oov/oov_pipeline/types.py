from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass(frozen=True)
class CharSpan:
    """원문 텍스트 기준의 문자 단위 span.

    - start: 포함(inclusive)
    - end: 제외(exclusive)
    - text: 반드시 original_text[start:end]와 정확히 일치해야 함
    - label: span의 예측 라벨 (예: NEG)
    - confidence: span 전체의 신뢰도(확률)
    - source: span 생성 출처 (예: 'koelectra', 'lexicon')
    """

    start: int
    end: int
    text: str
    label: str
    confidence: float
    source: str

    def length(self) -> int:
        """span의 문자 길이를 반환한다."""
        return max(0, self.end - self.start)

@dataclass(frozen=True)
class LexiconSpan:
    """사전(lexicon) 매칭으로 얻은 KNOWN span.

    - OOV 판정 시, 이 span과 한 글자라도 겹치면 제거 대상이 됨
    """

    start: int
    end: int
    text: str
    source: str = "lexicon"

@dataclass(frozen=True)
class TokenPred:
    """토큰 단위 예측 결과 (A ↔ B 인터페이스).

    - token: 토큰 문자열
    - label: BIO 형식의 토큰 라벨 (B-NEG, I-NEG, O 등)
    - confidence: 해당 토큰의 예측 확률
    - start/end: 원문 텍스트 기준의 char offset
    """

    token: str
    label: str
    confidence: float
    start: int
    end: int

@dataclass(frozen=True)
class TokenPredictions:
    """토큰 예측 결과 컨테이너.

    - text: 원문 텍스트
    - tokens: TokenPred 리스트
    - meta: 모델/추론 관련 부가 정보(optional)
    """

    text: str
    tokens: List[TokenPred]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """OOV 파이프라인 동작 설정.

    - enabled: 파이프라인 전체 활성화 여부
    - conf_threshold: OOV span 최소 confidence 임계값
    - write_candidates: 후보 JSONL 파일 누적 저장 여부
    - candidate_path: 후보 파일 저장 경로
    - max_examples_per_key: 후보별 최대 예시 문장 수
    """

    enabled: bool = True
    conf_threshold: float = 0.75
    write_candidates: bool = True
    candidate_path: str = "backend/data/oov_candidates.jsonl"
    max_examples_per_key: int = 5

@dataclass
class CandidateRecord:
    """OOV 표현 후보 누적 저장 레코드."""

    key: str
    text: str
    label: str
    count: int
    avg_confidence: float
    examples: List[str]
    first_seen_at: str
    last_seen_at: str

@dataclass
class PipelineResult:
    """OOV 파이프라인 실행 결과.

    - oov_spans: 최종 OOV span 목록
    - model_spans: 모델이 예측한 전체 span(NMS 적용 후)
    - lexicon_spans: 사전 매칭으로 얻은 KNOWN span 목록
    - debug: 중간 결과/통계 등 디버그 정보
    """

    oov_spans: List[CharSpan]
    model_spans: List[CharSpan]
    lexicon_spans: List[LexiconSpan]
    debug: Dict[str, Any] = field(default_factory=dict)

# Dependency types
PredictTokens = Callable[[str], TokenPredictions]
LexiconMatcher = Callable[[str], List[LexiconSpan]]