# Emotion Analyzer  
학생 일기 부정 감정 분석 웹 서비스 (최종 프로젝트)

---

## 1. 프로젝트 개요

이 프로젝트는 **부정 감성 사전 기반 규칙 분석**과 **LLM 기반 요약**을 결합하여  
학생의 일기(부정 경험 글)를 정량·정성적으로 분석하는 웹 서비스이다.

- **규칙 기반 분석 (Lexicon)**
  - 부정_감성_사전_완전판.yaml을 이용해 부정 감정 단어를 탐지
  - 감정 카테고리별 빈도·비율, 부정 비율, 상위 부정 키워드, 감정 점수 계산
- **LLM 기반 요약**
  - Together API를 통해 LLM을 호출
  - 규칙 기반 결과 + 원문 텍스트를 입력으로 받아 학생 정서 상태를 서술형으로 요약
- **웹 UI**
  - FastAPI + Jinja2 기반 웹 페이지에서 학생 정보와 일기를 입력하고
  - 분석 결과(지표, 그래프, 요약문)를 확인
- **결과 저장**
  - output/ 폴더에 학생별 YAML 결과 저장


---

## 2. 폴더 구조

프로젝트 루트(emotion_analyzer/) 기준 구조는 다음과 같다.

```bash
emotion_analyzer/
├─ .env                 # 실제 실행 환경 설정 (사용자가 직접 작성)
├─ .env.example         # .env 샘플
├─ requirements.txt     # 파이썬 의존성
├─ backend/
│  ├─ app/              # FastAPI 엔트리포인트 및 라우터
│  │  ├─ main.py        # uvicorn 실행 시 사용하는 app 객체
│  │  ├─ routes_analysis.py  # /analyze API
│  │  ├─ routes_health.py    # /health API
│  │  └─ routes_pages.py     # 웹 페이지 라우트
│  ├─ conf/             # LLM 프롬프트, 키 파일
│  │  └─ instruct/      # NA-system / NA-user 프롬프트 텍스트
│  ├─ core/
│  │  └─ config.py      # 기본 경로/환경 설정
│  ├─ data/
│  │  ├─ 부정_감성_사전_완전판.yaml     # 부정 감정 사전
│  │  └─ 학생_부정경험_글모음.yaml       # 학생 일기 원본 데이터(배치용)
│  ├─ domain/
│  │  ├─ analyzer.py    # NegativeEmotionAnalyzer (규칙 기반 핵심 로직)
│  │  └─ aggregation.py # 부정 비율, 감정 점수, 키워드 등 2차 지표 계산
│  ├─ infra/
│  │  ├─ paths.py       # data/conf/output 경로 관리
│  │  ├─ yaml_io.py     # YAML 로드/저장
│  │  ├─ lexicon_repo.py # 부정 사전 로딩/캐시
│  │  ├─ student_repo.py # 학생 원본 데이터 로딩
│  │  ├─ output_repo.py # 분석 결과 저장
│  │  ├─ llm_client.py  # Together API 호출 클라이언트
│  │  └─ prompts.py     # 프롬프트 파일 로딩
│  ├─ services/
│  │  ├─ analysis_service.py  # 웹/배치 공통 분석 서비스
│  │  └─ legacy_bridge.py     # 기존 스크립트 ↔ 새 구조 브릿지
│  ├─ usecases/
│  │  ├─ analyze_single_diary.py # 단일 일기 분석 유즈케이스
│  └─ exceptions.py     # ConfigError, AnalysisError, LLMError 등 예외 정의
├─ frontend/
│  ├─ templates/
│  │  ├─ base.html      # 공통 레이아웃
│  │  ├─ input.html     # 입력 페이지
│  │  └─ result.html    # 결과 페이지
│  └─ static/
│     ├─ style.css      # 기본 스타일
│     └─ main.js        # 폼 전송, 결과 렌더링 스크립트
└─ output/              # 분석 결과 YAML 파일 저장 디렉터리
```

---

## 3. 사전 준비

### 3.1 Python 및 가상환경

- Python **3.10 이상** 권장
- 가상환경 사용 권장

```bash
# (선택) 가상환경 생성
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3.2 라이브러리 설치

루트 디렉터리에서 다음 명령 실행:

```bash
pip install -r requirements.txt

# 필요 시 웹 서버 관련 패키지 추가 설치
pip install "fastapi[standard]" "uvicorn[standard]" python-dotenv jinja2 pyyaml together
```

### 3.3 환경 변수 설정 (`.env`)

.env.example을 복사해 .env 생성 후 값 채우기:

```bash
cp .env.example .env
```

예시:

```env
APP_ENV=development
LOG_LEVEL=INFO

TOGETHER_API_KEY=여기에_실제_Together_API_키_입력
TOGETHER_CHAT_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
TOGETHER_TEMPERATURE=0.2
TOGETHER_MAX_TOKENS=800
```

---

## 4. 실행 방법

### 4.1 개발 서버 실행

루트 디렉터리(emotion_analyzer/)에서 FastAPI 앱 실행:

```bash
uvicorn backend.app.main:app --reload --port 8000(or8001)
```

### 4.2 웹 사용 방법

1. 브라우저에서 http://localhost:8000/ 접속  
2. 학생 기본 정보와 일기 텍스트를 입력  
3. **분석 실행** 버튼 클릭  
4. 화면에서 다음 내용 확인
   - 부정 비율, 텍스트 길이, 상위 부정 키워드
   - 감정 카테고리별 점수(그래프)
   - LLM 요약문

---

## 5. 향후 확장 방향

### 5.1 학생별 위험도 점수 (Top N + 머신러닝)

- 규칙 기반 분석으로 특징 추출  
  - 부정 단어 비율, 감정 카테고리 점수, 고위험 키워드 수 등
- 이 특징으로 **임시 위험 점수(raw score)** 를 만들고  
  점수가 높은 순으로 정렬해 **상위 N명**을 고위험(1), 나머지를 일반(0)으로 라벨링
- 해당 데이터를 이용해  
  - **로지스틱 회귀(Logistic Regression)**  
  - **랜덤포레스트(Random Forest)**  
  모델을 학습
- 새 일기에 대해  
  - 고위험 확률(0~1) → 0~100 점수 및 low / medium / high 등급으로 변환

예시 응답:

```json
"risk": {
  "probability": 0.82,
  "score": 82,
  "level": "high"
}
```

### 5.2 LLM–Lexicon 일관성 체크 (일치도 점수)

- 사전 기반 분석과 LLM 분석에서 각각 감정 분포를 추출  
  - 예: 슬픔/불안/분노 비율
- 두 분포를 벡터로 만들어 코사인 유사도 등으로 0~1 유사도 계산
- 유사도 × 100 → **0~100점 “일치도 점수”** 로 표현

예시:

```json
"scores": {
  "sadness": 0.52,
  "anxiety": 0.31,
  "anger": 0.17,
  "lexicon_llm_alignment": 87.0
}
```

- 웹 화면에는  
  > LLM–Lexicon 일치도: **87%**  
  처럼 간단히 표시
- 점수가 낮을 경우  
  - 사전에 없는 표현이 많은지  
  - LLM 프롬프트/감정 축 정의를 조정해야 하는지  
  를 검토하는 기준으로 활용

---

---

## 6. OOV 태깅 모델 모듈 (ml_oov)

# ml_oov 디렉토리 트리 + 파일 역할(복붙용)

> 목적: A(모델 담당)가 `ml_oov`만 완성하면 B(파이프라인 담당)가 `predict_tokens()`를 바로 가져다 붙일 수 있게 하는 최소 구조.

```text
emotion_analyzer/
├─ backend/
│  ├─ ml_oov/                              # A(모델) 전용: OOV 토큰 태깅/평가/추론 모듈
│  │  ├─ __init__.py                       # 패키지 엔트리. (선택) predictor 등 핵심 export
│  │  ├─ labels.py                         # 라벨 단일 소스: LABELS/LABEL2ID/ID2LABEL/IGNORE_INDEX
│  │  ├─ sent_split.py                     # 문장 분할 + 원문 char offset 유지 유틸(. ! ? \n, strip 금지)
│  │  ├─ predictor.py                      # ✅ B 접점: predict_tokens(text) 제공(토큰 label+conf+start/end)
│  │  ├─ dataset_builder.py                # gold span(train/dev/test) → token BIO 라벨 데이터 생성(오프셋 정합)
│  │  ├─ train_tokenclf.py                 # KoELECTRA token classification 학습(CPU 가능) + 체크포인트 저장
│  │  ├─ eval_oov.py                       # 평가: OOV Precision/Recall/Char-F1(+선택 Exact span-F1) 산출
│  │  └─ schemas.py                        # (선택) pydantic 등 출력/입력 스키마 고정(TokenPrediction 등)
│  │
│  └─ data/
│     └─ oov_gold/                         # Gold 정답 데이터(원문+OOV span만 저장)
│        ├─ train.jsonl                    # 학습용 gold: {id,text,oov_spans[{start,end,label}]}
│        ├─ dev.jsonl                      # 검증용 gold
│        ├─ test.jsonl                     # 테스트/평가용 gold
│        └─ README.md                      # gold 규격 문서(start/end end-exclusive, offset 주의사항, 예시)
│
└─ models/
   └─ koelectra_oov/                       # A가 저장하는 로컬 체크포인트(모델/토크나이저/설정/메트릭)
      ├─ (model files...)                  # pytorch_model.bin or *.safetensors, config.json 등
      ├─ (tokenizer files...)              # tokenizer.json, vocab, merges 등
      ├─ training_config.json              # (권장) 학습 재현용 설정(lr/epoch/seed/max_len 등)
      └─ metrics.json                      # (권장) eval_oov.py 산출 지표(P/R/Char-F1 등)
```

---

## 파일별 역할(구체)

### `backend/ml_oov/__init__.py`
- `ml_oov` 패키지 선언.
- (선택) 외부에서 편하게 쓰도록 핵심 심볼을 re-export.
  - 예: `from .predictor import predict_tokens, OOVTokenPredictor`

### `backend/ml_oov/labels.py`
- 라벨 체계의 단일 소스(하드코딩 분산 금지).
- 포함:
  - `LABELS = ["O", "B-NEG_OOV", "I-NEG_OOV"]`
  - `LABEL2ID`, `ID2LABEL`
  - `IGNORE_INDEX = -100` (pad/special 토큰 학습 제외용)

### `backend/ml_oov/sent_split.py`
- 문장 분할 유틸: `. ! ? \n` 기준으로 분할하되 **원문 char offset 유지**.
- 출력(권장): `List[Tuple[sent_text, sent_start, sent_end]]`
- 주의: `strip()`/trim 금지(오프셋 깨짐).

### `backend/ml_oov/predictor.py`  ✅ B가 가져다 쓰는 접점
- A가 제공하는 최종 “추론 인터페이스”.
- 책임:
  1) 체크포인트 로드(모델/토크나이저)
  2) 문장 분할(필요시) + `offset_mapping`으로 원문 기준 (start,end) 복원
  3) 토큰별 `label` + `confidence(softmax prob)` 생성
- 반환(최소):
  - `[{ "token": str, "start": int, "end": int, "label": str, "confidence": float }, ...]`
- 하지 않는 것(=B 책임):
  - span 복원, NMS, known 제외, threshold 적용, 파일 누적 저장, API 통합

### `backend/ml_oov/dataset_builder.py`
- gold jsonl의 OOV span(start/end)을 토큰 BIO 라벨로 변환해서 학습용 데이터로 만드는 생성기.
- 책임:
  - tokenizer `offset_mapping`과 gold span 정합
  - BIO 라벨링(`B/I/O`) + `IGNORE_INDEX` 처리(pad/special)
  - (권장) 정합 검증 로그:
    - `text[start:end]` 불일치 gold는 폐기/기록

### `backend/ml_oov/train_tokenclf.py`
- KoELECTRA token classification 학습 스크립트(CPU 가능).
- 책임:
  - 데이터 로드(`dataset_builder.py`)
  - 학습 실행/저장: `models/koelectra_oov/`
  - 재현성 설정 저장: `training_config.json` (권장)

### `backend/ml_oov/eval_oov.py`
- 평가 스크립트(모델 품질 확인).
- 책임:
  - OOV Precision/Recall, Char-F1(문자 단위 overlap) 계산
  - (선택) Exact span-F1
  - 결과 저장: `metrics.json` (권장) + 예측 dump(선택)

### `backend/ml_oov/schemas.py` (선택)
- pydantic/dataclass로 출력 스키마 고정.
- 목적: A/B 인터페이스 흔들림 방지 + API 통합시 검증.

---

## Gold 데이터 규격(요약)

`backend/data/oov_gold/*.jsonl` 각 라인 예시:

```json
{"id":"doc_0001","text":"원문...","oov_spans":[{"start":18,"end":26,"label":"NEG_OOV"}]}
```

- `start/end`: 원문 char index, `end` exclusive
- `text[start:end]`가 실제 span 문자열과 일치해야 함(불일치면 폐기/로그 권장)
