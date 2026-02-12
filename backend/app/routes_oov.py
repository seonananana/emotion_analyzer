# backend/app/routes_oov.py
from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel
from pathlib import Path
import json
import yaml
import importlib
import subprocess
from backend.ml_oov.oov_pipeline.lexicon_matcher import build_lexicon_matcher
from backend.ml_oov.oov_pipeline.pipeline import run_oov_pipeline
from backend.ml_oov.oov_pipeline.review_store import ReviewStore

router = APIRouter()

# ✅ 캐시: 모델 predictor/lexicon_matcher는 서버 뜬 뒤 1번만 로딩
_PREDICTOR = None
_LEXICON_MATCHER = None

CANDIDATES_PATH = Path("backend/data/oov_candidates.yaml")
REVIEW_PATH = Path("backend/data/oov_candidates_review.yaml")


# -------------------------
# Helpers
# -------------------------

def _load_yaml_dict(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _candidates_records_as_list() -> list[dict]:
    """
    oov_candidates.yaml (현재 구조):
      { key1: {key,text,label,count,...}, key2: {...}, ... }
    -> 항상 list[record]로 normalize
    """
    data = _load_yaml_dict(CANDIDATES_PATH)
    records = []
    for _, v in data.items():
        if isinstance(v, dict):
            records.append(v)
    return records


def _review_map() -> dict:
    """
    oov_candidates_review.yaml:
      { "미치게": {"status": "approved", "notes": "...", "updated_at": "..."}, ... }
    """
    return _load_yaml_dict(REVIEW_PATH)


def _merge_candidates_with_review() -> list[dict]:
    """
    후보 records(list)에 review_status/notes/updated_at을 붙여 UI에서 바로 쓰게 함.
    """
    recs = _candidates_records_as_list()
    rev = _review_map()

    out = []
    for r in recs:
        key = (r.get("key") or r.get("text") or "").strip()
        rv = rev.get(key) if key else None
        rr = dict(r)
        rr["review_status"] = (rv or {}).get("status", "pending")
        rr["review_notes"] = (rv or {}).get("notes", "")
        rr["review_updated_at"] = (rv or {}).get("updated_at", "")
        out.append(rr)

    # 기본 정렬: pending 우선, 그 안에서는 count 높은 순
    out.sort(key=lambda x: (x.get("review_status") != "pending", -(x.get("count") or 0)))
    return out


def _set_review_status(key: str, status: str, notes: str | None = None) -> dict:
    key = (key or "").strip()
    if not key:
        return {"ok": False, "error": "empty key"}

    REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = ReviewStore(str(REVIEW_PATH))
    store.set_status(
        key,
        status,
        notes=(notes.strip() if isinstance(notes, str) and notes.strip() else None),
    )
    return {"ok": True, "key": key, "status": status, "review_file": str(REVIEW_PATH)}


# -------------------------
# 기존 GET들
# -------------------------

@router.get("/ui/rejected_candidates")
def rejected_candidates(limit: int = Query(200, ge=1, le=5000), include_processed: bool = False):
    p = Path("backend/data/rejected_candidates.jsonl")
    if not p.exists():
        return {"items": [], "count": 0}

    decisions_path = Path("backend/data/rejected_decisions.json")
    decisions = {}
    if decisions_path.exists():
        try:
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        except Exception:
            decisions = {}

    lines = p.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:]
    items = []
    for ln in tail:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        rid = obj.get("id")
        if rid and rid in decisions:
            obj["decision"] = decisions[rid]
            if not include_processed:
                continue
        items.append(obj)
    return {"items": items, "count": len(items)}


@router.get("/ui/oov_candidates")
def oov_candidates(include_review: bool = True):
    """
    UI에서 OOV 후보 목록을 가져옴.
    - 기본: 후보 + 리뷰상태 merge해서 내려줌 (include_review=True)
    - include_review=False면 후보만 list로 내려줌
    """
    if include_review:
        return {"records": _merge_candidates_with_review()}
    return {"records": _candidates_records_as_list()}


# -------------------------
# POST: rejected decide
# -------------------------

class DecideBody(BaseModel):
    id: str
    status: str  # "rejected" | "approved"
    note: str = ""


@router.post("/ui/rejected_candidates/decide")
def decide_rejected(body: DecideBody):
    decisions_path = Path("backend/data/rejected_decisions.json")
    decisions_path.parent.mkdir(parents=True, exist_ok=True)

    decisions = {}
    if decisions_path.exists():
        try:
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        except Exception:
            decisions = {}

    decisions[body.id] = {
        "status": body.status,
        "note": body.note,
        "decided_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }

    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "id": body.id, "status": body.status}


# -------------------------
# POST: OOV candidate review (UI 버튼용)
#   ✅ CLI(cli_review_candidates)와 동일하게 oov_candidates_review.yaml에 기록
# -------------------------

class ReviewBody(BaseModel):
    key: str
    notes: str = ""


@router.post("/ui/oov_candidates/approve")
def ui_approve_candidate(body: ReviewBody):
    return _set_review_status(body.key, "approved", body.notes)


@router.post("/ui/oov_candidates/reject")
def ui_reject_candidate(body: ReviewBody):
    return _set_review_status(body.key, "rejected", body.notes)


@router.post("/ui/oov_candidates/pending")
def ui_pending_candidate(body: ReviewBody):
    return _set_review_status(body.key, "pending", body.notes)


# -------------------------
# ✅ Quick Test API
# -------------------------

class RunBody(BaseModel):
    text: str
    threshold: float = 0.0


def _build_predictor_from_cli_run():
    """
    cli_run.py에서 predictor factory를 찾아 predictor(text) callable을 만든다.
    """
    m = importlib.import_module("backend.ml_oov.oov_pipeline.cli_run")

    candidates = [
        "build_predictor",  # ✅ cli_run.py에 build_predictor() 같은 함수가 있으면 그걸 사용
        "get_predictor",
        "make_predictor",
        "load_predictor",
        "create_predictor",
        "build_koelectra_predictor",
    ]

    for name in candidates:
        fn = getattr(m, name, None)
        if callable(fn):
            return fn()

    pred = getattr(m, "predictor", None)
    if callable(pred):
        return pred

    raise ImportError(
        "routes_oov: predictor factory를 cli_run.py에서 찾지 못했어. "
        "cli_run.py에 build_predictor() 같은 predictor 생성 함수를 1개 노출해줘."
    )


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = _build_predictor_from_cli_run()
    return _PREDICTOR


def _get_lexicon_matcher():
    global _LEXICON_MATCHER
    if _LEXICON_MATCHER is None:
        _LEXICON_MATCHER = build_lexicon_matcher("backend/data/부정_감성_사전_완전판.yaml")
    return _LEXICON_MATCHER


@router.post("/ui/oov_run")
def oov_run(body: RunBody):
    predictor = _get_predictor()
    lexicon_matcher = _get_lexicon_matcher()

    # ✅ UI 입력은 "항상 누적 저장"이 목적이므로 저장 옵션을 강제
    overrides = {
        "conf_threshold": float(body.threshold),

        # 후보 누적 저장 파일 고정
        "candidate_path": "backend/data/oov_candidates.yaml",

        # 지금 네 candidates 포맷(키=단어 dict) 기준이면 spans 추천
        "candidate_mode": "spans",

        # 프로젝트마다 이름이 다를 수 있어서 둘 다 넣어둠(있으면 먹고, 없으면 무시되게)
        "save_candidates": True,
        "write_candidates": True,
        "enable_candidate_writer": True,
        "enable_candidates": True,
    }

    res = run_oov_pipeline(
        body.text or "",
        predictor=predictor,
        lexicon_matcher=lexicon_matcher,
        target_labels=None,
        config_overrides=overrides,
    )

    def to_dict(x):
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "__dict__"):
            return dict(x.__dict__)
        return x

    return {
        "oov_spans": [to_dict(s) for s in res.oov_spans],
        "model_spans": [to_dict(s) for s in res.model_spans],
        "lexicon_spans": [to_dict(s) for s in res.lexicon_spans],
        "debug": res.debug,
    }

@router.post("/ui/oov_merge")
def ui_oov_merge():
    cmd = [
        "python3",
        "-m",
        "backend.ml_oov.oov_pipeline.cli_merge_approved_to_additions",
        "--min_count", "3",
        "--min_examples", "1"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }