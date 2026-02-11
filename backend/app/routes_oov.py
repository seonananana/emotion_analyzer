# backend/app/routes_oov.py
from fastapi import APIRouter, Query
from pathlib import Path
import json
import yaml

router = APIRouter()

@router.get("/ui/rejected_candidates")
def rejected_candidates(limit: int = Query(200, ge=1, le=5000)):
    p = Path("backend/data/rejected_candidates.jsonl")
    if not p.exists():
        return {"items": [], "count": 0}

    lines = p.read_text(encoding="utf-8").splitlines()
    tail = lines[-limit:]
    items = []
    for ln in tail:
        try:
            items.append(json.loads(ln))
        except Exception:
            continue
    return {"items": items, "count": len(items)}

@router.get("/ui/oov_candidates")
def oov_candidates():
    p = Path("backend/data/oov_candidates.yaml")
    if not p.exists():
        return {"records": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data if data else {"records": []}
