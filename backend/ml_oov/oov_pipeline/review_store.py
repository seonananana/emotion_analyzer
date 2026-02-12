from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import yaml


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_yaml(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        # 깨진 YAML이면 빈 상태로 시작(원하면 여기서 raise로 바꿔도 됨)
        return {}


def _dump_yaml(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


@dataclass
class ReviewItem:
    status: str  # "pending" | "approved" | "rejected"
    notes: str = ""
    updated_at: str = ""


class ReviewStore:
    """
    review yaml format:
    {
      "<key>": {status: "approved", notes: "...", updated_at: "..."},
      ...
    }
    """

    def __init__(self, review_path: str):
        self.review_path = review_path

    def load(self) -> Dict[str, ReviewItem]:
        raw = _load_yaml(self.review_path) or {}
        out: Dict[str, ReviewItem] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if not isinstance(k, str):
                    continue
                if not isinstance(v, dict):
                    continue
                status = str(v.get("status") or "pending")
                notes = str(v.get("notes") or "")
                updated_at = str(v.get("updated_at") or "")
                out[k] = ReviewItem(status=status, notes=notes, updated_at=updated_at)
        return out

    def save(self, items: Dict[str, ReviewItem]) -> None:
        raw: Dict[str, Dict[str, str]] = {}
        for k, it in items.items():
            raw[k] = {
                "status": it.status,
                "notes": it.notes or "",
                "updated_at": it.updated_at or "",
            }
        _dump_yaml(self.review_path, raw)

    def set_status(self, key: str, status: str, *, notes: Optional[str] = None) -> None:
        items = self.load()
        it = items.get(key) or ReviewItem(status="pending", notes="", updated_at="")
        it.status = status
        if notes is not None:
            it.notes = notes
        it.updated_at = _now()
        items[key] = it
        self.save(items)

    def get_status(self, key: str) -> str:
        items = self.load()
        return (items.get(key) or ReviewItem(status="pending")).status

    def get_notes(self, key: str) -> str:
        items = self.load()
        return (items.get(key) or ReviewItem(status="pending")).notes

    def count_by_status(self) -> Dict[str, int]:
        items = self.load()
        c = {"pending": 0, "approved": 0, "rejected": 0}
        for it in items.values():
            if it.status not in c:
                c[it.status] = 0
            c[it.status] += 1
        return c


def load_candidates_keys(candidate_yaml_path: str) -> List[str]:
    raw = _load_yaml(candidate_yaml_path) or {}
    if not isinstance(raw, dict):
        return []
    keys = [k for k in raw.keys() if isinstance(k, str)]
    return sorted(set(keys))


def get_approved_keys(candidate_yaml_path: str, review_yaml_path: str) -> List[str]:
    """
    승인된 후보만 반환.
    - candidates에 존재하고
    - review에서 status == "approved"
    """
    cand_keys = set(load_candidates_keys(candidate_yaml_path))
    store = ReviewStore(review_yaml_path)
    reviews = store.load()
    out = [k for k, it in reviews.items() if it.status == "approved" and k in cand_keys]
    return sorted(set(out))


def get_candidates_with_status(
    candidate_yaml_path: str,
    review_yaml_path: str,
    status: str,
) -> List[Tuple[str, str]]:
    """
    (key, notes) 목록 반환
    - status == "pending"이면: review에 없는 키도 pending으로 취급
    """
    cand_keys = load_candidates_keys(candidate_yaml_path)
    store = ReviewStore(review_yaml_path)
    reviews = store.load()

    out: List[Tuple[str, str]] = []
    for k in cand_keys:
        it = reviews.get(k)
        st = (it.status if it else "pending")
        if st == status:
            out.append((k, (it.notes if it else "")))
    return out
