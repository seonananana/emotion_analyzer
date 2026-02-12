from __future__ import annotations

import argparse
from typing import Optional

from backend.ml_oov.oov_pipeline.config import load_config
from backend.ml_oov.oov_pipeline.review_store import (
    ReviewStore,
    load_candidates_keys,
    get_candidates_with_status,
    get_approved_keys,
)
import re

def _norm_key(s: str) -> str:
    # 모든 공백(스페이스/탭/NBSP 등)을 1칸으로 정규화하고 양끝 제거
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE)
    return s

def _norm_key_nospace(s: str) -> str:
    # 공백 전부 제거한 버전(친척언니 vs 친척 언니 대응)
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s, flags=re.UNICODE)
    return s

def _resolve_key(user_key: str, candidate_keys: list[str]) -> tuple[str | None, list[str]]:
    """
    returns: (resolved_key or None, candidates list)
    - 정확히 매칭되면 resolved_key
    - 아니면 공백정규화/무공백 기준으로 후보를 찾아서
      1개면 resolved_key, 여러개면 후보 리스트 반환
    """
    if user_key in candidate_keys:
        return user_key, []

    nk = _norm_key(user_key)
    nk2 = _norm_key_nospace(user_key)

    exact_norm = [k for k in candidate_keys if _norm_key(k) == nk]
    if len(exact_norm) == 1:
        return exact_norm[0], []
    if len(exact_norm) > 1:
        return None, exact_norm

    nospace = [k for k in candidate_keys if _norm_key_nospace(k) == nk2]
    if len(nospace) == 1:
        return nospace[0], []
    if len(nospace) > 1:
        return None, nospace

    return None, []

def _default_paths(args) -> tuple[str, str]:
    cfg = load_config()
    candidate_path = args.candidates or getattr(cfg, "candidate_path", "backend/data/oov_candidates.yaml")
    review_path = args.review or "backend/data/oov_candidates_review.yaml"
    return candidate_path, review_path


def cmd_list(args) -> None:
    candidates_path, review_path = _default_paths(args)
    store = ReviewStore(review_path)

    keys = load_candidates_keys(candidates_path)
    if not keys:
        print(f"[ERR] no candidates found: {candidates_path}")
        return

    # ✅ status 지정이면 rows 만들고 -> contains 필터 -> 출력
    if args.status:
        rows = get_candidates_with_status(candidates_path, review_path, args.status)

        if getattr(args, "contains", None):
            q = _norm_key_nospace(args.contains)
            rows = [(k, notes) for (k, notes) in rows if q in _norm_key_nospace(k)]

        print(f"[OK] candidates={len(keys)}  status={args.status}  rows={len(rows)}")
        for k, notes in rows[: args.limit]:
            if notes:
                print(f"- {k}  # {notes}")
            else:
                print(f"- {k}")
        if len(rows) > args.limit:
            print(f"... ({len(rows) - args.limit} more)")
        return

    # ✅ status 미지정이면 요약만(+contains는 무시)
    counts = store.count_by_status()
    pending_rows = get_candidates_with_status(candidates_path, review_path, "pending")
    print(f"[OK] candidates={len(keys)}")
    print(f"- pending : {len(pending_rows)}")
    print(f"- approved: {counts.get('approved', 0)}")
    print(f"- rejected: {counts.get('rejected', 0)}")
    print(f"[OK] review_file={review_path}")


def _set_status(args, status: str) -> None:
    candidates_path, review_path = _default_paths(args)
    store = ReviewStore(review_path)

    keys_list = load_candidates_keys(candidates_path)
    resolved, cands = _resolve_key(args.key, keys_list)

    if resolved is None:
        print(f"[ERR] key not found in candidates (even after normalization): {args.key}")
        if cands:
            print("[HINT] did you mean one of these?")
            for k in cands[:20]:
                print(f" - {k!r}")
            if len(cands) > 20:
                print(f"... ({len(cands)-20} more)")
        print(f"      candidates={candidates_path}")
        return

    store.set_status(resolved, status, notes=args.notes)
    print(f"[OK] {status}: {resolved}")
    if args.notes:
        print(f"[OK] notes: {args.notes}")
    print(f"[OK] review_file={review_path}")


def cmd_approve(args) -> None:
    _set_status(args, "approved")


def cmd_reject(args) -> None:
    _set_status(args, "rejected")


def cmd_pending(args) -> None:
    _set_status(args, "pending")


def cmd_export_approved(args) -> None:
    candidates_path, review_path = _default_paths(args)
    keys = get_approved_keys(candidates_path, review_path)
    print(f"[OK] approved={len(keys)}")
    for k in keys[: args.limit]:
        print(k)
    if len(keys) > args.limit:
        print(f"... ({len(keys) - args.limit} more)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Review/approve/reject OOV candidates (sidecar review yaml).")
    ap.add_argument("--candidates", default=None, help="candidates yaml path (default: cfg.candidate_path)")
    ap.add_argument("--review", default=None, help="review yaml path (default: backend/data/oov_candidates_review.yaml)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List candidates by status or show summary.")
    p_list.add_argument("--status", default=None, choices=["pending", "approved", "rejected"])
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)
    p_list.add_argument("--contains", default=None, help="substring filter (unicode whitespace-insensitive)")

    p_ap = sub.add_parser("approve", help="Approve a candidate key.")
    p_ap.add_argument("key")
    p_ap.add_argument("--notes", default=None)
    p_ap.set_defaults(func=cmd_approve)

    p_rj = sub.add_parser("reject", help="Reject a candidate key.")
    p_rj.add_argument("key")
    p_rj.add_argument("--notes", default=None)
    p_rj.set_defaults(func=cmd_reject)

    p_pd = sub.add_parser("pending", help="Set a candidate back to pending.")
    p_pd.add_argument("key")
    p_pd.add_argument("--notes", default=None)
    p_pd.set_defaults(func=cmd_pending)

    p_exp = sub.add_parser("export-approved", help="Print approved keys (for piping/other scripts).")
    p_exp.add_argument("--limit", type=int, default=2000)
    p_exp.set_defaults(func=cmd_export_approved)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
