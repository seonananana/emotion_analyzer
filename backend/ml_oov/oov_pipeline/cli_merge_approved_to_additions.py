from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
import yaml

from backend.ml_oov.oov_pipeline.config import load_config
from backend.ml_oov.oov_pipeline.review_store import get_approved_keys

DEFAULT_REVIEW = "backend/data/oov_candidates_review.yaml"
DEFAULT_ADDITIONS = "backend/data/부정_감성_사전_oov_additions.yaml"

# ✅ 기본 허용 prefix
DEFAULT_ALLOW_PREFIX = ("CAND_OOV", "NEG", "NEG_OOV")

# ✅ 너무 일반적인 단어/불용어는 자동추가 금지(필요하면 늘려)
DEFAULT_BLOCKLIST = {
    "마음",
    "생각",
    "사람",
    "시간",
    "오늘",
    "내일",
    "정말",
    "진짜",
    "너무",
    "그냥",
    "이제",
}

DEFAULT_MIN_LEN = 2
DEFAULT_MAX_LEN = 12

DEFAULT_MIN_COUNT = 3
DEFAULT_MIN_EXAMPLES = 1


# ✅ 스팬 -> base 정규화 규칙
# - "이라서/라서/해서/고..." 같은 접속/이유 어미만 제거
# - "게/도록" 같은 부사화는 의미 훼손 위험이 커서 제거하지 않음 (미치게 -> 미치 방지)
_SUFFIX_PATTERNS = [
    # 이유/접속 (핵심)
    r"(?:이라서|라서|해서|하여서|해서는|했어서|했어서요)$",
    # 연결어미 (핵심)
    r"(?:고|고서|고도|고는|고요|고나서|고나니)$",
]
_SUFFIX_RE = re.compile("|".join(_SUFFIX_PATTERNS))
_MAX_STRIP_ROUNDS = 2


def _load_yaml(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dump_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _load_candidates_map(candidates_path: str) -> dict[str, dict]:
    """
    candidates yaml은 2가지 형태 모두 대응:
    1) { key: {key,text,label,count,...}, ... }
    2) {"records":[{key:..., ...}, ...]}
    """
    p = Path(candidates_path)
    data = _load_yaml(p)

    out: dict[str, dict] = {}

    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        for r in data["records"]:
            if isinstance(r, dict):
                kk = (r.get("key") or "").strip()
                if kk:
                    out[kk] = r
        return out

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                kk = (v.get("key") or k or "").strip()
                if kk:
                    out[kk] = v
        return out

    return {}


def _is_allowed_label(label: str, allow_prefix: tuple[str, ...]) -> bool:
    lab = str(label or "")

    for pref in allow_prefix:
        if lab.startswith(pref):
            return True

    if lab.startswith("STAT_RANK_"):
        up = lab.upper()
        if ("NEGATION" in up) or ("NEG_VS" in up) or ("_NEG_" in up) or up.endswith("_NEG"):
            return True

    return False


def _normalize_key(key: str) -> str:
    """
    스팬(극혐이라서) -> base(극혐) 로 최대한 안전하게 축약.
    - 과도한 stemming은 위험해서 접속/이유 어미 일부만 제거.
    """
    orig = (key or "").strip()
    s = orig
    if not s:
        return s

    for _ in range(_MAX_STRIP_ROUNDS):
        ns = _SUFFIX_RE.sub("", s).strip()
        if ns == s:
            break
        s = ns

    # ✅ 너무 짧아지면(예: 한 글자) 오히려 의미 훼손이므로 원본 유지
    if len(s) < 2:
        return orig

    return s


def _passes_quality_gate(
    key: str,
    rec: dict,
    *,
    min_count: int,
    min_examples: int,
    min_len: int,
    max_len: int,
    blocklist: set[str],
) -> tuple[bool, str]:
    k = (key or "").strip()
    if not k:
        return False, "empty"

    if k in blocklist:
        return False, "blocklist"

    if len(k) < min_len:
        return False, f"too_short(<{min_len})"
    if len(k) > max_len:
        return False, f"too_long(>{max_len})"

    cnt = rec.get("count", 0)
    try:
        cnt_i = int(cnt)
    except Exception:
        cnt_i = 0
    if cnt_i < min_count:
        return False, f"count<{min_count} (count={cnt_i})"

    ex = rec.get("examples", [])
    ex_n = len(ex) if isinstance(ex, list) else 0
    if ex_n < min_examples:
        return False, f"examples<{min_examples} (examples={ex_n})"

    return True, "ok"


def main():
    ap = argparse.ArgumentParser(
        description="Merge approved keys into additions lexicon yaml (with anti-anchor gates + span->base normalization)."
    )
    ap.add_argument("--candidates", default=None, help="candidates yaml path (default: cfg.candidate_path)")
    ap.add_argument("--review", default=DEFAULT_REVIEW, help="review yaml path")
    ap.add_argument("--out", default=DEFAULT_ADDITIONS, help="additions lexicon yaml path")
    ap.add_argument("--dry", action="store_true", help="dry-run (no write)")

    ap.add_argument(
        "--allow_prefix",
        default=",".join(DEFAULT_ALLOW_PREFIX),
        help="comma-separated label prefixes allowed (default: CAND_OOV,NEG,NEG_OOV)",
    )

    ap.add_argument("--min_count", type=int, default=DEFAULT_MIN_COUNT)
    ap.add_argument("--min_examples", type=int, default=DEFAULT_MIN_EXAMPLES)
    ap.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument(
        "--blocklist",
        default=",".join(sorted(DEFAULT_BLOCKLIST)),
        help="comma-separated blocklist keys",
    )

    ap.add_argument("--no_normalize", action="store_true", help="do not normalize span keys into base keys")
    args = ap.parse_args()

    allow_prefix = tuple(x.strip() for x in (args.allow_prefix or "").split(",") if x.strip()) or DEFAULT_ALLOW_PREFIX
    blocklist = set(x.strip() for x in (args.blocklist or "").split(",") if x.strip())

    cfg = load_config()
    candidates_path = args.candidates or getattr(cfg, "candidate_path", "backend/data/oov_candidates.yaml")

    approved = get_approved_keys(candidates_path, args.review)
    print(f"[OK] approved keys: {len(approved)}")
    print(f"[OK] allow_prefix={allow_prefix}")
    print(
        f"[OK] gates: min_count={args.min_count}, min_examples={args.min_examples}, "
        f"len={args.min_len}..{args.max_len}, blocklist={len(blocklist)}"
    )
    print(f"[OK] normalize: {not args.no_normalize}")

    cand_map = _load_candidates_map(candidates_path)

    out_path = Path(args.out)
    cur = _load_yaml(out_path)
    if not isinstance(cur, dict):
        cur = {}
    words = cur.get("words")
    if not isinstance(words, list):
        words = []
    existing = set(str(w).strip() for w in words if isinstance(w, str))

    meta = cur.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        prov = {}

    added: list[str] = []
    skipped_not_found: list[str] = []
    skipped_not_allowed: list[tuple[str, str]] = []
    skipped_quality: list[tuple[str, str]] = []
    norm_examples: list[tuple[str, str]] = []

    for orig_k in approved:
        orig_k = str(orig_k).strip()
        if not orig_k:
            continue

        rec = cand_map.get(orig_k)
        if not rec:
            skipped_not_found.append(orig_k)
            continue

        lab = str(rec.get("label") or "")
        if not _is_allowed_label(lab, allow_prefix):
            skipped_not_allowed.append((orig_k, lab))
            continue

        ok, reason = _passes_quality_gate(
            orig_k,
            rec,
            min_count=args.min_count,
            min_examples=args.min_examples,
            min_len=args.min_len,
            max_len=args.max_len,
            blocklist=blocklist,
        )
        if not ok:
            skipped_quality.append((orig_k, reason))
            continue

        if args.no_normalize:
            norm_k = orig_k
        else:
            norm_k = _normalize_key(orig_k)

        norm_k = (norm_k or "").strip()
        if not norm_k:
            skipped_quality.append((orig_k, "normalized_empty"))
            continue

        if norm_k in blocklist:
            skipped_quality.append((orig_k, f"normalized_to_blocklist({norm_k})"))
            continue

        if len(norm_k) < args.min_len or len(norm_k) > args.max_len:
            skipped_quality.append((orig_k, f"normalized_len_out_of_range({norm_k})"))
            continue

        prov.setdefault(norm_k, [])
        if orig_k not in prov[norm_k]:
            prov[norm_k].append(orig_k)
            prov[norm_k] = prov[norm_k][:20]

        if norm_k in existing:
            continue

        words.append(norm_k)
        existing.add(norm_k)
        added.append(norm_k)

        if norm_k != orig_k:
            norm_examples.append((orig_k, norm_k))

    cur["words"] = words
    meta["provenance"] = prov
    cur["meta"] = meta

    print(f"[OK] will add (allowed + gate passed): {len(added)}")
    for k in added[:50]:
        print(" +", k)
    if len(added) > 50:
        print(f"... ({len(added)-50} more)")

    if skipped_not_found:
        print(f"[WARN] skipped (not found in candidates): {len(skipped_not_found)}")
        for k in skipped_not_found[:20]:
            print(" -", k)

    if skipped_not_allowed:
        print(f"[WARN] skipped (label not allowed): {len(skipped_not_allowed)}")
        for k, lab in skipped_not_allowed[:30]:
            print(" -", k, "label=", lab)

    if skipped_quality:
        print(f"[WARN] skipped (quality gate): {len(skipped_quality)}")
        for k, reason in skipped_quality[:30]:
            print(" -", k, "reason=", reason)
        if len(skipped_quality) > 30:
            print(f"... ({len(skipped_quality)-30} more)")

    if norm_examples:
        print(f"[INFO] normalized examples: {len(norm_examples)}")
        for o, n in norm_examples[:30]:
            print(f" ~ {o} -> {n}")

    if args.dry:
        print("[DRY] not writing")
        return

    _dump_yaml(out_path, cur)
    print(f"[OK] wrote: {out_path}  (words={len(words)})")


if __name__ == "__main__":
    main()
