# scripts/run_oov_on_excel.py
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

def pick_text_columns(df: pd.DataFrame):
    scores = []
    for c in df.columns:
        s = df[c]
        if s.dtype == "object":
            ss = s.dropna().astype(str)
            if len(ss) == 0:
                continue
            avg_len = ss.map(len).mean()
            nonempty = (ss.str.strip() != "").mean()
            scores.append((avg_len * nonempty, c))
    scores.sort(reverse=True)
    return [c for _, c in scores]

def run_cli(text: str, ckpt: str, threshold: float):
    cmd = [
        sys.executable, "-m", "backend.ml_oov.oov_pipeline.cli_run",
        "--text", text,
        "--threshold", str(threshold),
        "--ckpt", ckpt,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, (p.stderr or p.stdout)
    try:
        return json.loads(p.stdout), None
    except Exception:
        return None, "JSON parse failed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    xlsx = Path(args.input)
    df_or_dict = pd.read_excel(xlsx, sheet_name=args.sheet, engine="openpyxl")

    # sheet_name=None이면 dict(모든 시트)로 들어옴 → 첫 시트 사용
    if isinstance(df_or_dict, dict):
        # 첫 시트 선택
        first_name = next(iter(df_or_dict.keys()))
        df = df_or_dict[first_name]
        print(f"[INFO] sheet_name not provided. Using first sheet: {first_name}")
    else:
        df = df_or_dict

    df.columns = [str(c) for c in df.columns]

    text_col = args.text_col
    if not text_col:
        cols = pick_text_columns(df)
        if not cols:
            print("[ERR] no text column")
            print("cols:", list(df.columns))
            sys.exit(1)
        text_col = cols[0]
        print(f"[INFO] auto text_col = {text_col}")

    total = 0
    oov_rows = 0
    saved_sum = 0
    for i, v in df[text_col].items():
        if args.limit and total >= args.limit:
            break
        if pd.isna(v): 
            continue
        text = str(v).strip()
        if not text:
            continue
        total += 1
        res, err = run_cli(text, args.ckpt, args.threshold)
        if err:
            continue
        debug = res.get("debug", {})
        saved_sum += int(debug.get("candidate_records", 0) or 0)
        if res.get("oov_spans"):
            oov_rows += 1

    print("===== SUMMARY =====")
    print("rows_processed:", total)
    print("rows_with_oov_spans:", oov_rows)
    print("candidate_records_sum:", saved_sum)
    print("candidate_path: backend/data/oov_candidates.yaml")

if __name__ == "__main__":
    main()
