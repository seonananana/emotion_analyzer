from __future__ import annotations

import argparse
import os
from datetime import datetime

import yaml
import matplotlib.pyplot as plt
from openpyxl import Workbook

import matplotlib
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

def load_yaml_candidates(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
    rows = list(data.values())
    rows.sort(key=lambda r: (-int(r.get("count", 0)), -float(r.get("avg_confidence", 0.0)), r.get("key", "")))
    return rows


def write_excel(out_xlsx: str, items):
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "oov_candidates"
    ws.append(["rank", "key", "count", "avg_confidence", "label", "examples"])
    for i, r in enumerate(items, 1):
        ex = " | ".join((r.get("examples") or [])[:3])
        ws.append([i, r.get("key"), r.get("count"), r.get("avg_confidence"), r.get("label"), ex])
    wb.save(out_xlsx)


def write_chart(out_png: str, items, top_k: int = 50):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    top = items[:top_k]
    counts = [int(r.get("count",0)) for r in top]

    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), [f"TOP{i+1}" for i in range(len(counts))], rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_html(out_html: str, items, chart_png: str):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    rows = []
    for i, r in enumerate(items, 1):
        ex = "<br/>".join((r.get("examples") or [])[:2])
        rows.append(
            f"<tr><td>{i}</td><td>{r.get('key')}</td><td>{r.get('count')}</td>"
            f"<td>{float(r.get('avg_confidence',0.0)):.3f}</td><td>{r.get('label')}</td><td>{ex}</td></tr>"
        )

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>OOV Candidates Report</title>
<style>
body {{ font-family: Arial, sans-serif; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h2>OOV Candidates Report</h2>
<div class="small">generated: {datetime.now().isoformat(timespec="seconds")}</div>
<p><img src="{os.path.basename(chart_png)}" style="max-width: 900px; width: 100%;"/></p>
<table>
<thead><tr><th>rank</th><th>key</th><th>count</th><th>avg_conf</th><th>label</th><th>examples</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_yaml", default="backend/data/oov_candidates.yaml")
    ap.add_argument("--out_dir", default="backend/reports/oov")
    ap.add_argument("--top", type=int, default=300)
    args = ap.parse_args()

    items = load_yaml_candidates(args.in_yaml)[: args.top]

    out_xlsx = os.path.join(args.out_dir, "report.xlsx")
    out_png = os.path.join(args.out_dir, "top50.png")
    out_html = os.path.join(args.out_dir, "report.html")

    write_excel(out_xlsx, items)
    write_chart(out_png, items, top_k=min(50, len(items)))
    write_html(out_html, items, out_png)

    print("[report] wrote:")
    print(" -", out_xlsx)
    print(" -", out_png)
    print(" -", out_html)


if __name__ == "__main__":
    main()
