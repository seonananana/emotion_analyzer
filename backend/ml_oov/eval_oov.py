"""Evaluate OOV tagging with Precision/Recall and Char-F1.

Usage:
  python -m backend.ml_oov.eval_oov \
    --gold backend/data/oov_gold/test.jsonl \
    --ckpt models/koelectra_oov \
    --out models/koelectra_oov/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

from .predictor import OOVTokenPredictor

def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def bio_tokens_to_spans(tokens: List[Dict]) -> List[Tuple[int,int,float]]:
    """BIO token predictions -> spans (robust). Returns [(start,end,mean_conf)]."""
    spans = []
    cur_s = None
    cur_e = None
    confs: List[float] = []

    def flush():
        nonlocal cur_s, cur_e, confs
        if cur_s is not None and cur_e is not None and cur_e > cur_s:
            spans.append((cur_s, cur_e, float(sum(confs)/len(confs)) if confs else 0.0))
        cur_s, cur_e, confs = None, None, []

    for t in tokens:
        lab = t["label"]
        s = int(t["start"]); e = int(t["end"]); c = float(t["confidence"])
        if lab == "O":
            flush()
        elif lab.startswith("B-"):
            flush()
            cur_s, cur_e, confs = s, e, [c]
        elif lab.startswith("I-"):
            if cur_s is None:
                cur_s, cur_e, confs = s, e, [c]
            else:
                cur_e = max(cur_e, e)
                confs.append(c)

    flush()
    return spans

def char_prf1(text_len: int, gold_spans: List[Tuple[int,int]], pred_spans: List[Tuple[int,int]]) -> Dict[str,float]:
    gold = [0]*text_len
    pred = [0]*text_len
    for s,e in gold_spans:
        s = max(0, min(text_len, int(s)))
        e = max(0, min(text_len, int(e)))
        for i in range(s,e):
            gold[i] = 1
    for s,e in pred_spans:
        s = max(0, min(text_len, int(s)))
        e = max(0, min(text_len, int(e)))
        for i in range(s,e):
            pred[i] = 1

    tp = sum(1 for g,p in zip(gold,pred) if g==1 and p==1)
    fp = sum(1 for g,p in zip(gold,pred) if g==0 and p==1)
    fn = sum(1 for g,p in zip(gold,pred) if g==1 and p==0)
    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec  = tp / (tp + fn) if (tp+fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec,"f1":f1}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    rows = _load_jsonl(args.gold)
    pred = OOVTokenPredictor(ckpt_dir=args.ckpt, max_length=args.max_length, device="cpu", sentence_split=True)

    agg_tp = agg_fp = agg_fn = 0
    debug = []

    for r in rows:
        doc_id = r.get("id")
        text = r["text"]
        gold_spans = [(int(s["start"]), int(s["end"])) for s in (r.get("oov_spans") or [])]

        token_preds = pred.predict_tokens(text)
        pred_spans_conf = bio_tokens_to_spans(token_preds)
        pred_spans = [(s,e) for (s,e,_) in pred_spans_conf]

        m = char_prf1(len(text), gold_spans, pred_spans)
        agg_tp += m["tp"]; agg_fp += m["fp"]; agg_fn += m["fn"]

        debug.append({"id": doc_id, "gold_spans": gold_spans, "pred_spans": pred_spans_conf})

    prec = agg_tp / (agg_tp + agg_fp) if (agg_tp+agg_fp) else 0.0
    rec  = agg_tp / (agg_tp + agg_fn) if (agg_tp+agg_fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0

    metrics = {"char_level": {"tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "precision": prec, "recall": rec, "f1": f1}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out.parent / "predictions.debug.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in debug), encoding="utf-8")
    print(f"[eval_oov] wrote metrics -> {args.out}")

if __name__ == "__main__":
    main()
