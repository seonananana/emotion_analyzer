"""Inference utilities (A→B integration point).

Public API:
  - predict_tokens(text, ckpt_dir=..., **kwargs) -> List[dict]
  - OOVTokenPredictor class (reusable, loads model once)

Returned token dict schema (minimal):
  {
    "token": str,
    "start": int,  # doc-level char start
    "end": int,    # doc-level char end (exclusive)
    "label": str,  # "O" | "B-NEG_OOV" | "I-NEG_OOV"
    "confidence": float
  }

Notes:
- Does NOT do span reconstruction / NMS / known-filter / thresholding (B-side).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import math

from .labels import ID2LABEL
from .sent_split import split_sentences

def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps] if s else [0.0 for _ in exps]

@dataclass
class OOVTokenPredictor:
    ckpt_dir: str
    max_length: int = 256
    device: str = "cpu"          # default enforce CPU
    sentence_split: bool = True  # split long texts into sentences

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch

        self._tok = AutoTokenizer.from_pretrained(self.ckpt_dir, use_fast=True)
        self._model = AutoModelForTokenClassification.from_pretrained(self.ckpt_dir)
        self._model.eval()
        self._device = torch.device(self.device)
        self._model.to(self._device)

    def predict_tokens(self, text: str) -> List[Dict]:
        import torch

        outputs: List[Dict] = []
        sents = [(text, 0, len(text))] if not self.sentence_split else split_sentences(text)

        for sent_text, sent_start, _sent_end in sents:
            if not sent_text:
                continue

            enc = self._tok(
                sent_text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(self._device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self._model(**enc).logits[0]  # [seq, num_labels]
            logits = logits.detach().cpu().tolist()

            tokens = self._tok.convert_ids_to_tokens(enc["input_ids"][0].detach().cpu().tolist())

            for tok, (os, oe), logit_vec in zip(tokens, offsets, logits):
                if os == 0 and oe == 0:
                    continue  # special tokens

                doc_s = sent_start + int(os)
                doc_e = sent_start + int(oe)

                probs = _softmax([float(x) for x in logit_vec])
                pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
                label = ID2LABEL.get(pred_id, "O")
                conf = float(probs[pred_id])

                outputs.append({
                    "token": tok,
                    "start": doc_s,
                    "end": doc_e,
                    "label": label,
                    "confidence": conf,
                })

        return outputs

def predict_tokens(
    text: str,
    ckpt_dir: str = "models/koelectra_oov",
    max_length: int = 256,
    device: str = "cpu",
    sentence_split: bool = True,
) -> List[Dict]:
    """Convenience wrapper. Prefer OOVTokenPredictor for repeated calls."""
    pred = OOVTokenPredictor(ckpt_dir=ckpt_dir, max_length=max_length, device=device, sentence_split=sentence_split)
    return pred.predict_tokens(text)
