"""OOV token tagging (A-side) package.

Public entrypoints:
- predict_tokens(text, ckpt_dir=..., **kwargs)
- OOVTokenPredictor
"""

from .predictor import OOVTokenPredictor, predict_tokens
