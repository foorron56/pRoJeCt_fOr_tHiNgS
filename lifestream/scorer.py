"""Learned risk scorer — a trained, fully transparent linear model.

The rule engines (risk_engine / flood_engine) hand-weight their features.
This module is the **learned** alternative: a logistic regression trained on
labeled simulator data (see ``training.py`` and ``scripts/train.py``) that
maps the same behavioral features to a calibrated probability of danger.

Why logistic regression and not a deep net?
  * every point of score stays explainable: score = sigmoid(w·x + b), and the
    per-feature contributions w_i * x_i are surfaced in ``RiskState.terms``
    for the lifeguard HUD,
  * it trains in seconds on a laptop (pure numpy, no new dependencies),
  * small data (simulated scenarios) cannot support a high-capacity model
    honestly — a linear model trained on weak labels beats a black box
    pretending otherwise.

Model file format (JSON, versioned)::

    {
      "kind": "pool" | "flood",
      "version": 1,
      "feature_names": [...],          # order matters
      "weights": [...],                # standardized-space coefficients
      "bias": float,
      "mean": [...], "std": [...],     # feature standardization
      "alert_threshold": float,        # calibrated operating point
      "metrics": {...},                # validation metrics at that threshold
      "trained_at": "ISO timestamp",
      "n_train": int, "n_val": int, "seed": int
    }
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

MODEL_VERSION = 1


class LearnedScorer:
    """Loads a trained model JSON and scores feature dicts."""

    def __init__(self, model: Dict) -> None:
        self.model = model
        self.kind: str = model["kind"]
        self.names: List[str] = list(model["feature_names"])
        self.weights: List[float] = [float(w) for w in model["weights"]]
        self.bias: float = float(model["bias"])
        self.mean: List[float] = [float(m) for m in model.get("mean", [])]
        self.std: List[float] = [float(s) for s in model.get("std", [])]
        self.alert_threshold: float = float(model.get("alert_threshold", 0.5))

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str) -> "LearnedScorer":
        with open(path, "r", encoding="utf-8") as fh:
            model = json.load(fh)
        if int(model.get("version", 0)) != MODEL_VERSION:
            raise ValueError(f"model {path!r}: unsupported version "
                             f"{model.get('version')}")
        if len(model["weights"]) != len(model["feature_names"]):
            raise ValueError(f"model {path!r}: weights/features mismatch")
        return cls(model)

    @staticmethod
    def available(path: str) -> bool:
        return os.path.isfile(path)

    # ------------------------------------------------------------------ #

    def score(self, features: Dict[str, float]) -> float:
        """Calibrated danger probability in [0, 1] for one frame."""
        z = self.bias
        for i, name in enumerate(self.names):
            x = max(0.0, min(1.0, float(features.get(name, 0.0))))
            s = self.std[i] if i < len(self.std) and self.std[i] > 1e-9 else 1.0
            m = self.mean[i] if i < len(self.mean) else 0.0
            z += self.weights[i] * ((x - m) / s)
        return _sigmoid(z)

    def contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """Per-feature logit contributions (for explainable HUD terms)."""
        out: Dict[str, float] = {}
        for i, name in enumerate(self.names):
            x = max(0.0, min(1.0, float(features.get(name, 0.0))))
            s = self.std[i] if i < len(self.std) and self.std[i] > 1e-9 else 1.0
            m = self.mean[i] if i < len(self.mean) else 0.0
            out[name] = self.weights[i] * ((x - m) / s)
        return out

    def describe(self) -> str:
        m = self.model.get("metrics", {})
        return (f"learned[{self.kind}] val_f1={m.get('f1', float('nan')):.2f} "
                f"val_recall={m.get('recall', float('nan')):.2f} "
                f"val_fp_rate={m.get('fp_rate', float('nan')):.3f} "
                f"threshold={self.alert_threshold:.2f}")

    def save(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.model, fh, indent=2, ensure_ascii=False)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
