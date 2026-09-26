"""Training pipeline for the learned LifeStream risk scorers.

Everything runs on a laptop CPU with numpy only:

  1. ``build_pool_dataset``  — rolls thousands of *randomized* swim episodes
     through the same pose model the demo simulator uses (posture, arm
     amplitude/frequency, head dip, sinking, going quiet), extracts features
     with the REAL ``SwimFeatureExtractor``, and labels each frame with a
     physics heuristic of drowning (upright + thrashing + head dipping, or
     descending + silent + head under). Randomization — not the four fixed
     demo scripts — is what makes the learned weights generalize instead of
     memorizing the demo.
  2. ``build_flood_dataset`` — randomized flood actors (wading, drifting,
     clinging, stranded, safe floating) + a deterministic flow stand-in,
     labeled with the same phases the hand engine encodes.
  3. ``train_logistic``      — standardized L2 logistic regression, class
     balanced, full-batch gradient descent with a fixed seed.
  4. ``evaluate``            — precision/recall/F1 + fp-rate at a threshold.

The physics labels are *weak* labels: noisy, correlated with real drowning
mechanics, and good enough to learn weights that rank danger correctly. The
roadmap item is retraining the same scorer on human-verified real footage.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import SwimFeatureExtractor, SwimObservation
from .flow import FlowResult

# ---------------------------------------------------------------------- #
# dataset types

FEATURES_POOL = ("vertical_posture", "limb_distress", "stillness",
                 "submersion", "head_dip")
FEATURES_FLOOD = ("drift_ratio", "struggle", "submersion", "flow_exposure")

POOL_FEATURE_NAMES = list(FEATURES_POOL) + ["isolated"]
# "stranded" is the engineered stall gate — computed identically at runtime
# by FloodRiskEngine (flow strong AND person stalled) — so the learned model
# can express the flow x speed interaction a linear model otherwise cannot.
FLOOD_FEATURE_NAMES = list(FEATURES_FLOOD) + ["speed_ms", "flow_ms",
                                              "stranded"]


@dataclass
class Dataset:
    """A labeled feature matrix (rows = frames)."""

    kind: str                                   # "pool" | "flood"
    names: List[str]
    X: np.ndarray                               # (n, d) raw features
    y: np.ndarray                               # (n,) 1 = danger, 0 = safe
    meta: List[Dict] = field(default_factory=list)   # per-row provenance


# ---------------------------------------------------------------------- #
# pool dataset

def _pool_label(arm: float, v: float, dip: float, mode: str,
                ease: float) -> int:
    """Ground-truth label from the SIMULATED body state (not from the
    extracted features) — a simulator's whole point: the labeler knows the
    physical truth, the model only sees the imperfect measured features.

    Danger in progress:
      frantic — upright torso + vigorous thrashing + head dipping under
      sinking — the quiet arc, past the point of going under
    Treading water (upright but nose up, moderate motion) is safe; laps and
    playing are safe.
    """
    frantic = (arm >= 0.32 and v >= 0.50 and dip >= 0.10)
    sinking = (mode == "still_arc" and ease >= 0.40)
    return 1 if (frantic or sinking) else 0


def build_pool_dataset(episodes: int = 220, episode_s: float = 22.0,
                       fps: float = 25.0, seed: int = 13) -> Dataset:
    """Randomized swim episodes through the REAL PoolSimulator body model,
    features from the real production extractor, labels from the simulator's
    own ground-truth pose parameters.

    The labeler knows the physical truth (is the torso upright? are the arms
    thrashing? is the head dipping under? is the body sinking?) while the
    model only sees the imperfect windowed features — the honest division of
    labor that makes the learned weights transferable.
    """
    from .simulator import PoolSimulator

    rng = random.Random(seed)
    rows: List[List[float]] = []
    labels: List[int] = []
    meta: List[Dict] = []
    dt = 1.0 / fps
    n = int(episode_s * fps)

    for ep in range(episodes):
        mode = rng.choice(["safe", "safe", "distress", "still_arc"])
        params: Dict[str, float] = {}
        if mode == "distress":
            params = dict(
                v=rng.uniform(0.65, 1.00),
                arm=rng.uniform(0.34, 0.55),
                freq=rng.uniform(9.0, 14.0),
                kick=rng.uniform(0.08, 0.20),
                dip=rng.uniform(0.16, 0.34),
                dip_freq=rng.uniform(1.6, 3.0),
                phase=rng.choice([0.0, math.pi]),
                x_speed=0.0,
            )
        elif mode == "still_arc":
            params = dict(
                v=rng.uniform(0.15, 0.40),
                arm=rng.uniform(0.18, 0.34),
                freq=rng.uniform(4.5, 7.5),
                kick=rng.uniform(0.05, 0.12),
                phase=math.pi,
                x_speed=rng.uniform(-0.01, 0.01),
                sink_onset=rng.uniform(2.0, 6.0),
                sink_dur=rng.uniform(10.0, 16.0),
            )
        else:                                    # safe: laps-ish or playing
            params = dict(
                v=rng.uniform(0.02, 0.30),
                arm=rng.uniform(0.06, 0.28),
                freq=rng.uniform(4.0, 9.0),
                kick=rng.uniform(0.02, 0.10),
                dip=rng.uniform(0.0, 0.03),
                dip_freq=rng.uniform(1.6, 3.0),
                phase=rng.choice([0.0, math.pi]),
                x_speed=rng.uniform(0.01, 0.05) * rng.choice([-1.0, 1.0]),
            )

        sim = PoolSimulator()
        sim.actors = [sim.actors[0]]             # victim only
        sim.actors[0]["behavior"] = "custom"
        sim.actors[0]["pose_params"] = params
        ex = SwimFeatureExtractor()              # real production extractor

        onset = params.get("sink_onset")
        dur = params.get("sink_dur", 12.0)
        for i in range(n):
            bt = i * dt
            obs = sim.step(dt)[0]
            f = ex.update(obs)
            # ground truth from the scripted body state, not the features
            ease = (min(1.0, max(0.0, (bt - onset) / dur)) if onset is not None
                    else 0.0)
            ease = ease * ease * (3.0 - 2.0 * ease)
            y = _pool_label(params.get("arm", 0.0), params.get("v", 0.0),
                            params.get("dip", 0.0) if onset is None else 0.34,
                            mode, ease)
            rows.append([f[k] for k in FEATURES_POOL] + [0.0])  # isolated=0
            labels.append(y)
            meta.append({"episode": ep, "mode": mode, "t": round(bt, 2)})

    return Dataset("pool", POOL_FEATURE_NAMES,
                   np.asarray(rows, float), np.asarray(labels, int), meta)


# ---------------------------------------------------------------------- #
# flood dataset

def build_flood_dataset(episodes: int = 220, episode_s: float = 22.0,
                        fps: float = 25.0, seed: int = 29) -> Dataset:
    """Randomized flood actors -> labeled (X, y) with a known-flow stand-in.

    Modes include ``sweep_mid`` — a person wading in control who then slips
    and is carried away mid-episode — because "bị lũ cuốn" is a TRANSITION,
    not a static behavior. Labels flip with the scripted truth; rows are
    withheld during warm-up (feature windows not yet full) and for a short
    lag after the switch (measured features trail the behavior change), so
    the model never trains on contradictory (features, label) pairs.
    """
    rng = random.Random(seed)
    rows: List[List[float]] = []
    labels: List[int] = []
    meta: List[Dict] = []
    dt = 1.0 / fps
    n = int(episode_s * fps)
    ppm = 50.0
    warm = int(1.0 * fps)          # frames until the feature window is full

    for ep in range(episodes):
        mode = rng.choice(["wading", "wading", "drift", "drift",
                           "clinging", "stranded", "sweep_mid",
                           "sweep_mid"])
        flow_ms = rng.uniform(1.2, 3.2)
        flow = FlowResult(vx_ms=flow_ms, vy_ms=0.0, speed_ms=flow_ms,
                          coherence=0.95, ready=True)
        ux, uy = flow.unit

        # A wader can only push against a flood current at a fraction of
        # its speed — anyone moving at ~ the current's speed IS being
        # carried, so keep wading well below it and leave the high-drift
        # region to the swept class.
        if mode == "sweep_mid":
            switch_t = rng.uniform(6.0, 9.0)       # the slip: wading -> swept
            wade = dict(self_speed=rng.uniform(0.25, 0.55) * flow_ms,
                        dunk_frac=rng.uniform(0.0, 0.04),
                        struggle=rng.uniform(0.1, 0.4))
            swept = dict(self_speed=rng.uniform(0.85, 1.0) * flow_ms,
                         dunk_frac=rng.uniform(0.15, 0.45),
                         struggle=rng.uniform(0.3, 0.7))
        elif mode == "wading":
            wade = dict(self_speed=rng.uniform(0.25, 0.55) * flow_ms,
                        dunk_frac=rng.uniform(0.0, 0.04),
                        struggle=rng.uniform(0.1, 0.4))
        elif mode == "drift":
            swept = dict(self_speed=rng.uniform(0.85, 1.0) * flow_ms,
                         dunk_frac=rng.uniform(0.15, 0.45),
                         struggle=rng.uniform(0.3, 0.7))
        elif mode == "clinging":
            swept = dict(self_speed=rng.uniform(0.0, 0.15) * flow_ms,
                         dunk_frac=rng.uniform(0.28, 0.55),
                         struggle=rng.uniform(0.5, 0.95))
        else:                                       # stranded
            swept = dict(self_speed=rng.uniform(0.0, 0.10) * flow_ms,
                         dunk_frac=rng.uniform(0.0, 0.12),
                         struggle=rng.uniform(0.1, 0.5))

        ex = _FloodAccumulator(fps=fps, ppm=ppm)
        x = rng.uniform(0.0, 300.0)
        for i in range(n):
            bt = i * dt
            # pick the active phase + ground-truth label for this frame
            if mode == "sweep_mid":
                phase, label = (("wading", 0) if bt < switch_t
                                else ("swept", 1))
                params = wade if phase == "wading" else swept
            elif mode == "wading":
                phase, label, params = "wading", 0, wade
            else:
                phase, label, params = "swept", 1, swept

            if i < warm:                    # window not full yet: no decision
                active, label_row = params, None
            elif (mode == "sweep_mid" and switch_t - 0.5 <= bt
                    < switch_t + 1.0):      # features lag the slip: skip
                active, label_row = params, None
            else:
                active, label_row = params, label
            dunking = (bt % 4.0) < active["dunk_frac"] * 4.0

            visible = not dunking
            dx = active["self_speed"] * ux * dt * ppm
            x += dx
            sb = active["struggle"]
            wobble = (0.35 if phase in ("clinging", "stranded") else 0.55) \
                * sb * math.sin(bt * 9.0) * ppm * dt
            ex.push(visible=visible,
                    dx_px=(dx + wobble),
                    dy_px=0.3 * sb * math.sin(bt * 7.0) * ppm * dt,
                    flow=flow,
                    dt=dt,
                    struggle_energy=sb * (0.2 if dunking else 1.0))

            if label_row is None:           # warm-up / transition lag
                continue
            f = ex.features()
            rows.append([f[k] for k in FLOOD_FEATURE_NAMES])
            labels.append(label_row)
            meta.append({"episode": ep, "mode": phase, "t": round(bt, 2)})

    return Dataset("flood", FLOOD_FEATURE_NAMES,
                   np.asarray(rows, float), np.asarray(labels, int), meta)


class _FloodAccumulator:
    """Minimal per-person accumulator mirroring FloodFeatureExtractor's
    physics (it lives here to keep the trainer dependency-light and the
    synthetic loop fully deterministic)."""

    def __init__(self, fps: float, ppm: float) -> None:
        self.fps = max(1.0, fps)
        self.ppm = ppm
        self.window: List[Optional[bool]] = []     # visibility per frame
        self.pos: List[Tuple[float, float]] = []   # (x, y) px
        self._struggle = 0.0
        self.frames_seen = 0

    def push(self, visible: bool, dx_px: float, dy_px: float,
             flow: FlowResult, dt: float, struggle_energy: float) -> None:
        self._flow_ms = flow.speed_ms if flow.ready else 0.0
        self.window.append(visible)
        if len(self.window) > 30:
            self.window.pop(0)
        if visible:
            self.frames_seen += 1
            px, py = self.pos[-1] if self.pos else (0.0, 0.0)
            self.pos.append((px + dx_px, py + dy_px))
            if len(self.pos) > int(0.8 * self.fps):
                self.pos.pop(0)
            # frame-to-frame limb energy -> EMA (mirrors the real extractor)
            e = abs(dx_px - flow.speed_ms * self.ppm * dt) + abs(dy_px)
            self._struggle = 0.75 * self._struggle + 0.25 * min(
                1.0, e / 0.09) * 0.5 + 0.25 * struggle_energy * 0.5
        else:
            self._struggle *= 0.92

    def features(self) -> Dict[str, float]:
        vis = self.window
        misses = sum(1 for v in vis if not v)
        submersion = (misses / len(vis)) if vis else 0.0

        speed_ms = 0.0
        drift = 0.0
        if len(self.pos) >= 2:
            (x0, y0), (x1, y1) = self.pos[0], self.pos[-1]
            dt = (len(self.pos) - 1) / self.fps
            if dt > 0:
                speed_ms = math.hypot(x1 - x0, y1 - y0) / dt / self.ppm

        flow_ms = getattr(self, "_flow_ms", 0.0)
        if flow_ms > 0.3 and speed_ms > 0.08:
            drift = min(1.0, max(0.0, speed_ms / flow_ms))
        exposure = min(1.0, self.frames_seen / (20.0 * self.fps))
        flow_term = min(1.0, flow_ms / 2.5)
        flow_exposure = flow_term * (0.3 + 0.7 * exposure)
        # engineered stall gate — mirrors FloodRiskEngine exactly
        stranded = 1.0 if (flow_ms >= 1.2
                           and speed_ms <= 0.12 * flow_ms) else 0.0
        return {"drift_ratio": drift,
                "struggle": min(1.0, self._struggle),
                "submersion": submersion,
                "flow_exposure": flow_exposure,
                "exposure": exposure,
                "speed_ms": speed_ms,
                "flow_ms": flow_ms,
                "stranded": stranded}


# ---------------------------------------------------------------------- #
# trainer

@dataclass
class TrainResult:
    model: Dict
    metrics: Dict[str, float]


def train_logistic(ds: Dataset, alert_threshold: float = 0.5,
                   l2: float = 1e-3, epochs: int = 4000, lr: float = 0.5,
                   val_fraction: float = 0.25, seed: int = 7) -> TrainResult:
    """Standardized, class-balanced L2 logistic regression (numpy only)."""
    X, y, names = ds.X, ds.y.astype(float), ds.names
    n, d = X.shape

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std

    # deterministic split by episode so frames of one episode never straddle
    # train and validation (leakage would inflate the metrics)
    eps = np.array([m["episode"] for m in ds.meta]) if ds.meta \
        else np.arange(n)
    uniq = np.unique(eps)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_fraction))
    val_eps = set(uniq[:n_val].tolist())
    val_mask = np.array([e in val_eps for e in eps])
    tr_mask = ~val_mask

    Xtr, ytr = Xs[tr_mask], y[tr_mask]
    # class balancing: upweight the rare danger class to 50/50 effective
    n_pos = max(1.0, float(ytr.sum()))
    n_neg = max(1.0, float(len(ytr) - ytr.sum()))
    w_pos = n_neg / (n_pos + n_neg) * 2.0
    w_neg = n_pos / (n_pos + n_neg) * 2.0
    sw = np.where(ytr > 0.5, w_pos, w_neg)

    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = Xtr @ w + b
        p = _sigmoid_np(z)
        grad = Xtr.T @ ((p - ytr) * sw) / len(ytr) + l2 * w
        gb = float(np.mean((p - ytr) * sw))
        w -= lr * grad
        b -= lr * gb

    model = {
        "kind": ds.kind,
        "version": 1,
        "feature_names": list(names),
        "weights": [round(float(x), 6) for x in w],
        "bias": round(float(b), 6),
        "mean": [round(float(x), 6) for x in mean],
        "std": [round(float(x), 6) for x in std],
        "alert_threshold": alert_threshold,
        "trained_at": _utc_now(),
        "n_train": int(tr_mask.sum()),
        "n_val": int(val_mask.sum()),
        "seed": seed,
        "l2": l2,
        "epochs": epochs,
    }
    metrics = evaluate(model, Xs[val_mask], y[val_mask])
    model["metrics"] = metrics
    return TrainResult(model=model, metrics=metrics)


def evaluate(model: Dict, Xs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Precision/recall/F1/fp-rate at the model's operating threshold."""
    if len(y) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fp_rate": 1.0}
    w = np.asarray(model["weights"])
    b = float(model["bias"])
    p = _sigmoid_np(Xs @ w + b)
    pred = (p >= float(model["alert_threshold"])).astype(float)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    fp_rate = fp / (fp + tn) if fp + tn > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fp_rate": round(fp_rate, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ---------------------------------------------------------------------- #

def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
