"""Drowning-risk engine: weighted feature fusion + alert lifecycle.

Produces, per swimmer and per frame:
  * a risk score in [0, 1] (transparent weighted sum of 5 signals — every
    point of score is explainable to a judge or a lifeguard),
  * a behavioral PHASE (SWIMMING, LAPS, DISTRESS, STILL_INACTIVE, SINKING),
  * an alert lifecycle (risk detected -> alert fired -> cleared), with a
    danger grace period and hysteresis so a single noisy frame never fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import RiskConfig, DEFAULT_RISK

# PHASES
SWIMMING = "SWIMMING"
LAPS = "LAPS"
DISTRESS = "DISTRESS"
STILL_INACTIVE = "STILL_INACTIVE"
SINKING = "SINKING"

ACTIVE_PHASES = {SWIMMING, LAPS}


@dataclass
class RiskState:
    """Mutable per-swimmer state kept by the engine across frames."""

    score: float = 0.0
    phase: str = SWIMMING
    alert_active: bool = False
    danger_frames: int = 0
    calm_frames: int = 0
    low_frames: int = 0
    terms: Dict[str, float] = field(default_factory=dict)


class DrowningRiskEngine:
    """Fuses behavioral features into a risk score and manages alerts."""

    def __init__(self, fps: float = 25.0, config: RiskConfig = DEFAULT_RISK,
                 scorer: Optional["LearnedScorer"] = None) -> None:
        self.cfg = config
        self.fps = max(1.0, fps)
        # Optional trained scorer (lifestream/scorer.py): replaces the hand
        # weights with a calibrated logistic model; phases + alert lifecycle
        # below keep working unchanged on the learned score.
        self.scorer = scorer
        self.states: Dict[int, RiskState] = {}

    def update(self, swimmer_id: int, features: Dict[str, float],
               is_isolated: bool, has_pose: bool) -> RiskState:
        """Feed one frame of features; returns the swimmer's RiskState."""
        st = self.states.setdefault(swimmer_id, RiskState())
        w = self.cfg.weights

        f = {
            "vertical_posture": _clamp(features.get("vertical_posture", 0.0)),
            "limb_distress": _clamp(features.get("limb_distress", 0.0)),
            "stillness": _clamp(features.get("stillness", 0.0)),
            "submersion": _clamp(features.get("submersion", 0.0)),
            "head_dip": _clamp(features.get("head_dip", 0.0)),
            "isolated": 1.0 if is_isolated else 0.0,
        }
        # a missing pose read is itself suspicious, but must not fake features
        if not has_pose:
            f["vertical_posture"] *= 0.4
            f["limb_distress"] *= 0.3

        if self.scorer is not None:
            st.score = _clamp(self.scorer.score(f))
            st.terms = {k: round(v, 4)
                        for k, v in self.scorer.contributions(f).items()}
        else:
            score = sum(w[k] * f[k] for k in w)
            st.terms = {k: w[k] * f[k] for k in w}
            st.score = _clamp(score)

        self._update_phase(st, f, has_pose)
        self._update_alert(st)
        return st

    # ------------------------------------------------------------------ #

    def _update_phase(self, st: RiskState, f: Dict[str, float],
                      has_pose: bool) -> None:
        # DISTRESS must be earned every frame: frantic limbs + upright torso
        # + head repeatedly dipping under. The head-dip requirement separates
        # a drowning victim from vigorous-but-safe treading water.
        if (f["limb_distress"] >= 0.45 and f["vertical_posture"] >= 0.35
                and f["head_dip"] >= 0.35):
            st.phase = DISTRESS
            st.calm_frames = 0
            return
        if st.score >= 0.65:
            st.phase = SINKING
            return
        if f["stillness"] >= 0.50:
            st.calm_frames += 1
            if st.calm_frames >= self._frames(self.cfg.calm_frames):
                st.phase = STILL_INACTIVE
            return
        st.calm_frames = 0
        if st.phase in (STILL_INACTIVE, SINKING, DISTRESS):
            st.phase = SWIMMING
        if st.phase in ACTIVE_PHASES and f["vertical_posture"] < 0.2:
            # horizontal, steady progress across the pool = lap swimming
            st.phase = LAPS

    def _update_alert(self, st: RiskState) -> None:
        high = st.score >= 0.60 or st.phase in (DISTRESS, SINKING, STILL_INACTIVE)
        if high:
            st.low_frames = 0
            st.danger_frames += 1
            if not st.alert_active and st.danger_frames >= self.cfg.danger_frames:
                st.alert_active = True       # AlertFired event (pipeline emits)
        else:
            st.danger_frames = max(0, st.danger_frames - 1)
            st.low_frames += 1
            if st.alert_active and st.low_frames >= self.cfg.hysteresis_frames:
                st.alert_active = False      # AlertCleared event

    def _frames(self, seconds_like_count: int) -> int:
        """Config counts are in frames at 25 fps; rescale for other rates."""
        return max(3, int(round(seconds_like_count * self.fps / 25.0)))


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))
