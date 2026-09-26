"""Flood-risk engine: feature fusion + behavioral phases + alert lifecycle.

Phases
  WADING    — person in the water but in control (default, safe)
  SWEPT     — carried by the current: drift_ratio sustained >= swept_ratio
              while the local current exceeds the knockdown speed
  STRANDED  — stalled against a strong current (pinned to an obstacle or
              stuck on an island): speed <= 12% of the current, sustained
  CRITICAL  — repeatedly lost under the murky surface, or extreme score

The alert lifecycle mirrors the pool engine: a grace period before the first
fire and hysteresis before a clear, so a single noisy frame never flips the
alarm. The "stranded" term is derived here (it is a sustained gate, not a
per-frame feature) and then scored with the other signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config import DEFAULT_FLOOD_RISK, FloodRiskConfig
from .risk_engine import RiskState

WADING = "WADING"
SWEPT = "SWEPT"
STRANDED = "STRANDED"
CRITICAL = "CRITICAL"

ALERT_PHASES = (SWEPT, STRANDED, CRITICAL)


@dataclass
class FloodRiskState(RiskState):
    swept_cnt: int = 0
    stranded_cnt: int = 0
    critical_cnt: int = 0


class FloodRiskEngine:
    """Fuses flood features into a risk score and manages alerts."""

    def __init__(self, fps: float = 25.0,
                 config: FloodRiskConfig = DEFAULT_FLOOD_RISK,
                 scorer: Optional["LearnedScorer"] = None) -> None:
        self.cfg = config
        self.fps = max(1.0, fps)
        # Optional trained scorer (lifestream/scorer.py) — see pool engine.
        self.scorer = scorer
        self.states: Dict[int, FloodRiskState] = {}

    def update(self, person_id: int, features: Dict[str, float]) -> FloodRiskState:
        st = self.states.get(person_id)
        if st is None:
            st = self.states[person_id] = FloodRiskState()
        c = self.cfg

        flow_ms = max(0.0, float(features.get("flow_ms", 0.0)))
        speed_ms = max(0.0, float(features.get("speed_ms", 0.0)))

        f = {k: _clamp(features.get(k, 0.0)) for k in c.weights}
        # sustained-stall gate: barely moving inside a strong current
        stalled = (flow_ms >= c.stranded_flow_ms
                   and speed_ms <= c.stranded_speed_frac * flow_ms)
        f["stranded"] = 1.0 if stalled else 0.0

        if self.scorer is not None:
            st.score = _clamp(self.scorer.score(f))
            st.terms = {k: round(v, 4)
                        for k, v in self.scorer.contributions(f).items()}
        else:
            st.terms = {k: c.weights[k] * f[k] for k in c.weights}
            st.score = _clamp(sum(st.terms.values()))

        # swept: the water, not the person, controls the movement
        if f["drift_ratio"] >= c.swept_ratio and flow_ms >= c.knockdown_ms:
            st.swept_cnt += 1
        else:
            st.swept_cnt = max(0, st.swept_cnt - 1)

        st.stranded_cnt = (st.stranded_cnt + 1 if stalled
                           else max(0, st.stranded_cnt - 2))
        # repeated dunking compounds: one clean glimpse of the head between
        # waves must not wipe out everything the person went through
        st.critical_cnt = (st.critical_cnt + 1
                           if f["submersion"] >= c.critical_submersion
                           else max(0, st.critical_cnt - 1))

        if st.critical_cnt >= self._frames(c.critical_frames) or st.score >= 0.85:
            st.phase = CRITICAL
        elif st.swept_cnt >= self._frames(c.swept_frames):
            st.phase = SWEPT
        elif st.stranded_cnt >= self._frames(c.stranded_frames):
            st.phase = STRANDED
        else:
            st.phase = WADING

        self._update_alert(st)
        return st

    # ------------------------------------------------------------------ #

    def _update_alert(self, st: FloodRiskState) -> None:
        high = st.phase in ALERT_PHASES or st.score >= 0.60
        if high:
            st.low_frames = 0
            st.danger_frames += 1
            if (not st.alert_active
                    and st.danger_frames >= self._frames(self.cfg.danger_frames)):
                st.alert_active = True       # AlertFired event (pipeline emits)
        else:
            st.danger_frames = max(0, st.danger_frames - 1)
            st.low_frames += 1
            if (st.alert_active
                    and st.low_frames >= self._frames(self.cfg.hysteresis_frames)):
                st.alert_active = False      # AlertCleared event

    def _frames(self, n: int) -> int:
        """Config counts are in frames at 25 fps; rescale for other rates."""
        return max(3, int(round(n * self.fps / 25.0)))


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))
