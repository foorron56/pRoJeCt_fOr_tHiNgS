"""Per-swimmer features that feed the drowning-risk engine.

A SwimObservation is one frame's raw pose reading for one tracked swimmer.
SwimFeatureExtractor converts a sliding window of observations into the five
behavioral signals lifeguards are trained to spot:

  1. vertical_posture — drowning victims are vertical, head back, unable to
     swim horizontally; lap swimmers are mostly horizontal.
  2. limb_distress   — frantic, wide-amplitude thrashing arms/legs.
  3. stillness       — an active swimmer that suddenly stops moving (Instinctive
     Drowning Response often looks like quiet vertical stillness, not shouting).
  4. submersion      — head/torso drifting below the estimated water surface.
  5. isolation       — distance to nearest other swimmer (no one within reach).

Convention: keypoint coordinates are bbox-local FRACTIONS in [0, 1]
(x / bbox_width, y / bbox_height) — the same normalized convention MediaPipe
uses for its landmarks. Every feature value is in [0, 1] where 1 = maximal
drowning evidence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import FeatureConfig, DEFAULT_FEATURES

# MediaPipe Pose landmark names relevant to LifeStream (subset kept explicit
# so the pure-logic layer has zero dependency on MediaPipe itself).
KEYPOINT_NAMES = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass
class SwimObservation:
    """One frame of raw pose data for one swimmer.

    keypoints: name -> (x, y) bbox-local fractions in [0, 1], or None when
    the keypoint is not visible. bbox is (x1, y1, x2, y2) in frame pixels.
    center_dx/dy: per-frame bbox-center travel in frame fractions (an
    optional tracker-provided signal; None when unknown).
    """

    frame_idx: int
    bbox: Sequence[float]
    keypoints: Dict[str, Optional] = field(default_factory=dict)
    visible: bool = False
    center_dx: Optional[float] = None
    center_dy: Optional[float] = None


class SwimFeatureExtractor:
    """Turns sliding windows of SwimObservations into risk features."""

    def __init__(self, config: FeatureConfig = DEFAULT_FEATURES) -> None:
        self.cfg = config
        self.history: deque = deque(maxlen=config.window)
        self._prev: Optional[SwimObservation] = None
        self._recent: deque = deque(maxlen=5)   # fast motion-energy EMA
        self._baseline = 0.0                    # peak-hold motion energy
        self._peak = 0.0                        # slow-decaying activity memory

    def update(self, obs: SwimObservation) -> Dict[str, float]:
        # per-frame motion energy with peak-hold baseline: activity raises the
        # baseline quickly; it decays slowly, so "stillness" stays high for
        # tens of seconds after a swimmer goes quiet (not just a 1-frame blip).
        energy = (_motion_energy([self._prev, obs])
                  if self._prev is not None else 0.0)
        self._prev = obs
        if energy > self._baseline:
            self._baseline = 0.6 * self._baseline + 0.4 * energy
        else:
            self._baseline *= 0.9993          # ~26 s half-life at 25 fps
        self._peak = max(self._peak * 0.9995, self._baseline)
        self._recent.append(energy)
        self.history.append(obs)
        return self.features()

    # ------------------------------------------------------------------ #
    # individual signals

    def _vertical_posture(self) -> float:
        """Head-up / feet-down orientation score (mean over window)."""
        scores: List[float] = []
        for obs in self.history:
            if not obs.visible:
                continue
            shoulder = _mid(obs, "left_shoulder", "right_shoulder")
            hip = _mid(obs, "left_hip", "right_hip")
            if shoulder is None or hip is None:
                continue
            dx, dy = abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1])
            ratio = dy / (dy + dx + 1e-6)      # ~1 = vertical, ~0 = horizontal
            scores.append(min(1.0, max(0.0, (ratio - 0.25) / 0.25)))
        return sum(scores) / len(scores) if scores else 0.0

    def _limb_distress(self) -> float:
        """Wide, fast vertical swings of wrists/ankles = frantic thrashing."""
        amp = _swing_amplitude(self.history, ("left_wrist", "right_wrist"))
        leg_amp = _swing_amplitude(self.history, ("left_ankle", "right_ankle"))
        return min(1.0, 0.7 * amp + 0.3 * leg_amp)

    def _stillness(self) -> float:
        """Recent motion vs. peak-hold baseline (sudden quiet = danger).

        Requires the swimmer to HAVE been active (peak above gate) so a
        buoy or lane rope that never moves never accumulates stillness.
        """
        if len(self.history) < max(8, self.cfg.window // 2):
            return 0.0
        if self._peak < 0.008:
            return 0.0
        recent = sum(self._recent) / len(self._recent)
        drop = 1.0 - min(1.0, recent / (self._baseline + 1e-6))
        return min(1.0, max(0.0, drop))

    def _submersion(self) -> float:
        """Two complementary drowning evidences, the max of which wins:

        * surface cut — head/torso keypoints below the estimated water
          surface band (top ``surface_band_frac`` of the bbox is above water);
        * descent — sustained downward drift of the bbox center: a sinking
          body keeps translating down frame after frame, which no safe
          behavior does (bobbing oscillates around zero mean).
        """
        scores: List[float] = []
        surface = self.cfg.surface_band_frac
        drifts: List[float] = []
        for obs in list(self.history)[-16:]:
            if obs.center_dy is not None:
                drifts.append(obs.center_dy)
            if not obs.visible:
                continue
            depth = 0.0
            nose = obs.keypoints.get("nose")
            sh = _mid(obs, "left_shoulder", "right_shoulder")
            if nose is not None:
                depth = max(depth, nose[1] / surface)
            if sh is not None:
                depth = max(depth, 0.8 * (sh[1] / surface))
            scores.append(min(1.0, max(0.0, (depth - 0.85) / 0.65)))
        surface_score = (sum(scores) / len(scores)) if scores else 0.0
        # 0.0015 frame-heights/frame ≈ one body length per ~7 s of steady sink
        descent = ((sum(drifts) / len(drifts)) / 0.0015) if drifts else 0.0
        descent = min(1.0, max(0.0, descent))
        return max(surface_score, descent)

    def _head_dip(self) -> float:
        """Vertical travel of the nose across the window.

        A drowning victim's head repeatedly bobs under; a treader's or a
        lap swimmer's nose stays in a narrow band. Used (in the risk engine)
        to separate frantic DISTRESS from vigorous-but-safe treading.
        """
        vals = [o.keypoints["nose"][1] for o in self.history
                if o.visible and o.keypoints.get("nose") is not None]
        if len(vals) < 4:
            return 0.0
        return min(1.0, (max(vals) - min(vals)) / 0.15)

    # ------------------------------------------------------------------ #

    def features(self) -> Dict[str, float]:
        return {
            "vertical_posture": self._vertical_posture(),
            "limb_distress": self._limb_distress(),
            "stillness": self._stillness(),
            "submersion": self._submersion(),
            "head_dip": self._head_dip(),
        }


def _mid(obs: SwimObservation, a: str, b: str):
    """Mean of a left/right keypoint pair; None if neither side is visible."""
    pa, pb = obs.keypoints.get(a), obs.keypoints.get(b)
    if pa is None and pb is None:
        return None
    if pa is None:
        return pb
    if pb is None:
        return pa
    return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)


def _swing_amplitude(history: Sequence[SwimObservation], names) -> float:
    """Mean peak-to-trough vertical travel of the given keypoints (fractions)."""
    amps: List[float] = []
    for name in names:
        vals = [o.keypoints[name][1] for o in history
                if o.visible and o.keypoints.get(name) is not None]
        if len(vals) >= 4:
            amps.append(max(vals) - min(vals))
    return sum(amps) / len(amps) if amps else 0.0


def _motion_energy(obs: Sequence[SwimObservation]) -> float:
    """Mean per-frame travel: keypoint steps plus bbox-center drift."""
    steps: List[float] = []
    for prev, cur in zip(obs, obs[1:]):
        moves = []
        if prev.visible and cur.visible:
            for name in KEYPOINT_NAMES:
                a, b = prev.keypoints.get(name), cur.keypoints.get(name)
                if a is not None and b is not None:
                    moves.append(abs(b[0] - a[0]) + abs(b[1] - a[1]))
        if cur.center_dx is not None:
            moves.append(abs(cur.center_dx) + abs(cur.center_dy or 0.0))
        if moves:
            steps.append(sum(moves) / len(moves))
    return sum(steps) / len(steps) if steps else 0.0
