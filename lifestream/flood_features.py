"""Per-person behavioral features for floodwater risk.

Fed one SwimObservation per frame — or ``None`` when the person is lost under
the murky water — plus the local FlowResult, the extractor produces:

  drift_ratio    — velocity of the person ALONG the current divided by the
                   current speed. ~1.0 = the water is in control (swept),
                   ~0 = the person is moving independently or holding on.
  struggle       — EMA of high-frequency body motion (frantic flailing).
  submersion     — fraction of the recent window with NO observation: murky
                   floodwater hides a person the moment the head goes under.
  flow_exposure  — strong-current term x time-in-water ramp.
  speed_ms / flow_ms — context passed through for the engine's sustained
                   stall (stranded) gate.

All scored features are in [0, 1].
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

from .config import DEFAULT_FLOOD, FloodConfig
from .features import KEYPOINT_NAMES, SwimObservation
from .flow import FlowResult


class FloodFeatureExtractor:
    """Turns a sliding window of observations + flow into flood features."""

    def __init__(self, fps: float = 25.0,
                 config: FloodConfig = DEFAULT_FLOOD) -> None:
        self.fps = max(1.0, fps)
        self.cfg = config
        self.history: deque = deque(maxlen=config.window)      # obs | None
        self._pos: deque = deque(maxlen=max(2, int(0.8 * self.fps)))
        self._prev_visible: Optional[SwimObservation] = None
        self._struggle = 0.0
        self.frames_seen = 0

    def update(self, obs: Optional[SwimObservation],
               flow: Optional[FlowResult]) -> Dict[str, float]:
        self.history.append(obs)

        vel_px_s = (0.0, 0.0)
        if obs is None:
            self._struggle *= 0.92
        else:
            self.frames_seen += 1
            x1, y1, x2, y2 = obs.bbox
            self._pos.append((obs.frame_idx, 0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            if len(self._pos) >= 2:
                (f0, x0, y0), (f1, x1c, y1c) = self._pos[0], self._pos[-1]
                dt = (f1 - f0) / self.fps
                if dt > 0:
                    vel_px_s = ((x1c - x0) / dt, (y1c - y0) / dt)

            if obs.visible:
                if (self._prev_visible is not None
                        and obs.frame_idx == self._prev_visible.frame_idx + 1):
                    steps = []
                    for name in KEYPOINT_NAMES:
                        pa = self._prev_visible.keypoints.get(name)
                        pb = obs.keypoints.get(name)
                        if pa is not None and pb is not None:
                            steps.append(abs(pb[0] - pa[0]) + abs(pb[1] - pa[1]))
                    if steps:
                        e = sum(steps) / len(steps)
                        self._struggle = (0.75 * self._struggle
                                          + 0.25 * min(1.0, e / 0.09))
                self._prev_visible = obs
            else:
                self._struggle *= 0.92

        px_vx, px_vy = vel_px_s
        speed_ms = float(np.hypot(px_vx, px_vy)) / self.cfg.pixels_per_meter

        drift = 0.0
        flow_ms = flow.speed_ms if (flow is not None and flow.ready) else 0.0
        if flow_ms > 0.3 and speed_ms > 0.08:
            ux, uy = flow.unit
            along_ms = (px_vx * ux + px_vy * uy) / self.cfg.pixels_per_meter
            drift = min(1.0, max(0.0, along_ms / flow_ms))

        misses = sum(1 for o in self.history if o is None)
        submersion = (misses / len(self.history)) if self.history else 0.0

        exposure = min(1.0, self.frames_seen / (self.cfg.exposure_s * self.fps))
        flow_term = min(1.0, flow_ms / 2.5)
        flow_exposure = flow_term * (0.3 + 0.7 * exposure)

        return {
            "drift_ratio": drift,
            "struggle": min(1.0, self._struggle),
            "submersion": submersion,
            "flow_exposure": flow_exposure,
            "exposure": exposure,
            "speed_ms": speed_ms,
            "flow_ms": flow_ms,
        }
