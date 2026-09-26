"""Water-current estimation from live frames (Farnebäck optical flow).

Flood Guard needs to know how fast the water moves *where the person is*.
The estimator computes dense optical flow on a downscaled frame, ignores the
pixels inside detected-person boxes (people are not water), then reports:

  * a global FlowResult — median current over the whole water body, and
  * per-person local samples via ``local_flow(cx, cy)`` — the median current
    in a disc around the person, which is what the drift-ratio feature uses.

It works on real footage (foam, debris and surface texture carry the flow)
and on the synthetic flood simulator (advected speckle, streaks and debris).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


@dataclass
class FlowResult:
    """One current measurement (m/s in frame coordinates)."""

    vx_ms: float = 0.0
    vy_ms: float = 0.0
    speed_ms: float = 0.0
    speed_px_s: float = 0.0
    coherence: float = 0.0             # directional agreement, [0, 1]
    ready: bool = False

    @property
    def unit(self) -> Tuple[float, float]:
        """Unit flow direction (0, 0) when speed is zero."""
        s = self.speed_ms
        if s <= 1e-6:
            return (0.0, 0.0)
        return (self.vx_ms / s, self.vy_ms / s)


class FloodFlowEstimator:
    """Dense optical flow -> water current, person regions excluded."""

    def __init__(self, fps: float = 25.0, pixels_per_meter: float = 50.0,
                 downscale: int = 4, min_speed: float = 0.25,
                 min_support: int = 40) -> None:
        self.fps = max(1.0, fps)
        self.ppm = max(1.0, pixels_per_meter)
        self.downscale = max(1, downscale)
        self.min_speed = min_speed           # px/frame noise floor (downscaled)
        self.min_support = min_support
        self._prev_gray: Optional[np.ndarray] = None
        self._field: Optional[np.ndarray] = None   # HxWx2 downscaled flow field
        self._mask: Optional[np.ndarray] = None    # usable (non-person) pixels
        self._ema: Optional[Tuple[float, float]] = None  # smoothed current (m/s)

    # ------------------------------------------------------------------ #

    def estimate(self, frame_bgr: np.ndarray,
                 exclude_boxes: Sequence[Box] = ()) -> FlowResult:
        """Compute the global current for this frame (call once per frame)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        sw, sh = max(8, w // self.downscale), max(8, h // self.downscale)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mask = np.ones((sh, sw), bool)
        for b in exclude_boxes:
            x1, y1, x2, y2 = b
            # wide margin: Farneback's smoothness term smears person motion
            # into the surrounding water vectors for tens of pixels
            gx1 = max(0, int(x1 / self.downscale) - 6)
            gy1 = max(0, int(y1 / self.downscale) - 6)
            gx2 = min(sw, int(x2 / self.downscale) + 6)
            gy2 = min(sh, int(y2 / self.downscale) + 6)
            mask[gy1:gy2, gx1:gx2] = False

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray, self._mask = gray, mask
            return FlowResult()

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self._prev_gray, self._mask = gray, mask
        self._field = flow
        return self._stats(flow, mask)

    def local_flow(self, cx: float, cy: float,
                   radius_px: float = 140.0) -> FlowResult:
        """Current in a disc around a pixel position (uses the last field)."""
        if self._field is None or self._mask is None:
            return FlowResult()
        d = self.downscale
        h, w = self._mask.shape
        gx, gy, gr = cx / d, cy / d, max(3.0, radius_px / d)
        yy, xx = np.mgrid[0:h, 0:w]
        disc = (xx - gx) ** 2 + (yy - gy) ** 2 <= gr * gr
        return self._stats(self._field, self._mask & disc)

    # ------------------------------------------------------------------ #

    def _stats(self, flow: np.ndarray, mask: np.ndarray) -> FlowResult:
        vx, vy = flow[..., 0], flow[..., 1]
        mag2 = vx * vx + vy * vy
        sel = mask & (mag2 > self.min_speed ** 2)
        if int(sel.sum()) < self.min_support:
            return FlowResult()

        mvx = float(np.median(vx[sel]))
        mvy = float(np.median(vy[sel]))
        px_s = math.hypot(mvx, mvy) * self.downscale * self.fps
        speed_ms = px_s / self.ppm

        # coherence: share of selected vectors within 30 deg of the median
        ang = np.arctan2(vy[sel], vx[sel])
        dom = math.atan2(mvy, mvx)
        diff = np.abs((ang - dom + math.pi) % (2 * math.pi) - math.pi)
        coherence = float(np.mean(diff < math.radians(30.0)))

        k = self.downscale * self.fps / self.ppm
        vx_ms, vy_ms = mvx * k, mvy * k
        # Temporal EMA: currents change slowly, and jitter would reset the
        # sustained-count gates (swept/stranded) downstream every few frames.
        if self._ema is not None:
            a = 0.20
            vx_ms = a * vx_ms + (1.0 - a) * self._ema[0]
            vy_ms = a * vy_ms + (1.0 - a) * self._ema[1]
        self._ema = (vx_ms, vy_ms)
        px_s = math.hypot(vx_ms, vy_ms) / k if k > 0 else 0.0
        return FlowResult(vx_ms=vx_ms, vy_ms=vy_ms,
                          speed_ms=math.hypot(vx_ms, vy_ms),
                          speed_px_s=px_s, coherence=coherence, ready=True)
