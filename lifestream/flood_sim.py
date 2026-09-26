"""Synthetic flood scene generator — the LifeStream Flood Guard live demo.

Renders a flooded river channel seen from a bank-mounted camera: murky fast
water with advected speckle, streaks and drifting debris, a concrete
embankment with a railing, a bridge pillar standing in the channel and a
mid-channel rock. One person is driven by a scripted behavior so every
flood-risk phase can be demonstrated deterministically on any laptop:

  drift     — slips at the bank (0-2.5 s), is swept downstream by the
              current, then gets pinned against the bridge pillar
  clinging  — pinned against the pillar from the start, struggling, head
              repeatedly dipping under
  wading    — crosses the shallow near-bank water in full control, then
              walks the flooded road (must never alert)
  stranded  — stuck on the mid-channel rock while the current rages past

Submersion is honest: while the person's head is under, the simulator emits
NO observation at all — exactly what murky water does to a real detector —
so the pipeline's murky-water path is exercised, not bypassed.

Observations use the same SwimObservation convention as the MediaPipe
detector (bbox-local fractional keypoints).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import KEYPOINT_NAMES, SwimObservation

Box = Tuple[float, float, float, float]
Pt = Tuple[float, float]

SCENARIOS = ("drift", "clinging", "wading", "stranded")

# Side-view stick figure: pixel offsets from the person's water-center point.
FIGURE = {
    "nose": (0.0, -44.0),
    "left_shoulder": (-7.0, -32.0), "right_shoulder": (7.0, -32.0),
    "left_elbow": (-12.0, -18.0), "right_elbow": (12.0, -18.0),
    "left_wrist": (-15.0, -6.0), "right_wrist": (15.0, -6.0),
    "left_hip": (-5.0, -4.0), "right_hip": (5.0, -4.0),
    "left_knee": (-6.0, 10.0), "right_knee": (6.0, 10.0),
    "left_ankle": (-7.0, 24.0), "right_ankle": (7.0, 24.0),
}
PAD_PX = 14.0

BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
)


class FloodSimulator:
    """Procedural flood channel with one scripted person; renders frames."""

    def __init__(self, width: int = 960, height: int = 540,
                 flow_ms: float = 2.6, pixels_per_meter: float = 50.0,
                 scenario: str = "drift") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown flood scenario {scenario!r}; choose one of {SCENARIOS}")
        self.w, self.h = float(width), float(height)
        self.ppm = pixels_per_meter
        self.flow_ms = flow_ms
        self.y_bank = 0.16 * self.h
        self.t = 0.0
        self.frame_idx = 0

        # bridge pillar standing in the channel + mid-channel rock
        self.pillar = (0.66 * self.w, self.y_bank, 0.70 * self.w, 0.62 * self.h)
        self.rock = (0.45 * self.w, 0.70 * self.h)

        # flow visualization: advected speckle + streaks + debris
        rng = np.random.default_rng(7)
        n = 340
        self.speckle = np.column_stack([
            rng.uniform(0, self.w, n),
            rng.uniform(self.y_bank + 6, self.h - 6, n),
        ])
        n = 110
        self.streaks = np.column_stack([
            rng.uniform(0, self.w, n),
            rng.uniform(self.y_bank + 6, self.h - 6, n),
        ])
        self.debris = [
            {"x": rng.uniform(0, self.w),
             "y": rng.uniform(self.y_bank + 14, self.h - 14),
             "w": float(rng.integers(10, 24)), "h": float(rng.integers(4, 7))}
            for _ in range(8)
        ]

        self.actor = {"scenario": scenario, "x": 0.0, "y": 0.0, "bt": 0.0,
                      "pinned": False}
        self._place_actor()

    def _place_actor(self) -> None:
        a, s = self.actor, self.actor["scenario"]
        a["pinned"] = False
        if s == "drift":
            a["x"], a["y"] = 0.16 * self.w, 0.34 * self.h
        elif s == "clinging":
            a["x"], a["y"] = self.pillar[0] - 11.0, 0.40 * self.h
        elif s == "wading":
            a["x"], a["y"] = 0.22 * self.w, self.y_bank + 0.10 * (self.h - self.y_bank)
        else:  # stranded
            a["x"], a["y"] = self.rock[0], self.rock[1] - 26.0

    def set_scenario(self, scenario: str) -> None:
        self.actor["scenario"] = scenario
        self.actor["bt"] = 0.0
        self._place_actor()

    # ------------------------------------------------------------------ #

    def local_flow_ms(self, y: float) -> float:
        """Current is slower near the banks, fastest mid-channel."""
        frac = (y - self.y_bank) / max(1.0, (self.h - self.y_bank))
        frac = min(1.0, max(0.0, frac))
        return self.flow_ms * (0.30 + 0.70 * frac)

    def step(self, dt: float = 1.0 / 25.0) -> List[Optional[SwimObservation]]:
        self.t += dt
        self.frame_idx += 1
        self._advect_water(dt)
        self._advance_actor(dt)
        return [self._observation()]

    def _advect_water(self, dt: float) -> None:
        for p in self.speckle:
            self._advect_point(p, dt, 6.0)
        for p in self.streaks:
            self._advect_point(p, dt, 6.0)
        for d in self.debris:
            d["x"] += self.local_flow_ms(d["y"]) * self.ppm * dt
            if d["x"] > self.w + 30:
                d["x"] = -30
                d["y"] = float(np.clip(d["y"] + np.random.uniform(-40, 40),
                                       self.y_bank + 14, self.h - 14))

    def _advect_point(self, p: np.ndarray, dt: float, pad: float) -> None:
        p[0] += self.local_flow_ms(p[1]) * self.ppm * dt
        if p[0] > self.w + pad:
            p[0] = -pad
            p[1] = float(np.clip(p[1] + np.random.uniform(-30, 30),
                                 self.y_bank + pad, self.h - pad))

    def _advance_actor(self, dt: float) -> None:
        a = self.actor
        a["bt"] += dt
        t, bt, s = self.t, a["bt"], a["scenario"]

        if s == "wading":
            local = self.local_flow_ms(a["y"]) * self.ppm
            if a["y"] < 0.62 * self.h:
                vx, vy = 0.30 * local, 20.0       # crossing, partly with flow
            else:
                vx, vy = 30.0, 0.0                # walking the flooded road
            a["x"] = float(np.clip(a["x"] + vx * dt, 0.06 * self.w, 0.94 * self.w))
            a["y"] = float(np.clip(a["y"] + vy * dt, self.y_bank + 24, 0.86 * self.h))

        elif s == "drift":
            if bt < 2.5:
                # losing footing at the bank edge
                a["x"] += 16.0 * math.sin(t * 9.0) * dt
                a["y"] += 10.0 * math.cos(t * 7.0) * dt
            else:
                local = self.local_flow_ms(a["y"]) * self.ppm
                if not a["pinned"]:
                    a["x"] += 0.95 * local * dt
                    a["y"] += 6.0 * dt
                    if a["x"] >= self.pillar[0] - 12.0:
                        a["pinned"] = True
                else:
                    a["y"] += 4.0 * math.sin(t * 3.0) * dt

        elif s == "clinging":
            a["x"] = self.pillar[0] - 11.0 + 0.8 * math.sin(t * 5.0)
            a["y"] = 0.40 * self.h + 2.0 * math.sin(t * 1.3)

        else:  # stranded
            a["x"] = self.rock[0] + 2.0 * math.sin(t * 0.7)
            a["y"] = self.rock[1] - 26.0 + 1.0 * math.sin(t * 1.1)

    # ------------------------------------------------------------------ #

    def _observation(self) -> Optional[SwimObservation]:
        pts, submerged = self._figure()
        if submerged:
            return None                            # lost under murky water
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        x1, y1 = min(xs) - PAD_PX, min(ys) - PAD_PX
        x2, y2 = max(xs) + PAD_PX, max(ys) + PAD_PX
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

        kps: Dict[str, Optional[Pt]] = {n: None for n in KEYPOINT_NAMES}
        for name, (px, py) in pts.items():
            kps[name] = ((px - x1) / bw, (py - y1) / bh)
        return SwimObservation(frame_idx=self.frame_idx, bbox=(x1, y1, x2, y2),
                               keypoints=kps, visible=True)

    def _figure(self) -> Tuple[Dict[str, Pt], bool]:
        """Current body points (px) + submerged flag for the actor."""
        a = self.actor
        t, bt, s = self.t, a["bt"], a["scenario"]
        cx, cy = a["x"], a["y"]
        pts = {n: (cx + u, cy + v) for n, (u, v) in FIGURE.items()}
        submerged = False
        flail = False

        if s == "wading":
            bob = 1.5 * math.sin(t * 4.0)
            pts = {n: (cx + u, cy + v + bob) for n, (u, v) in FIGURE.items()}
            step = math.sin(t * 5.0)
            pts["left_ankle"] = (cx - 7.0 - 4.0 * step, cy + 24.0 + 3.0 * max(0.0, step))
            pts["right_ankle"] = (cx + 7.0 + 4.0 * step, cy + 24.0 + 3.0 * max(0.0, -step))
            pts["left_knee"] = (cx - 6.0 - 2.0 * step, cy + 10.0)
            pts["right_knee"] = (cx + 6.0 + 2.0 * step, cy + 10.0)
            pts["left_wrist"] = (cx - 15.0 + 3.0 * step, cy - 6.0)
            pts["right_wrist"] = (cx + 15.0 - 3.0 * step, cy - 6.0)

        elif s == "drift":
            if bt < 2.5:
                flail = True
                pts = {n: (cx + u, cy + v + 2.0 * math.sin(t * 7.0))
                       for n, (u, v) in FIGURE.items()}
            else:
                if a.get("pinned"):
                    period, length, cycle = 3.5, 1.3, bt
                else:
                    period, length, cycle = 4.5, 1.1, bt - 2.5
                submerged = (cycle % period) < length
                drag = 0.0 if a.get("pinned") else 10.0
                pts = {n: (cx + u + (drag if ("knee" in n or "ankle" in n
                                              or "hip" in n) else 0.0), cy + v)
                       for n, (u, v) in FIGURE.items()}
                flail = True

        elif s == "clinging":
            pull = math.sin(t * 9.0)
            # both arms up, gripping the pillar to the right
            pts["right_wrist"] = (cx + 16.0, cy - 38.0 + 3.0 * pull)
            pts["right_elbow"] = (cx + 12.0, cy - 22.0)
            pts["left_wrist"] = (cx + 13.0, cy - 46.0 - 3.0 * pull)
            pts["left_elbow"] = (cx + 8.0, cy - 26.0)
            pts["left_knee"] = (cx - 8.0 - 2.5 * pull, cy + 8.0)
            pts["right_knee"] = (cx + 4.0 - 2.5 * pull, cy + 10.0)
            pts["left_ankle"] = (cx - 10.0 - 3.5 * pull, cy + 20.0)
            pts["right_ankle"] = (cx + 2.0 - 3.5 * pull, cy + 22.0)
            # waves keep breaking over them: head goes under every ~3 s
            submerged = (bt % 3.0) < 1.2

        else:  # stranded
            if (t % 3.0) < 0.6:                    # periodic help wave
                pts["right_wrist"] = (cx + 17.0, cy - 50.0)
                pts["right_elbow"] = (cx + 11.0, cy - 30.0)

        if flail:
            sw = math.sin(t * 11.0)
            pts["left_wrist"] = (cx - 14.0 - 5.0 * sw, cy - 24.0 + 8.0 * sw)
            pts["right_wrist"] = (cx + 14.0 + 5.0 * sw, cy - 24.0 - 8.0 * sw)
            pts["left_elbow"] = (cx - 13.0 - 2.0 * sw, cy - 16.0)
            pts["right_elbow"] = (cx + 13.0 + 2.0 * sw, cy - 16.0)

        return pts, submerged

    # ------------------------------------------------------------------ #
    # rendering

    def render(self, obs: Optional[List[Optional[SwimObservation]]] = None
               ) -> "np.ndarray":
        import cv2

        W, H = int(self.w), int(self.h)
        yt = int(self.y_bank)
        frame = np.empty((H, W, 3), np.uint8)
        frame[:] = (58, 52, 44)                    # mud base
        for y in range(yt, H):                     # murky water gradient
            f = (y - yt) / max(1, H - yt)
            frame[y, :] = (int(52 + 34 * f), int(74 + 30 * f), int(58 + 22 * f))

        # embankment + railing
        cv2.rectangle(frame, (0, 0), (W, yt), (96, 96, 92), -1)
        cv2.line(frame, (0, yt - 1), (W, yt - 1), (60, 60, 58), 2)
        for x in range(20, W, 48):
            cv2.line(frame, (x, yt - 26), (x, yt - 4), (70, 70, 66), 2)
        cv2.line(frame, (0, yt - 26), (W, yt - 26), (80, 80, 76), 2)

        # advected water texture
        for p in self.speckle:
            x, y = int(p[0]), int(p[1])
            cv2.circle(frame, (x, y), 1, (84, 106, 86), -1)
        for p in self.streaks:
            x, y = int(p[0]), int(p[1])
            v = self.local_flow_ms(y) * self.ppm
            cv2.line(frame, (int(x - v * 0.20), y), (x, y),
                     (104, 128, 102), 2, cv2.LINE_AA)
        for d in self.debris:
            x, y = int(d["x"]), int(d["y"])
            cv2.rectangle(frame, (x, y), (x + int(d["w"]), y + int(d["h"])),
                          (38, 44, 52), -1)

        # bridge pillar + mid-channel rock
        px1, py1, px2, py2 = map(int, self.pillar)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (78, 82, 88), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (50, 54, 60), 2)
        rx, ry = int(self.rock[0]), int(self.rock[1])
        cv2.ellipse(frame, (rx, ry), (46, 15), 0, 0, 360, (88, 90, 94), -1)

        # the person
        pts, submerged = self._figure()
        color = (95, 120, 152) if submerged else (150, 190, 235)
        for a, b in BONES:
            pa, pb = pts[a], pts[b]
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, 3, cv2.LINE_AA)
        nose = pts["nose"]
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 5, color, -1, cv2.LINE_AA)
        if submerged:                              # ripple where they went under
            cv2.ellipse(frame, (int(pts["left_hip"][0]), int(self.actor["y"] + 6)),
                        (26, 7), 0, 0, 360, (120, 138, 120), 2, cv2.LINE_AA)
        return frame
