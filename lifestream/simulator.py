"""Synthetic pool scene generator — the LifeStream live demo, no pool needed.

Renders swimmers procedurally in an overhead-angled pool view and drives them
with behavior scripts so every risk phase can be demonstrated deterministically
on any laptop:

  * laps      — horizontal freestyle swimmer crossing the pool (safe)
  * distress  — vertical posture, frantic arm thrashing, head bobbing under
  * still     — the classic Instinctive Drowning Response arc: treads water,
                goes quiet, sinks slowly, head slips below the surface
  * playing   — safe splashing in the shallow end (must never alert)

The simulator emits SwimObservations with bbox-local fractional keypoints —
the exact convention of the MediaPipe detector — so the whole pipeline is
exercised identically in demo and production.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import KEYPOINT_NAMES, SwimObservation

Box = Tuple[float, float, float, float]
Pt = Tuple[float, float]

# Body model in canonical units (fractions of frame height), head toward -w.
BODY = {
    "nose": (0.00, -0.46),
    "left_shoulder": (-0.07, -0.30), "right_shoulder": (0.07, -0.30),
    "left_elbow": (-0.15, -0.14), "right_elbow": (0.15, -0.14),
    "left_wrist": (-0.19, -0.28), "right_wrist": (0.19, -0.28),
    "left_hip": (-0.05, 0.02), "right_hip": (0.05, 0.02),
    "left_knee": (-0.06, 0.16), "right_knee": (0.06, 0.16),
    "left_ankle": (-0.07, 0.30), "right_ankle": (0.07, 0.30),
}
PAD = 0.06          # bbox padding (frame-height fraction)

# Behavior presets: posture v (0 horizontal..1 vertical), arm amplitude,
# arm frequency, phase (0 = both arms together / pi = alternating), kick amp.
PRESETS = {
    "laps": dict(v=0.05, arm=0.20, freq=7.0, phase=math.pi, kick=0.05),
    "distress": dict(v=0.85, arm=0.42, freq=12.0, phase=0.0, kick=0.14),
    "playing": dict(v=0.25, arm=0.12, freq=6.0, phase=0.0, kick=0.04),
}


class PoolSimulator:
    """Procedural swimmers with scripted behaviors; also renders preview frames."""

    def __init__(self, width: int = 960, height: int = 540) -> None:
        self.w, self.h = float(width), float(height)
        self.t = 0.0
        self.frame_idx = 0
        self.actors = [
            {"name": "victim", "behavior": "laps", "x": 0.62, "y": 0.52,
             "bt": 0.0, "dir": 1},
            {"name": "swimmer", "behavior": "laps", "x": 0.15, "y": 0.35,
             "bt": 0.0, "dir": 1},
        ]

    def set_behavior(self, actor_name: str, behavior: str) -> None:
        """Switch one actor's behavior script and restart its stage timer."""
        for actor in self.actors:
            if actor["name"] == actor_name:
                actor["behavior"] = behavior
                actor["bt"] = 0.0

    # ------------------------------------------------------------------ #

    def step(self, dt: float = 1.0 / 25.0) -> List[SwimObservation]:
        self.t += dt
        self.frame_idx += 1
        return [self._observation(a, dt) for a in self.actors]

    def _observation(self, actor: Dict, dt: float) -> SwimObservation:
        actor["bt"] += dt
        pose = self._pose_for(actor)
        pts = self._body_points(actor, pose)

        # bbox from extremities + padding
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        x1, x2 = min(xs) - PAD, max(xs) + PAD
        y1, y2 = min(ys) - PAD, max(ys) + PAD
        bbox = (x1 * self.w, y1 * self.h, x2 * self.w, y2 * self.h)
        bw = max(1.0, (x2 - x1) * self.w)
        bh = max(1.0, (y2 - y1) * self.h)

        kps: Dict[str, Optional[Pt]] = {n: None for n in KEYPOINT_NAMES}
        for name, (fx, fy) in pts.items():
            px, py = fx * self.w, fy * self.h
            kps[name] = ((px - bbox[0]) / bw, (py - bbox[1]) / bh)

        # tracker-provided center drift (frame fractions per frame)
        cdx = (pose["x_delta"] if "x_delta" in pose else 0.0)
        cdy = (pose["sink_dy"] if "sink_dy" in pose else 0.0)
        return SwimObservation(frame_idx=self.frame_idx, bbox=bbox,
                               keypoints=kps, visible=True,
                               center_dx=cdx, center_dy=cdy)

    def _pose_for(self, actor: Dict) -> Dict:
        t, bt = self.t, actor["bt"]
        behavior = actor["behavior"]

        if behavior == "laps":
            dx = 0.045 * actor["dir"] * (1.0 / 25.0)
            actor["x"] += dx
            if not 0.08 <= actor["x"] <= 0.92:
                actor["dir"] *= -1
                actor["x"] = float(np.clip(actor["x"], 0.08, 0.92))
                dx = 0.0
            return {**PRESETS["laps"], "sink": 0.0, "dip": 0.0, "bob": 0.0,
                    "x_delta": dx, "sink_dy": 0.0}

        if behavior == "custom":
            # Parametric pose driven by actor["pose_params"] — used by the
            # training pipeline to roll randomized episodes through the REAL
            # body geometry (rotation, bbox, limbs) instead of a re-derived
            # upright-only model. Safe and dangerous modes differ by arm
            # amplitude, posture v, head-dip depth and optional sinking arc.
            p = actor.get("pose_params") or {}
            v = p.get("v", 0.3)
            arm = p.get("arm", 0.2)
            kick = p.get("kick", 0.06)
            onset = p.get("sink_onset")
            if onset is not None and bt >= onset:
                dur = p.get("sink_dur", 12.0)
                s = min(1.0, (bt - onset) / dur)
                ease = s * s * (3.0 - 2.0 * s)             # smoothstep
                v = v + (p.get("sink_v", 0.90) - v) * ease
                arm = arm * (1.0 - ease) + 0.015
                kick = kick * (1.0 - ease) + 0.004
                dip = p.get("sink_dip", 0.34) * ease
                sink_dy = 0.30 * (6.0 * s * (1.0 - s) / dur) / 25.0
                sink = ease
            else:
                dip = p.get("dip", 0.0) * max(
                    0.0, math.sin(t * p.get("dip_freq", 2.2)))
                sink, sink_dy = 0.0, 0.0
            drift = p.get("x_speed", 0.0) * (1.0 / 25.0)
            actor["x"] += drift
            if not 0.08 <= actor["x"] <= 0.92:             # bounce at walls
                p["x_speed"] = -p.get("x_speed", 0.0)
                actor["x"] = float(np.clip(actor["x"], 0.08, 0.92))
                drift = 0.0
            return dict(v=v, arm=arm, freq=p.get("freq", 6.0),
                        phase=p.get("phase", math.pi), kick=kick,
                        sink=sink, dip=dip, bob=0.006 * math.sin(t * 2.5),
                        x_delta=drift, sink_dy=sink_dy)

        if behavior == "distress":
            return {**PRESETS["distress"], "sink": 0.0, "x_delta": 0.0,
                    "sink_dy": 0.0,
                    "dip": 0.22 * max(0.0, math.sin(t * 2.2)),   # head bobs under
                    "bob": 0.008 * math.sin(t * 3.1)}

        if behavior == "still":
            # Stage 1 (0-6 s): treading water, active. Stage 2: goes quiet,
            # sinks, head slips below the surface.
            if bt < 6.0:
                # treading water: vigorous, head well above surface
                return dict(v=0.28, arm=0.28, freq=6.0, phase=math.pi,
                            kick=0.09, sink=0.0, x_delta=0.0, sink_dy=0.0,
                            dip=0.02 * max(0.0, math.sin(t * 1.8)),
                            bob=0.0)
            sink = min(1.0, (bt - 6.0) / 12.0)
            ease = sink * sink * (3 - 2 * sink)                # smoothstep
            # continuous downward drift while sinking (frame fraction/frame)
            sink_dy = 0.30 * (6.0 * sink * (1.0 - sink) / 12.0) / 25.0
            return dict(v=0.28 + 0.62 * ease, arm=0.28 * (1.0 - ease) + 0.015,
                        freq=6.0, phase=math.pi, kick=0.09 * (1.0 - ease),
                        sink=ease, x_delta=0.0, sink_dy=sink_dy,
                        dip=0.36 * ease, bob=0.0)

        return {**PRESETS["playing"], "sink": 0.0, "x_delta": 0.0,
                "sink_dy": 0.0, "dip": 0.0,
                "bob": 0.006 * math.sin(t * 2.5)}

    def _body_points(self, actor: Dict, pose: Dict) -> Dict[str, Pt]:
        """Orient the canonical body by posture: v=1 upright, v=0 horizontal
        (head pointing along +x, the swimming direction)."""
        phi = pose["v"] * (math.pi / 2.0)       # elevation from horizontal
        cph, sph = math.cos(phi), math.sin(phi)
        cx = actor["x"]
        cy = actor["y"] + pose["bob"] + pose["sink"] * 0.30    # sinks downward
        t = self.t

        pts: Dict[str, Pt] = {}
        for name, (u, w) in BODY.items():
            if name == "nose":
                w = w + pose["dip"]
            elif name in ("left_wrist", "right_wrist"):
                side = -1.0 if name.startswith("left") else 1.0
                swing = math.sin(t * pose["freq"] +
                                 (pose["phase"] if side < 0 else 0.0))
                w = w + swing * pose["arm"]
                u = u + 0.04 * swing * side
            elif name in ("left_elbow", "right_elbow"):
                side = -1.0 if name.startswith("left") else 1.0
                swing = math.sin(t * pose["freq"] +
                                 (pose["phase"] if side < 0 else 0.0))
                w = w + 0.4 * swing * pose["arm"]
            elif name in ("left_ankle", "right_ankle"):
                side = -1.0 if name.startswith("left") else 1.0
                kick = math.sin(t * pose["freq"] * 1.3 +
                                (0.5 if side < 0 else 0.0))
                u = u + kick * pose["kick"] * side
                w = w + 0.3 * kick * pose["kick"]

            # canonical (u = perpendicular, w = along body, head = -w);
            # head direction h = (cos phi, -sin phi)
            fx = cx - w * cph + u * sph
            fy = cy + w * sph + u * cph
            pts[name] = (fx, fy)
        return pts

    # ------------------------------------------------------------------ #
    # rendering

    def render(self, obs: List[SwimObservation]) -> "np.ndarray":
        import cv2

        W, H = int(self.w), int(self.h)
        frame = np.full((H, W, 3), (150, 130, 100), np.uint8)
        for gx in range(0, W, 60):
            cv2.line(frame, (gx, 0), (gx, H), (165, 145, 115), 1)
        for gy in range(0, H, 60):
            cv2.line(frame, (0, gy), (W, gy), (165, 145, 115), 1)
        cv2.rectangle(frame, (50, 35), (W - 50, H - 35), (160, 115, 60), -1)
        # gentle water shimmer
        for i in range(6):
            yy = 60 + i * 80 + int(6 * math.sin(self.t * 1.5 + i))
            cv2.line(frame, (70, yy), (W - 70, yy), (185, 145, 90), 1)

        skin = (150, 185, 240)
        for o in obs:
            x1, y1, x2, y2 = o.bbox
            bw, bh = x2 - x1, y2 - y1

            def px(name: str) -> Optional[Tuple[int, int]]:
                p = o.keypoints.get(name)
                if p is None:
                    return None
                return (int(x1 + p[0] * bw), int(y1 + p[1] * bh))

            bones = [
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                ("left_hip", "right_hip"),
                ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ]
            for a, b in bones:
                pa, pb = px(a), px(b)
                if pa and pb:
                    cv2.line(frame, pa, pb, skin, 5)
            nose = px("nose")
            if nose:
                cv2.circle(frame, nose, max(6, int(0.035 * bh)), skin, -1)
        return frame
