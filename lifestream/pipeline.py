"""LifeStream pipeline: sources -> detection/tracking -> features -> risk -> alerts.

Sources:
  * Webcam / RTSP / video file via OpenCV   (real deployment)
  * PoolSimulator                            (demo without hardware)

The pipeline is source-agnostic: every source yields SwimObservations through
the same ``process()`` contract, so the tracking/feature/risk stack is
identical in the demo and in production.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alerts import AlertManager
from .config import (DEFAULT_ALERT, DEFAULT_RISK, DEFAULT_TRACKER,
                     AlertConfig, RiskConfig, TrackerConfig)
from .features import SwimFeatureExtractor, SwimObservation
from .risk_engine import DrowningRiskEngine, RiskState
from .tracker import CentroidTracker

Box = Tuple[float, float, float, float]


@dataclass
class PipelineFrame:
    """Everything the UI needs for one processed frame."""

    frame_idx: int
    observations: List[SwimObservation]
    states: Dict[int, RiskState] = field(default_factory=dict)
    boxes: Dict[int, Box] = field(default_factory=dict)   # swimmer_id -> bbox
    latency_ms: float = 0.0


class _TrackExtractors:
    """Hands each track its own sliding-window feature extractor."""

    def __init__(self) -> None:
        self._map: Dict[int, SwimFeatureExtractor] = {}

    def get(self, tid: int) -> SwimFeatureExtractor:
        if tid not in self._map:
            self._map[tid] = SwimFeatureExtractor()
        return self._map[tid]


class LifeStreamPipeline:
    """Wires tracker + features + risk engine + alerts together."""

    def __init__(self, fps: float = 25.0,
                 tracker_cfg: TrackerConfig = DEFAULT_TRACKER,
                 risk_cfg: RiskConfig = DEFAULT_RISK,
                 alert_cfg: AlertConfig = DEFAULT_ALERT,
                 alert_manager: Optional[AlertManager] = None,
                 scorer: Optional["LearnedScorer"] = None) -> None:
        self.fps = fps
        self.tracker = CentroidTracker(tracker_cfg)
        self.engine = DrowningRiskEngine(fps=fps, config=risk_cfg,
                                         scorer=scorer)
        self.alerts = alert_manager or AlertManager(alert_cfg)
        self._extractors = _TrackExtractors()
        self._frame_idx = 0

    def process(self, observations: List[SwimObservation],
                frame: Optional[np.ndarray] = None) -> PipelineFrame:
        t0 = time.perf_counter()
        self._frame_idx += 1

        tracks = self.tracker.update([o.bbox for o in observations])
        matched = _assign(observations, [t.bbox for t in tracks])

        pf = PipelineFrame(frame_idx=self._frame_idx, observations=observations)
        centers = [t.centroid for t in tracks]

        for i, track in enumerate(tracks):
            obs = matched[i]
            extractor = self._extractors.get(track.track_id)
            if obs is not None:
                feats = extractor.update(obs)
                has_pose = obs.visible
            else:
                feats = extractor.features()
                has_pose = False

            state = self.engine.update(
                track.track_id, feats,
                is_isolated=_isolated(i, centers), has_pose=has_pose,
            )
            pf.states[track.track_id] = state
            pf.boxes[track.track_id] = track.bbox
            # AlertManager dedupes: only state *transitions* are emitted.
            self.alerts.notify(track.track_id, state, pf.frame_idx)

        pf.latency_ms = (time.perf_counter() - t0) * 1000.0
        return pf


def _assign(observations: List[SwimObservation],
            boxes: List[Box]) -> List[Optional[SwimObservation]]:
    """Greedy bbox-overlap matching between observations and track boxes."""
    used_obs = set()
    out: List[Optional[SwimObservation]] = []
    for box in boxes:
        best_j, best_area = None, 0.0
        for j, obs in enumerate(observations):
            if j in used_obs:
                continue
            area = _overlap_area(box, obs.bbox)
            if area > best_area:
                best_j, best_area = j, area
        if best_j is None:
            out.append(None)
        else:
            used_obs.add(best_j)
            out.append(observations[best_j])
    return out


def _overlap_area(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _isolated(index: int, centers: List[Tuple[float, float]]) -> bool:
    """No other swimmer within ~1.5 bbox widths => alone in the water."""
    if len(centers) < 2 or index >= len(centers):
        return True
    cx, cy = centers[index]
    for j, (ox, oy) in enumerate(centers):
        if j != index and np.hypot(ox - cx, oy - cy) < 170:
            return False
    return True
