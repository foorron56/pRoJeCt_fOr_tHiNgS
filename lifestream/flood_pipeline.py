"""Flood Guard pipeline: frames -> detection/tracking -> flow -> risk -> alerts.

Source-agnostic like the pool pipeline: the simulator yields SwimObservations
under the same contract as the MediaPipe detector, and the current is
measured from the actual frames (rendered or real) with optical flow, with
detected-person regions excluded. A generous ``max_missing`` keeps identities
alive through multi-second submersions in murky water.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .alerts import AlertManager
from .config import (DEFAULT_FLOOD, DEFAULT_FLOOD_RISK, AlertConfig,
                     FloodConfig, FloodRiskConfig, TrackerConfig)
from .features import SwimObservation
from .flow import FloodFlowEstimator, FlowResult
from .flood_engine import FloodRiskEngine
from .flood_features import FloodFeatureExtractor
from .risk_engine import RiskState
from .tracker import CentroidTracker, Track

Box = Tuple[float, float, float, float]


@dataclass
class FloodPipelineFrame:
    """Everything the UI needs for one processed frame."""

    frame_idx: int
    observations: List[Optional[SwimObservation]]
    states: Dict[int, RiskState] = field(default_factory=dict)
    boxes: Dict[int, Box] = field(default_factory=dict)
    flow: Optional[FlowResult] = None
    latency_ms: float = 0.0


class _TrackExtractors:
    """Hands each track its own sliding-window feature extractor."""

    def __init__(self) -> None:
        self._map: Dict[int, FloodFeatureExtractor] = {}

    def get(self, tid: int, fps: float, cfg: FloodConfig) -> FloodFeatureExtractor:
        if tid not in self._map:
            self._map[tid] = FloodFeatureExtractor(fps=fps, config=cfg)
        return self._map[tid]


class FloodPipeline:
    """Wires tracker + flow + flood features + risk engine + alerts."""

    def __init__(self, fps: float = 25.0,
                 tracker_cfg: Optional[TrackerConfig] = None,
                 risk_cfg: FloodRiskConfig = DEFAULT_FLOOD_RISK,
                 flood_cfg: FloodConfig = DEFAULT_FLOOD,
                 alert_manager: Optional[AlertManager] = None,
                 scorer: Optional["LearnedScorer"] = None) -> None:
        self.fps = fps
        self.flood_cfg = flood_cfg
        # people stay "missing" longer in murky water; keep their identity
        self.tracker = CentroidTracker(
            tracker_cfg or TrackerConfig(max_distance=110.0, max_missing=45))
        self.engine = FloodRiskEngine(fps=fps, config=risk_cfg, scorer=scorer)
        self.alerts = alert_manager or AlertManager(
            AlertConfig(event_label="person in distress in floodwater"))
        self.flow_est = FloodFlowEstimator(
            fps=fps, pixels_per_meter=flood_cfg.pixels_per_meter,
            downscale=flood_cfg.flow_downscale,
            min_speed=flood_cfg.flow_min_speed,
            min_support=flood_cfg.flow_min_support)
        self._extractors = _TrackExtractors()
        self._frame_idx = 0

    def process(self, observations: List[Optional[SwimObservation]],
                frame: Optional[np.ndarray] = None) -> FloodPipelineFrame:
        t0 = time.perf_counter()
        self._frame_idx += 1

        vis = [o for o in observations if o is not None]
        tracks = self.tracker.update([o.bbox for o in vis])
        flow = (self.flow_est.estimate(frame, [t.bbox for t in tracks])
                if frame is not None else None)
        matched = _assign(vis, [t.bbox for t in tracks])

        fpf = FloodPipelineFrame(frame_idx=self._frame_idx,
                                 observations=list(observations), flow=flow)
        for i, track in enumerate(tracks):
            local = flow
            if flow is not None and flow.ready:
                near = self.flow_est.local_flow(track.centroid[0],
                                                track.centroid[1])
                if near.ready:
                    local = near          # current at the person's location
            feats = self._extractors.get(track.track_id, self.fps,
                                         self.flood_cfg).update(matched[i], local)
            state = self.engine.update(track.track_id, feats)
            fpf.states[track.track_id] = state
            fpf.boxes[track.track_id] = track.bbox
            # AlertManager dedupes: only state *transitions* are emitted.
            self.alerts.notify(track.track_id, state, fpf.frame_idx,
                               source="flood-cam")

        fpf.latency_ms = (time.perf_counter() - t0) * 1000.0
        return fpf


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
