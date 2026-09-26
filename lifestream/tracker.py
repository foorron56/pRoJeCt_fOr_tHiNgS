"""Lightweight centroid tracker keeping stable swimmer IDs across frames.

Chosen over heavier MOT stacks on purpose: pool scenes have few, slow targets
and frequent splash occlusions. Centroid + short missing-memory handles both
while staying dependency-free and real-time on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import TrackerConfig, DEFAULT_TRACKER

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]
    bbox: Box
    age: int = 0
    missing: int = 0
    extractor: object = None          # SwimFeatureExtractor, set by pipeline
    history: List = field(default_factory=list)


class CentroidTracker:
    """Assigns persistent IDs to swimmer detections frame to frame."""

    def __init__(self, config: TrackerConfig = DEFAULT_TRACKER) -> None:
        self.cfg = config
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: Sequence[Box]) -> List[Track]:
        """Match detections to existing tracks; spawn/prune as needed."""
        self._age_tracks()

        centroids = [_centroid(b) for b in detections]
        unmatched_det = set(range(len(detections)))

        if self.tracks:
            track_ids = list(self.tracks.keys())
            dist = np.zeros((len(track_ids), len(centroids)), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                for di, c in enumerate(centroids):
                    dist[ti, di] = np.hypot(
                        c[0] - self.tracks[tid].centroid[0],
                        c[1] - self.tracks[tid].centroid[1],
                    )
            # greedy nearest-neighbour matching (few swimmers => fine)
            used_tracks = set()
            while unmatched_det:
                candidates = [
                    (dist[ti, di], ti, di)
                    for ti in range(len(track_ids)) if ti not in used_tracks
                    for di in unmatched_det
                ]
                if not candidates:
                    break
                d, ti, di = min(candidates)
                if d > self.cfg.max_distance:
                    break
                tid = track_ids[ti]
                self._update_track(self.tracks[tid], detections[di], centroids[di])
                used_tracks.add(ti)
                unmatched_det.discard(di)

        for di in unmatched_det:
            self._spawn(detections[di], centroids[di])

        for tid in [t for t, tr in self.tracks.items() if tr.missing > self.cfg.max_missing]:
            del self.tracks[tid]

        return list(self.tracks.values())

    # ------------------------------------------------------------------ #

    def _age_tracks(self) -> None:
        for tr in self.tracks.values():
            tr.missing += 1

    def _update_track(self, tr: Track, bbox: Box, centroid) -> None:
        tr.bbox, tr.centroid, tr.missing = bbox, centroid, 0
        tr.age += 1

    def _spawn(self, bbox: Box, centroid) -> None:
        self.tracks[self._next_id] = Track(
            track_id=self._next_id, centroid=centroid, bbox=bbox, age=1
        )
        self._next_id += 1


def _centroid(b: Box) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
