# -*- coding: utf-8 -*-
"""Profile real-footage clips (data/*.mp4) through the REAL LifeStream
pipeline: person detection -> tracking -> optical-flow current -> flood
features -> risk engine. Produces per-file statistics (JSON + table) so the
clips can be labeled honestly for real-data training.

Usage:
    python scripts/probe_dataset.py                 # all files
    python scripts/probe_dataset.py --start 0 --end 8   # chunk 1
    python scripts/probe_dataset.py --start 8 --end 16  # chunk 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lifestream.detector import SwimDetector  # noqa: E402
from lifestream.flood_engine import FloodRiskEngine  # noqa: E402
from lifestream.flood_features import FloodFeatureExtractor  # noqa: E402
from lifestream.flow import FloodFlowEstimator  # noqa: E402
from lifestream.tracker import CentroidTracker  # noqa: E402

SEG_LEN = 200          # consecutive frames per analyzed segment (flow needs pairs)
N_SEGMENTS = 2         # segments per video (start + middle)
MAX_BOXES = 3


def probe_file(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"file": os.path.basename(path), "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    det = SwimDetector(confidence=0.5)
    tracker = CentroidTracker()
    flow = FloodFlowEstimator(fps=fps)
    fex = FloodFeatureExtractor(fps=fps)
    engine = FloodRiskEngine(fps=fps)

    # segment frame ranges: beginning + middle
    segs = []
    for s in (0, max(0, total // 2)):
        segs.append((s, min(total, s + SEG_LEN)))

    st = {
        "frames": 0, "pose_ready_frames": 0, "visible_frames": 0,
        "person_frames": 0, "flow_ready": 0,
        "flow_ms": [], "coherence": [], "drift": [], "struggle": [],
        "submersion": [], "speed_ms": [], "exposure": [],
        "alert_frames": 0, "phases": {},
        "motion_pct": [], "brightness": [],
    }
    main_id = None          # person with most visibility so far
    main_vis = {}

    for a, b in segs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        for i in range(a, b):
            ok, frame = cap.read()
            if not ok:
                break
            st["frames"] += 1
            st["pose_ready_frames"] += int(det.pose_ready)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            st["brightness"].append(float(np.mean(gray)))

            boxes = det.detect_people(frame)[:MAX_BOXES]
            obs_list = det.observe(i, frame, boxes)
            tracks = tracker.update(boxes)
            st["person_frames"] += int(bool(boxes))

            # pick the most persistent visible person
            by_track = {}
            for tr, ob in zip(tracks, obs_list):
                by_track[tr.track_id] = ob
            if main_id is None or main_vis.get(main_id, 0) == 0:
                for tid, ob in by_track.items():
                    if ob.visible:
                        main_id = tid
                        break
            main_vis[main_id] = main_vis.get(main_id, 0) + 1 \
                if main_id in by_track else 0

            chosen = by_track.get(main_id)
            if chosen is None and by_track:
                chosen = next(iter(by_track.values()))
            st["visible_frames"] += int(bool(chosen and chosen.visible))

            fr = flow.estimate(frame, exclude_boxes=boxes)
            if fr.ready:
                st["flow_ready"] += 1
                st["flow_ms"].append(fr.speed_ms)
                st["coherence"].append(fr.coherence)
            feats = fex.update(chosen, fr)
            engine_state = engine.update(1, feats)
            if engine_state.alert_active:
                st["alert_frames"] += 1
            st["phases"][engine_state.phase] = \
                st["phases"].get(engine_state.phase, 0) + 1

            st["drift"].append(feats.get("drift_ratio", 0.0))
            st["struggle"].append(feats.get("struggle", 0.0))
            st["submersion"].append(feats.get("submersion", 0.0))
            st["speed_ms"].append(feats.get("speed_ms", 0.0))
            st["exposure"].append(feats.get("exposure", 0.0))
            if i > a:
                st["motion_pct"].append(float(np.mean(
                    cv2.absdiff(gray, prev_gray) > 18)))
            prev_gray = gray
    cap.release()

    def _m(key, digits=3):
        v = st[key]
        return (round(float(np.mean(v)), digits), round(float(np.max(v)), digits)
                if v else 0.0) if v else (0.0, 0.0)

    flow_m, flow_max = _m("flow_ms", 2)
    coh_m, _ = _m("coherence", 2)
    drift_m, drift_max = _m("drift")
    strug_m, strug_max = _m("struggle")
    sub_m, sub_max = _m("submersion")
    spd_m, _ = _m("speed_ms", 2)
    exp_m, _ = _m("exposure")
    mot_m, _ = _m("motion_pct", 2)
    bri_m, _ = _m("brightness", 0)
    n = max(1, st["frames"])
    return {
        "file": os.path.basename(path), "total_frames": total,
        "fps": round(fps, 2),
        "analyzed": st["frames"],
        "pose_ready": round(st["pose_ready_frames"] / n, 2),
        "person_pct": round(st["person_frames"] / n, 2),
        "visible_pct": round(st["visible_frames"] / n, 2),
        "flow_ready_pct": round(st["flow_ready"] / n, 2),
        "flow_ms_mean": flow_m, "flow_ms_max": flow_max,
        "coherence_mean": coh_m,
        "drift_mean": drift_m, "drift_max": drift_max,
        "struggle_mean": strug_m, "struggle_max": strug_max,
        "submersion_mean": sub_m, "submersion_max": sub_max,
        "speed_ms_mean": spd_m, "exposure_mean": exp_m,
        "motion_pct": mot_m, "brightness_mean": bri_m,
        "alert_pct": round(st["alert_frames"] / n, 2),
        "phases": st["phases"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=999)
    ap.add_argument("--out", default="data_probe.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.mp4")))
    # skip exact duplicates (name(N).ext twin with identical size)
    kept = []
    seen_size = {}
    for f in files:
        sz = os.path.getsize(f)
        base = os.path.basename(f)
        if "(1)" in base and seen_size.get(sz) is not None:
            print(f"skip duplicate: {base}")
            continue
        seen_size[sz] = base
        kept.append(f)

    results = []
    for f in kept[args.start:args.end]:
        print(f"probing {os.path.basename(f)} ...", flush=True)
        r = probe_file(f)
        results.append(r)
        print(f"  person={r['person_pct']:.0%} visible={r['visible_pct']:.0%} "
              f"flow={r['flow_ms_mean']:.2f}m/s coh={r['coherence_mean']:.2f} "
              f"drift={r['drift_mean']:.2f} struggle={r['struggle_mean']:.2f} "
              f"subm={r['submersion_mean']:.2f} alert={r['alert_pct']:.0%} "
              f"phases={r['phases']}", flush=True)

    all_rows = []
    if os.path.isfile(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as fh:
                all_rows = json.load(fh)
        except Exception:  # noqa: BLE001
            all_rows = []
    by_file = {r["file"]: r for r in all_rows}
    for r in results:
        by_file[r["file"]] = r
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(list(by_file.values()), fh, indent=2)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
