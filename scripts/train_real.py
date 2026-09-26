# -*- coding: utf-8 -*-
"""Train the flood scorer on the REAL flood footage in data/.

Two-stage pipeline (so a timeout never loses more than one clip):

  extract   roll each clip once through the production pipeline
            (detector -> FloodFlowEstimator -> FloodFeatureExtractor),
            label frames with measured physics, cache features to
            data_cache/<clip>.npz
  train     load every cache, leave-clip-out CV, fit the final model on
            all clips, save models/flood_scorer_real.json

Measured-physics weak labels (no hand annotation needed):
  carried    — box moving along the current at >= 85% of measured flow
               speed while the flow is >= 1 m/s (the water is in control)
  submerged  — tracked box gone >= 0.5 s while flow >= 1 m/s (murky water
               swallowed the person)
  otherwise  — 0 (person visible and not carried)
  missing box with weak/no flow — NO label (empty scene vs out-of-frame
               is indistinguishable)

Human clip overrides replace physics labels per clip:
    python scripts/train_real.py --label-override labels.json
    labels.json: {"<clip-substring>": "danger" | "safe", ...}

Usage:
    python scripts/train_real.py --stage extract --clip-start 0 --clip-end 4
    python scripts/train_real.py --stage train
    python scripts/train_real.py                 # extract missing + train
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
from lifestream.flow import FloodFlowEstimator  # noqa: E402
from lifestream.flood_features import FloodFeatureExtractor  # noqa: E402
from lifestream.scorer import LearnedScorer  # noqa: E402
from lifestream.training import (FLOOD_FEATURE_NAMES, Dataset,  # noqa: E402
                                 evaluate, train_logistic)

CARRIED_DRIFT = 0.85        # along-current speed >= 85% of the flow
CARRIED_MIN_FLOW = 1.0      # m/s — below this "carried" is meaningless
SUBMERGED_MIN_FLOW = 1.0    # m/s
SUBMERGED_MIN_FRAMES = 15   # ~0.5 s at 30 fps
WARMUP_FRAMES = 45          # flow pairs + extractor windows need ~1.5 s
CACHE_DIR = "data_cache"


# ----------------------------------------------------------------- extract
def roll_video(path: str, stride: int = 1) -> dict:
    """One pass over a clip -> feature rows + measured-physics labels."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"rows": [], "y": [], "stats": {"error": "cannot open"}}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_eff = fps / max(1, stride)  # we only see every `stride`-th frame

    det = SwimDetector(confidence=0.5)
    flow = FloodFlowEstimator(fps=fps_eff)
    fex = FloodFeatureExtractor(fps=fps_eff)

    rows: list = []
    ys: list = []
    st = {"visible": 0, "missing": 0, "carried": 0, "submerged": 0,
          "labeled": 0, "flow_ready": 0, "frames": 0, "fps": fps,
          "max_drift": 0.0, "max_flow": 0.0, "mean_flow": 0.0}
    missing_run = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride:
            i += 1
            continue
        st["frames"] += 1

        boxes = det.detect_people(frame)[:2]
        if boxes:
            b = max(boxes, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
            from lifestream.features import KEYPOINT_NAMES, SwimObservation
            obs = SwimObservation(frame_idx=i, bbox=b,
                                  keypoints={n: None for n in KEYPOINT_NAMES},
                                  visible=False)
            missing_run = 0
        else:
            obs = None
            missing_run += 1

        fr = flow.estimate(frame, exclude_boxes=boxes)
        feats = fex.update(obs, fr)

        flow_ok = fr.ready and fr.coherence >= 0.5
        flow_ms = fr.speed_ms if flow_ok else 0.0
        if flow_ok:
            st["flow_ready"] += 1
            st["max_flow"] = max(st["max_flow"], flow_ms)
            st["mean_flow"] += flow_ms
        drift = feats.get("drift_ratio", 0.0)
        if obs is not None:
            st["visible"] += 1
            st["max_drift"] = max(st["max_drift"], drift)

        # ---------------- measured-physics weak label ----------------
        y = None
        if i >= WARMUP_FRAMES:
            if obs is not None:
                if (flow_ok and flow_ms >= CARRIED_MIN_FLOW
                        and drift >= CARRIED_DRIFT
                        and fex.frames_seen >= fps_eff):
                    y = 1
                    st["carried"] += 1
                else:
                    y = 0
            else:
                st["missing"] += 1
                if (flow_ok and flow_ms >= SUBMERGED_MIN_FLOW
                        and missing_run >= SUBMERGED_MIN_FRAMES):
                    y = 1
                    st["submerged"] += 1
        if y is not None:
            rows.append([feats.get(k, 0.0) for k in FLOOD_FEATURE_NAMES])
            ys.append(y)
            st["labeled"] += 1
        i += 1
    cap.release()
    if st["flow_ready"]:
        st["mean_flow"] = round(st["mean_flow"] / st["flow_ready"], 2)
    return {"rows": rows, "y": ys, "stats": st}


def stage_extract(files: list, stride: int, lo: int, hi: int,
                  force: bool) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    sel = files[lo:hi] if hi > 0 else files[lo:]
    print(f"extracting {len(sel)} clips (stride={stride}) -> {CACHE_DIR}/")
    for path in sel:
        stem = os.path.splitext(os.path.basename(path))[0]
        npz = os.path.join(CACHE_DIR, stem + ".npz")
        js = os.path.join(CACHE_DIR, stem + ".json")
        if os.path.exists(npz) and not force:
            print(f"  {stem}: cached, skip (use --force to redo)")
            continue
        res = roll_video(path, stride=stride)
        st = res["stats"]
        if "error" in st:
            print(f"  {stem}: ERROR {st['error']}")
            continue
        np.savez_compressed(npz,
                            rows=np.asarray(res["rows"], np.float32),
                            y=np.asarray(res["y"], np.int8))
        with open(js, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
        print(f"  {stem}: {st['frames']}fr vis={st['visible']} "
              f"miss={st['missing']} carry={st['carried']} "
              f"subm={st['submerged']} labeled={st['labeled']} "
              f"maxDr={st['max_drift']:.2f} maxFl={st['max_flow']:.2f} "
              f"meanFl={st['mean_flow']:.2f}")


# ------------------------------------------------------------------- train
def stage_train(files: list, overrides: dict, epochs: int, out: str) -> int:
    caches = []
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        npz = os.path.join(CACHE_DIR, stem + ".npz")
        if os.path.exists(npz):
            caches.append((stem, npz))
    if not caches:
        print("no caches — run --stage extract first")
        return 1

    X_rows, y_all, clip_names = [], [], []
    print(f"\n{'clip':36s} {'labeled':>7s} {'danger':>6s} {'safe':>6s} "
          f"{'override':>9s}")
    for ci, (stem, npz) in enumerate(caches):
        d = np.load(npz)
        rows, y = d["rows"].astype(float), d["y"].astype(int)
        human = next((v for k, v in overrides.items()
                      if k.lower() in stem.lower()), None)
        if human == "danger":
            y = np.ones_like(y)
        elif human == "safe":
            y = np.zeros_like(y)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        print(f"{stem[:36]:36s} {len(y):7d} {n_pos:6d} {n_neg:6d} "
              f"{human or '-':>9s}")
        X_rows.extend(rows.tolist())
        y_all.extend(y.tolist())
        clip_names.extend([ci] * len(y))

    X = np.asarray(X_rows, float)
    y = np.asarray(y_all, int)
    clip_of = np.asarray(clip_names)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"\ndataset: {len(y)} labeled frames (danger={n_pos}, safe={n_neg}) "
          f"from {len(caches)} clips")
    if n_pos == 0 or n_neg == 0:
        print("single-class data — nothing to train. Use --label-override "
              "to mark clips, or lower the label thresholds.")
        return 1

    # ---------------- leave-clip-out cross-validation ----------------
    clips = np.unique(clip_of)
    k = min(5, len(clips))
    folds = np.array_split(np.random.default_rng(11).permutation(clips), k)
    agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    print(f"\nleave-clip-out CV ({k} folds over {len(clips)} clips):")
    for fi, fold in enumerate(folds):
        val_clips = set(int(c) for c in fold)
        val_mask = np.isin(clip_of, list(val_clips))
        tr_ds = Dataset("flood", list(FLOOD_FEATURE_NAMES), X[~val_mask],
                        y[~val_mask],
                        [{"episode": int(c)} for c in clip_of[~val_mask]])
        res = train_logistic(tr_ds, epochs=args_epochs())
        Xs_val = ((X[val_mask] - np.asarray(res.model["mean"]))
                  / np.asarray(res.model["std"]))
        m = evaluate(res.model, Xs_val, y[val_mask])
        for key in agg:
            agg[key] += m[key]
        print(f"  fold {fi}: clips {sorted(val_clips)} "
              f"n_val={int(val_mask.sum())} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} fp={m['fp_rate']:.4f}")

    tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    cv_metrics = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(f1, 4), "fp_rate": round(fpr, 4),
                  "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    print(f"  CV aggregate: P={prec:.3f} R={rec:.3f} F1={f1:.3f} fp={fpr:.4f}")

    # ---------------- final model on ALL clips ----------------
    ds = Dataset("flood", list(FLOOD_FEATURE_NAMES), X, y,
                 [{"episode": int(c)} for c in clip_of])
    final = train_logistic(ds, epochs=args_epochs())
    model = final.model
    model["metrics"] = cv_metrics                  # honest generalization
    model["internal_val_metrics"] = final.metrics
    model["trained_on"] = "real_footage"
    model["source_clips"] = [c for c, _ in caches]
    model["label_rules"] = {
        "carried_drift": CARRIED_DRIFT, "carried_min_flow_ms": CARRIED_MIN_FLOW,
        "submerged_min_flow_ms": SUBMERGED_MIN_FLOW,
        "submerged_min_frames": SUBMERGED_MIN_FRAMES,
        "human_overrides": overrides}
    weights = dict(zip(model["feature_names"], model["weights"]))
    print(f"\nfinal weights: { {k: round(v, 2) for k, v in weights.items()} }")

    LearnedScorer(model).save(out)
    print(f"saved -> {out}  (simulator model untouched)")
    return 0


_EPOCHS = 4000


def args_epochs() -> int:
    return _EPOCHS


def main() -> int:
    global _EPOCHS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data")
    ap.add_argument("--stage", choices=["extract", "train", "all"],
                    default="all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--clip-start", type=int, default=0)
    ap.add_argument("--clip-end", type=int, default=0,
                    help="exclusive; 0 = to the end")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if cached")
    ap.add_argument("--label-override", default=None,
                    help="JSON {clip-substring: danger|safe}")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", default=os.path.join("models",
                                                  "flood_scorer_real.json"))
    args = ap.parse_args()
    _EPOCHS = args.epochs

    overrides = {}
    if args.label_override:
        with open(args.label_override, "r", encoding="utf-8") as fh:
            overrides = json.load(fh)

    files = [f for f in sorted(glob.glob(os.path.join(args.dir, "*.mp4")))
             if "(1)" not in os.path.basename(f)]
    if not files:
        print("no clips found in", args.dir)
        return 1

    if args.stage in ("extract", "all"):
        stage_extract(files, args.stride, args.clip_start, args.clip_end,
                      args.force)
    if args.stage in ("train", "all"):
        return stage_train(files, overrides, args.epochs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
