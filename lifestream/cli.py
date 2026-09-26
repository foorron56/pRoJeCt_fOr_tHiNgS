"""LifeStream AI command-line interface.

Examples
--------
Simulated pool demo (no camera needed):
    python -m lifestream --simulate --scenario distress --seconds 30

Simulated flood demo (LifeStream Flood Guard):
    python -m lifestream --mode flood --simulate --flood-scenario drift
    python -m lifestream --mode flood --simulate --flood-scenario clinging

Real camera / RTSP / video file:
    python -m lifestream --source 0
    python -m lifestream --source rtsp://pool-cam.local/stream
    python -m lifestream --source demo_pool.mp4
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from . import __app_name__, __version__
from .alerts import AlertManager
from .config import DEFAULT_ALERT, DEFAULT_RISK, DEFAULT_TRACKER
from .detector import SwimDetector
from .flood_engine import CRITICAL, STRANDED, SWEPT, WADING
from .flood_pipeline import FloodPipeline
from .flood_sim import FloodSimulator
from .pipeline import LifeStreamPipeline
from .risk_engine import DISTRESS, SINKING, STILL_INACTIVE, SWIMMING
from .simulator import PoolSimulator

# BGR colors for HUD phases
PHASE_COLORS = {
    SWIMMING: (180, 220, 180),
    "LAPS": (200, 230, 160),
    DISTRESS: (60, 60, 240),
    STILL_INACTIVE: (40, 80, 255),
    SINKING: (40, 40, 220),
    # Flood Guard phases
    WADING: (200, 230, 160),
    SWEPT: (60, 60, 240),
    STRANDED: (40, 80, 255),
    CRITICAL: (40, 40, 220),
}

DEFAULT_FPS = 25.0
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lifestream",
        description=f"{__app_name__} v{__version__} — real-time drowning & "
                    "flood-sweep detection with Computer Vision",
    )
    p.add_argument("--mode", default="pool", choices=["pool", "flood"],
                   help="pool drowning detection or floodwater sweep detection")
    p.add_argument("--source", default=None,
                   help="camera index, video file, or rtsp/http URL")
    p.add_argument("--simulate", action="store_true",
                   help="run the built-in procedural demo (no camera)")
    p.add_argument("--scenario", default="distress",
                   choices=["distress", "still", "laps", "mixed", "playing"],
                   help="simulated pool scenario to run")
    p.add_argument("--flood-scenario", default="drift",
                   choices=["drift", "clinging", "wading", "stranded"],
                   help="simulated flood scenario to run")
    p.add_argument("--flow-ms", type=float, default=2.6,
                   help="flood simulator current speed in m/s")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="duration for --simulate runs")
    p.add_argument("--fps", type=float, default=None,
                   help="frame rate; defaults to the video file's rate, "
                        "otherwise 25")
    p.add_argument("--no-display", action="store_true",
                   help="headless run (no preview window)")
    p.add_argument("--webhook", default=None,
                   help="Slack/Discord webhook URL for alerts "
                        "(or set LIFESTREAM_WEBHOOK)")
    p.add_argument("--record", default=None,
                   help="save annotated output video to this path")
    p.add_argument("--scorer", default="rules",
                   choices=["rules", "learned"],
                   help="hand-tuned rule weights or the trained model "
                        "(models/pool_scorer.json / models/flood_scorer.json)")
    return p


def _validate_args(p: argparse.ArgumentParser, args) -> None:
    """Reject nonsense numbers up front instead of crashing mid-run."""
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error(f"--fps must be a positive number (got {args.fps!r})")
    if args.simulate and (not math.isfinite(args.seconds) or args.seconds <= 0):
        p.error(f"--seconds must be > 0 for --simulate (got {args.seconds!r})")


def _model_path(kind: str) -> Optional[str]:
    """Locate models/<kind>_scorer.json, preferring the current directory."""
    rel = os.path.join("models", f"{kind}_scorer.json")
    if os.path.isfile(rel):
        return rel
    packaged = os.path.join(PROJECT_ROOT, rel)
    if os.path.isfile(packaged):
        return packaged
    return None


def _load_scorer(kind: str, name: str):
    """Load the trained scorer for a mode; fall back to rules with a note."""
    if name != "learned":
        return None
    from .scorer import LearnedScorer

    path = _model_path(kind)
    if path is None:
        print(f"[LifeStream] WARNING: models/{kind}_scorer.json not found — "
              "run 'python scripts/train.py' first; falling back to rules.")
        return None
    sc = LearnedScorer.load(path)
    print(f"[LifeStream] using trained scorer: {sc.describe()}")
    return sc


def _build_pipeline(kind: str, fps: float, args, alerts: AlertManager):
    """Construct the pipeline for a mode at the given frame rate."""
    if kind == "flood":
        return FloodPipeline(fps=fps, alert_manager=alerts,
                             scorer=_load_scorer("flood", args.scorer))
    return LifeStreamPipeline(fps=fps,
                              tracker_cfg=DEFAULT_TRACKER,
                              risk_cfg=DEFAULT_RISK,
                              alert_cfg=DEFAULT_ALERT,
                              alert_manager=alerts,
                              scorer=_load_scorer("pool", args.scorer))


def _harden_streams() -> None:
    """Never let a non-ASCII character crash the alert output.

    Windows consoles and redirected stdout default to a legacy code page;
    printing a dash or arrow there raises UnicodeEncodeError mid-run (for
    example while a rescue alert is being written). Replacing unencodable
    characters keeps the pipeline alive on any console.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:                      # non-text or exotic stream
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _harden_streams()
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\n[LifeStream] interrupted — bye.")
        return 130


def _main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    alerts = AlertManager(DEFAULT_ALERT, webhook_url=args.webhook)

    if args.mode == "flood":
        if args.simulate:
            pipeline = _build_pipeline("flood", args.fps or DEFAULT_FPS,
                                       args, alerts)
            return _run_flood_simulated(pipeline, args)
        if args.source is None:
            parser.print_help()
            return 2
        return _run_capture(args, flood=True, alerts=alerts)

    if args.simulate:
        pipeline = _build_pipeline("pool", args.fps or DEFAULT_FPS,
                                   args, alerts)
        return _run_simulated(pipeline, args)
    if args.source is None:
        parser.print_help()
        return 2
    return _run_capture(args, flood=False, alerts=alerts)


# ---------------------------------------------------------------------- #

def _alert_counts(alerts) -> Tuple[int, int]:
    """(fired events, distinct people alerted) from the alert manager."""
    events = getattr(alerts, "events", None) or []
    fired = [e for e in events if getattr(e, "kind", "") == "fired"]
    return len(fired), len({e.swimmer_id for e in fired})


def _open_writer(path: str, fps: float, size: Tuple[int, int]):
    """Open the --record writer, warning once (and disabling) on failure."""
    import cv2

    ext = os.path.splitext(path)[1].lower()
    if ext not in _VIDEO_EXTS:
        print(f"[LifeStream] WARNING: --record {path!r} has no video "
              f"extension ({'/'.join(_VIDEO_EXTS)}) — OpenCV may refuse it.")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps), size)
    if not writer.isOpened():
        print(f"[LifeStream] WARNING: cannot open {path!r} for writing — "
              "recording disabled (check the folder exists).")
        return None
    return writer


class _Recorder:
    """Lazily opens the --record writer and appends frames to it.

    The writer is created on the first frame (the size is only known then),
    disabled permanently if it cannot be opened, and always released.
    """

    def __init__(self, args, fps: float) -> None:
        self.path: Optional[str] = args.record
        self.fps = fps
        self.writer = None
        self.failed = False

    def write(self, frame) -> None:
        if not self.path or self.failed:
            return
        if self.writer is None:
            self.writer = _open_writer(self.path, self.fps,
                                       (frame.shape[1], frame.shape[0]))
            if self.writer is None:
                self.failed = True
                return
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _run_simulated(pipeline: LifeStreamPipeline, args) -> int:
    import cv2

    if args.record:
        _ensure_parent(args.record)
    sim = PoolSimulator()
    if args.scenario == "distress":
        sim.set_behavior("victim", "distress")
    elif args.scenario == "still":
        # built-in arc: treads water 6 s, then goes quiet and sinks
        sim.set_behavior("victim", "still")
    elif args.scenario == "mixed":
        sim.set_behavior("victim", "distress")
    elif args.scenario == "playing":
        sim.set_behavior("victim", "playing")

    fps = args.fps or DEFAULT_FPS
    print(f"[LifeStream] simulated demo: scenario={args.scenario}, "
          f"{args.seconds:.1f}s, {fps:.0f} fps")
    dt = 1.0 / fps
    total = int(args.seconds * fps)
    elapsed = 0.0
    alert_frames = 0
    rec = _Recorder(args, fps)

    try:
        for _ in range(total):
            if args.scenario == "mixed" and elapsed > 18.0:
                sim.set_behavior("victim", "still")

            obs = sim.step(dt)
            elapsed += dt
            pf = pipeline.process(obs)
            alert_frames += sum(1 for s in pf.states.values() if s.alert_active)

            frame = sim.render(obs)
            _draw_hud(frame, pf)
            rec.write(frame)
            if not args.no_display:
                cv2.imshow(__app_name__, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        rec.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    fired, people = _alert_counts(pipeline.alerts)
    print(f"[LifeStream] done: {total} frames, {alert_frames} alert-active "
          f"frames, {fired} alert event(s) on {people} swimmer(s)")
    return 0


def _run_flood_simulated(pipeline: FloodPipeline, args) -> int:
    import cv2

    if args.record:
        _ensure_parent(args.record)
    sim = FloodSimulator(scenario=args.flood_scenario, flow_ms=args.flow_ms)

    fps = args.fps or DEFAULT_FPS
    print(f"[LifeStream Flood Guard] simulated demo: scenario="
          f"{args.flood_scenario}, current={args.flow_ms:.1f} m/s, "
          f"{args.seconds:.1f}s, {fps:.0f} fps")
    dt = 1.0 / fps
    total = int(args.seconds * fps)
    alert_frames = 0
    rec = _Recorder(args, fps)

    try:
        for _ in range(total):
            obs = sim.step(dt)
            frame = sim.render(obs)
            pf = pipeline.process(obs, frame=frame)
            alert_frames += sum(1 for s in pf.states.values() if s.alert_active)

            _draw_hud(frame, pf, banner="FLOOD EMERGENCY - RESCUE NOW",
                      show_flow=True)
            rec.write(frame)
            if not args.no_display:
                cv2.imshow(__app_name__, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        rec.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    fired, people = _alert_counts(pipeline.alerts)
    print(f"[LifeStream Flood Guard] done: {total} frames, {alert_frames} "
          f"alert-active frames, {fired} alert event(s) on {people} person(s)")
    return 0


# ---------------------------------------------------------------------- #

def _open_capture(source, timeout_s: float = 20.0):
    """Open a cv2.VideoCapture with a wall-clock timeout.

    Dead network streams (RTSP/HTTP) otherwise block forever inside
    ``VideoCapture()`` — the app would hang with no message at all.
    Runs the open attempt in a daemon thread and gives up after
    ``timeout_s`` seconds with a clear error. A capture that arrives only
    after the deadline is released instead of leaking a handle/thread.
    """
    import cv2
    import threading

    result = {"cap": None, "cancelled": False}

    def _open():
        try:
            cap = cv2.VideoCapture(source)
        except Exception:                      # defensive: never crash thread
            return
        if result["cancelled"]:
            cap.release()
            return
        result["cap"] = cap

    th = threading.Thread(target=_open, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        result["cancelled"] = True
        return None
    cap = result["cap"]
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    return cap


def _source_fps(cap, fallback: float) -> float:
    """Frame rate reported by the source, or ``fallback`` if unusable."""
    import cv2

    try:
        v = float(cap.get(cv2.CAP_PROP_FPS))
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(v) or v <= 0 or v > 240:
        return fallback
    return v


def _run_capture(args, flood: bool, alerts: AlertManager) -> int:
    """Shared source loop for pool and flood modes (video file / camera / URL).

    The pipeline is built *after* the capture opens so it runs at the source's
    real frame rate: the optical-flow speed estimates and the risk engine's
    time windows both scale with fps, so assuming 25 fps for a 30 fps file
    would under-report the current by ~17%.
    """
    import cv2

    if args.record:
        _ensure_parent(args.record)
    src: object = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    is_network = isinstance(args.source, str) and "://" in args.source
    cap = _open_capture(src, timeout_s=20.0 if is_network else 10.0)
    if cap is None:
        tag = "[LifeStream Flood Guard]" if flood else "[LifeStream]"
        print(f"{tag} ERROR: cannot open source {args.source!r} "
              f"(timed out or no signal)")
        return 1

    src_fps = _source_fps(cap, DEFAULT_FPS)
    fps = args.fps or src_fps
    if args.fps is None and abs(src_fps - DEFAULT_FPS) > 0.01:
        print(f"[LifeStream] source frame rate {src_fps:.1f} fps "
              "(override with --fps)")
    pipeline = _build_pipeline("flood" if flood else "pool", fps, args, alerts)
    detector = SwimDetector()
    if not detector.pose_ready:
        print("[LifeStream] WARNING: MediaPipe unavailable — running with "
              "motion proposals only; risk signals will be limited.")

    delay = 1.0 / max(1.0, fps)
    stale = 0
    rec = _Recorder(args, fps)
    tag = "[LifeStream Flood Guard]" if flood else "[LifeStream]"
    print(f"{tag} monitoring {args.source} "
          f"(pose={'on' if detector.pose_ready else 'off'}, {fps:.1f} fps, "
          "press q to quit)")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_network:
                    # a live stream can hiccup — tolerate a few empty reads
                    # before giving up; files end on the first failed read
                    stale += 1
                    if stale > int(fps) * 5:
                        print(f"{tag} stream lost after {stale} empty reads — "
                              "stopping.")
                        break
                    time.sleep(0.05)
                    continue
                break
            stale = 0
            boxes = detector.detect_people(frame)
            obs = detector.observe(pipeline._frame_idx + 1, frame, boxes)
            if flood:
                pf = pipeline.process(obs, frame=frame)
                _draw_hud(frame, pf, banner="FLOOD EMERGENCY - RESCUE NOW",
                          show_flow=True)
            else:
                pf = pipeline.process(obs)
                _draw_hud(frame, pf)
            rec.write(frame)
            if not args.no_display:
                cv2.imshow(__app_name__, frame)
                wait = max(1, int((delay - (time.perf_counter() - t0)) * 1000))
                if cv2.waitKey(wait) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        rec.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    fired, people = _alert_counts(pipeline.alerts)
    print(f"{tag} done: {pipeline._frame_idx} frames processed, "
          f"{fired} alert event(s) on {people} "
          f"{'person' if flood else 'swimmer'}(s)")
    return 0

# ---------------------------------------------------------------------- #

def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` if missing (for --record)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _draw_hud(frame: np.ndarray, pf, banner: str = "DROWNING RISK - INTERVENE",
              show_flow: bool = False) -> None:
    """Overlay per-person status and a global banner on the frame."""
    import cv2

    for sid, st in pf.states.items():
        color = (50, 50, 255) if st.alert_active else PHASE_COLORS.get(
            st.phase, (200, 200, 200))
        box = pf.boxes.get(sid)
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{sid} {st.phase} {st.score:.2f}"
        if st.alert_active:
            label = "!! " + label
        cv2.putText(frame, label, (x1, max(14, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if st.alert_active:
            cv2.putText(frame, banner, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 3)
    flow = getattr(pf, "flow", None)
    if show_flow and flow is not None and flow.ready:
        cv2.putText(frame, f"current {flow.speed_ms:.1f} m/s "
                           f"(coh {flow.coherence:.0%})",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 2)
    cv2.putText(frame, f"latency {pf.latency_ms:.1f} ms",
                (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
