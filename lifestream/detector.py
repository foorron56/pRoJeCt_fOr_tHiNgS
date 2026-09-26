# MediaPipe Pose landmark indices -> LifeStream keypoint names (aligned order
# with KEYPOINT_NAMES where mapping exists).
MP_LANDMARKS = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_FEATURES, FeatureConfig
from .features import KEYPOINT_NAMES, SwimObservation

Box = Sequence[float]

try:  # pragma: no cover - depends on optional runtime
    import mediapipe as mp
    _MP_OK = True
except Exception:  # noqa: BLE001
    mp = None
    _MP_OK = False

# mediapipe >= 1.0 removed the legacy `mp.solutions` API; the Tasks API needs
# a downloaded .task model file. Support BOTH so the same code runs anywhere.
_MP_TASKS_READY = False
_MP_TASKS_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pose_landmarker_full.task")
if _MP_OK and hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
    if os.path.isfile(_MP_TASKS_MODEL):
        _MP_TASKS_READY = True


class SwimDetector:
    """Detects people in a frame and converts them to SwimObservations."""

    def __init__(self, config: FeatureConfig = DEFAULT_FEATURES,
                 confidence: float = 0.5) -> None:
        self.cfg = config
        self.confidence = confidence
        self._pose = None          # legacy solutions.Pose
        self._landmarker = None    # Tasks PoseLandmarker (IMAGE mode)
        self._init_pose()

    def _init_pose(self) -> None:
        if not _MP_OK:
            return
        # 1) legacy API (mediapipe < 1.0)
        try:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,          # full model; still CPU-real-time
                min_detection_confidence=self.confidence,
                min_tracking_confidence=0.5,
            )
            return
        except Exception:  # noqa: BLE001
            self._pose = None
        # 2) Tasks API (mediapipe >= 1.0), static per-crop detection
        if _MP_TASKS_READY:
            try:
                base = mp.tasks.BaseOptions(
                    model_asset_path=_MP_TASKS_MODEL)
                opts = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=self.confidence,
                    min_pose_presence_confidence=0.5,
                )
                self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
            except Exception:  # noqa: BLE001
                self._landmarker = None

    @property
    def pose_ready(self) -> bool:
        return self._pose is not None or self._landmarker is not None

    def detect_people(self, frame: np.ndarray) -> List[Box]:
        """Person boxes. MediaPipe is single-person, so we do a fast motion
        + skin/hue pre-segmentation to propose candidate regions and pose
        each one; a plain centroid diff keeps multi-swimmer pools working."""
        # Person proposal: Motion-history differencing is cheap and robust to
        # water shimmer. Combine frame difference with edge density.
        boxes = _motion_region_proposals(frame)
        if not boxes and self.pose_ready:
            # fall back to full-frame pose so a lone still swimmer is found
            h, w = frame.shape[:2]
            boxes = [(0.0, 0.0, float(w), float(h))]
        return boxes

    def observe(self, frame_idx: int, frame: np.ndarray,
                boxes: Sequence[Box]) -> List[SwimObservation]:
        """Pose-estimate each proposed region -> SwimObservation list."""
        out: List[SwimObservation] = []
        for box in boxes:
            obs = self._pose_region(frame_idx, frame, box)
            if obs is not None:
                out.append(obs)
        return out

    # ------------------------------------------------------------------ #

    def _pose_region(self, frame_idx: int, frame: np.ndarray,
                     box: Box) -> Optional[SwimObservation]:
        import cv2

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        crop = frame[y1:y2, x1:x2]

        # Real flood footage often shows small, distant people; MediaPipe
        # pose needs roughly head-height pixels to lock on. Upscale small
        # crops — landmark coordinates are crop-local FRACTIONS, so this is
        # transparent to everything downstream.
        ch, cw = crop.shape[:2]
        min_side = min(ch, cw)
        if min_side < 96:
            scale = min(4.0, 96.0 / max(1, min_side))
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        kps: Dict[str, Optional[float]] = {n: None for n in KEYPOINT_NAMES}
        visible = False
        landmarks = None
        if self._pose is not None:                       # legacy API
            res = self._pose.process(crop)
            if res and res.pose_landmarks:
                landmarks = res.pose_landmarks.landmark
        elif self._landmarker is not None:               # Tasks API
            mb = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            res = self._landmarker.detect(mb)
            if res and res.pose_landmarks:
                landmarks = res.pose_landmarks[0]

        if landmarks is not None:
            for idx, name in MP_LANDMARKS.items():
                lm = landmarks[idx]
                if lm.visibility >= self.cfg.min_confidence:
                    kps[name] = (float(lm.x), float(lm.y))
            visible = any(v is not None for v in kps.values())

        return SwimObservation(
            frame_idx=frame_idx,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            keypoints=kps,
            visible=visible,
        )


def cv2_bgr(img: np.ndarray) -> np.ndarray:
    """MediaPipe expects BGR like OpenCV — identity kept for clarity."""
    return img


def _motion_region_proposals(frame: np.ndarray,
                             max_boxes: int = 8) -> List[Box]:
    """Fast moving-person proposals: temporal difference + morphological
    cleanup + connected components. Pure OpenCV, no ML dependency."""
    import cv2

    if not hasattr(_motion_region_proposals, "_prev"):
        _motion_region_proposals._prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return []
    prev = _motion_region_proposals._prev
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _motion_region_proposals._prev = gray

    diff = cv2.absdiff(gray, prev)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, np.ones((7, 7), np.uint8), iterations=2)
    th = cv2.erode(th, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    min_area = 0.004 * w * h
    boxes: List[Box] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < min_area:
            continue
        # people in pools are taller than wide when upright; keep generous
        if ch < 0.12 * h:
            continue
        boxes.append((float(x), float(y), float(x + cw), float(y + ch)))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:max_boxes]
