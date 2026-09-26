"""Central configuration for LifeStream AI.

All thresholds live here so judges/reviewers can tune sensitivity without
touching algorithm code. Values are calibrated for 25–30 fps cameras
(pool module and flood module).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrackerConfig:
    """Multi-swimmer tracking (IoU-free centroid tracker)."""

    max_distance: float = 90.0        # px; max centroid jump to keep identity
    max_missing: int = 12             # frames a swimmer may vanish before removal


@dataclass
class FeatureConfig:
    """Per-swimmer behavioral feature extraction."""

    window: int = 24                  # ~1 s feature window at 25 fps
    min_confidence: float = 0.35      # below this a keypoint is "not visible"
    surface_band_frac: float = 0.45   # top band of the swim bbox = water surface
    torso_keypoints: tuple = (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    )


@dataclass
class RiskConfig:
    """Drowning-risk scoring engine."""

    calm_frames: int = 26             # STILL_INACTIVE after ~1 s of stillness
    descent_frames: int = 20          # SINKING after sustained downward drift
    danger_frames: int = 10           # grace frames before an active alert fires
    hysteresis_frames: int = 20       # frames of low score needed to clear an alert
    weights: dict = field(default_factory=lambda: {
        "vertical_posture": 0.30,     # head-up torso while not swimming laps
        "limb_distress": 0.25,        # frantic thrashing of arms/legs
        "stillness": 0.25,            # motionless after being active
        "submersion": 0.15,           # head/torso sinking below surface band
        "isolated": 0.05,             # no other swimmers nearby to help
    })


@dataclass
class AlertConfig:
    """Alarm output behavior."""

    cooldown_s: float = 3.0           # min seconds between repeated alerts
    log_file: str = "alerts.log"      # persistent audit trail for lifeguards
    event_label: str = "possible drowning"   # human-readable event description


DEFAULT_TRACKER = TrackerConfig()
DEFAULT_FEATURES = FeatureConfig()
DEFAULT_RISK = RiskConfig()
DEFAULT_ALERT = AlertConfig()


@dataclass
class FloodConfig:
    """Flood Guard sensing: camera scale, flow estimation, feature windows."""

    pixels_per_meter: float = 50.0    # camera scale calibration (px per meter)
    window: int = 30                  # feature window ~1.2 s at 25 fps
    flow_downscale: int = 4           # optical-flow working resolution divider
    flow_min_speed: float = 0.25      # px/frame noise floor (downscaled)
    flow_min_support: int = 40        # min usable flow vectors for an estimate
    exposure_s: float = 20.0          # seconds in water for full exposure score


@dataclass
class FloodRiskConfig:
    """Flood-risk gates and scoring (phases WADING/SWEPT/STRANDED/CRITICAL)."""

    swept_ratio: float = 0.80         # person velocity >= 80% of current = swept
    swept_frames: int = 50            # sustained 2 s at 25 fps
    knockdown_ms: float = 1.0         # current fast enough to carry a person
    stranded_flow_ms: float = 1.2     # "strong current" for the stall gate
    stranded_speed_frac: float = 0.12 # speed <= 12% of current = stalled
    stranded_frames: int = 160        # stalled 6.4 s in strong current
    critical_submersion: float = 0.18 # >=18% of window lost under murky water
    critical_frames: int = 25         # sustained 1 s
    danger_frames: int = 8            # grace frames before an alert fires
    hysteresis_frames: int = 30       # frames of low score needed to clear
    weights: dict = field(default_factory=lambda: {
        "drift_ratio": 0.30,          # carried by the current, no control
        "submersion": 0.25,           # repeatedly lost under murky water
        "struggle": 0.15,             # frantic high-frequency body motion
        "flow_exposure": 0.15,        # strong current x time in water
        "stranded": 0.15,             # stalled against the current (gate)
    })


DEFAULT_FLOOD = FloodConfig()
DEFAULT_FLOOD_RISK = FloodRiskConfig()
