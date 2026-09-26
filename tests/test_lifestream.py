"""Unit + scenario tests for LifeStream AI.

Run: pytest -q
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from lifestream.alerts import AlertEvent, AlertManager, events_to_jsonl
from lifestream.config import (AlertConfig, FeatureConfig, RiskConfig,
                               TrackerConfig)
from lifestream.features import (KEYPOINT_NAMES, SwimFeatureExtractor,
                                 SwimObservation)
from lifestream.flow import FlowResult
from lifestream.flood_engine import (CRITICAL, STRANDED, SWEPT, WADING,
                                     FloodRiskEngine)
from lifestream.flood_features import FloodFeatureExtractor
from lifestream.flood_pipeline import FloodPipeline
from lifestream.flood_sim import FloodSimulator
from lifestream.pipeline import LifeStreamPipeline
from lifestream.risk_engine import (DISTRESS, LAPS, SINKING, STILL_INACTIVE,
                                    DrowningRiskEngine)
from lifestream.simulator import PoolSimulator
from lifestream.tracker import CentroidTracker


class SilentLogger:
    def __init__(self):
        self.lines = []

    def info(self, line):
        self.lines.append(line)


def make_manager(**kw) -> AlertManager:
    cfg = AlertConfig(log_file="test_alerts.log", **{k: v for k, v in kw.items()})
    return AlertManager(cfg, logger=SilentLogger())


def silent_pipeline(**kw) -> LifeStreamPipeline:
    return LifeStreamPipeline(
        alert_manager=AlertManager(AlertConfig(log_file="test_alerts.log"),
                                   logger=SilentLogger()), **kw)


# --------------------------------------------------------------------- #
# tracker

def test_tracker_keeps_identity_for_slow_motion():
    tr = CentroidTracker(TrackerConfig(max_distance=50, max_missing=5))
    t1 = tr.update([(100, 100, 160, 200)])
    t2 = tr.update([(104, 102, 164, 202)])
    assert t1[0].track_id == t2[0].track_id


def test_tracker_splits_two_swimmers():
    tr = CentroidTracker()
    a = tr.update([(50, 50, 110, 200), (400, 60, 460, 210)])
    b = tr.update([(56, 52, 116, 202), (394, 58, 454, 208)])
    assert len(b) == 2
    assert {t.track_id for t in b} == {t.track_id for t in a}


def test_tracker_drops_vanished_swimmer():
    tr = CentroidTracker(TrackerConfig(max_distance=50, max_missing=3))
    tr.update([(50, 50, 110, 200)])
    for _ in range(5):
        tracks = tr.update([])
    assert tracks == []


# --------------------------------------------------------------------- #
# feature extraction

def kp(x, y):
    return (x, y)


def obs_full_body(head_y, sway=0.0, motion=0.0, frame=0, visible=True):
    """Vertical upright body in bbox-local fractions; whole body bobs by
    ``motion`` so motion-energy registers (real swimmers move everything)."""
    bob = motion * math.cos(frame * 0.9)
    kps = {n: None for n in KEYPOINT_NAMES}
    kps["nose"] = kp(0.5, head_y + bob)
    kps["left_shoulder"] = kp(0.38, head_y + 0.14 + bob)
    kps["right_shoulder"] = kp(0.62, head_y + 0.14 + bob)
    kps["left_hip"] = kp(0.45, head_y + 0.42 + bob)
    kps["right_hip"] = kp(0.55, head_y + 0.42 + bob)
    kps["left_wrist"] = kp(0.2, head_y + 0.30 + sway + bob)
    kps["right_wrist"] = kp(0.8, head_y + 0.30 - sway + bob)
    kps["left_ankle"] = kp(0.45, head_y + 0.75 + motion + bob)
    kps["right_ankle"] = kp(0.55, head_y + 0.75 - motion + bob)
    return SwimObservation(frame_idx=frame, bbox=(0, 0, 100, 200),
                           keypoints=kps, visible=visible)


def test_vertical_posture_distinguishes_upright_from_horizontal():
    ex = SwimFeatureExtractor()
    for i in range(20):
        ex.update(obs_full_body(0.08, frame=i))
    assert ex.features()["vertical_posture"] > 0.6


def test_submersion_rises_when_head_sinks():
    ex = SwimFeatureExtractor()
    for i in range(10):
        ex.update(obs_full_body(0.10, frame=i))
    calm = ex.features()["submersion"]
    ex2 = SwimFeatureExtractor()
    for i in range(10):
        ex2.update(obs_full_body(0.60, frame=i))   # head below surface band
    sunk = ex2.features()["submersion"]
    assert sunk > calm + 0.3


def test_stillness_flags_activity_stop():
    ex = SwimFeatureExtractor()
    for i in range(24):                       # active phase
        ex.update(obs_full_body(0.08, sway=0.4 * math.sin(i * 1.1),
                                motion=0.12 * math.cos(i * 0.9), frame=i))
    quiet = ex.features()["stillness"]
    for i in range(24, 48):                   # goes completely still
        ex.update(obs_full_body(0.08, frame=i))
    assert ex.features()["stillness"] > quiet


# --------------------------------------------------------------------- #
# risk engine

FEATS_SAFE = dict(vertical_posture=0.05, limb_distress=0.05, stillness=0.0,
                  submersion=0.0)
FEATS_DANGER = dict(vertical_posture=0.9, limb_distress=0.8, stillness=0.2,
                    submersion=0.5)


def test_engine_scores_danger_higher_than_safe():
    eng = DrowningRiskEngine()
    safe = eng.update(1, FEATS_SAFE, True, True)
    eng2 = DrowningRiskEngine()
    danger = eng2.update(1, FEATS_DANGER, True, True)
    assert danger.score > safe.score + 0.4


def test_engine_alert_requires_persistence_and_clears_with_hysteresis():
    eng = DrowningRiskEngine()
    fired_at = None
    for i in range(40):
        st = eng.update(1, FEATS_DANGER, True, True)
        if st.alert_active and fired_at is None:
            fired_at = i
    assert fired_at is not None and fired_at >= 5     # grace period held

    cleared_at = None
    for i in range(40, 90):
        st = eng.update(1, FEATS_SAFE, True, True)
        if not st.alert_active and cleared_at is None:
            cleared_at = i
    assert cleared_at is not None                     # hysteresis eventually clears


def test_engine_ignores_single_noisy_frame():
    eng = DrowningRiskEngine()
    eng.update(1, FEATS_SAFE, True, True)
    eng.update(1, FEATS_DANGER, True, True)           # one bad frame
    st = eng.update(1, FEATS_SAFE, True, True)
    assert not st.alert_active


def test_missing_pose_dampens_but_does_not_fake_features():
    eng = DrowningRiskEngine()
    st = eng.update(1, FEATS_DANGER, True, has_pose=False)
    st2 = DrowningRiskEngine().update(1, FEATS_DANGER, True, has_pose=True)
    assert st.score < st2.score


# --------------------------------------------------------------------- #
# alerts

def test_alert_manager_emits_fire_and_clear_once():
    class FakeState:
        alert_active = False
        score = 0.8
        phase = DISTRESS

    mgr = make_manager(cooldown_s=0.0)
    st = FakeState()
    st.alert_active = True
    mgr.notify(7, st, 10)
    mgr.notify(7, st, 11)                # still active: no duplicate
    st.alert_active = False
    mgr.notify(7, st, 12)
    kinds = [(e.kind, e.swimmer_id) for e in mgr.events]
    assert kinds == [("fired", 7), ("cleared", 7)]


def test_events_jsonl_roundtrip():
    e = AlertEvent(1, 0.77, SINKING, 42, "fired", 123.0)
    assert '"phase": "SINKING"' in events_to_jsonl([e])


# --------------------------------------------------------------------- #
# end-to-end simulation scenarios

def run_scenario(behavior: str, seconds: float, fps: float = 25.0):
    sim = PoolSimulator()
    sim.set_behavior("victim", behavior)
    pipe = silent_pipeline(fps=fps)
    dt = 1.0 / fps
    victim_states = []
    for _ in range(int(seconds * fps)):
        obs = sim.step(dt)
        pf = pipe.process(obs)
        # snapshot: RiskState is mutated in place by the engine
        victim_states.append(replace(pf.states[1],
                                     terms=dict(pf.states[1].terms)))
    return victim_states


def test_laps_never_alerts():
    states = run_scenario("laps", seconds=24)
    assert not any(s.alert_active for s in states)
    # laps should read as horizontal: vertical_posture term stays small
    assert max(s.terms["vertical_posture"] for s in states) < 0.12


def test_playing_never_alerts():
    states = run_scenario("playing", seconds=24)
    assert not any(s.alert_active for s in states)


def test_distress_alerts_quickly_and_persists():
    states = run_scenario("distress", seconds=22)
    first = next(i for i, s in enumerate(states) if s.alert_active)
    assert first < 25.0 * 12                          # alarms within ~12 s
    assert states[-1].alert_active                    # stays alarmed


def test_still_arc_alerts_after_going_quiet():
    states = run_scenario("still", seconds=26)
    assert not any(s.alert_active for s in states[:int(6 * 25)])   # treading ok
    assert any(s.alert_active for s in states[int(12 * 25):])      # sunk => alert


def test_pipeline_latency_is_realtime_capable():
    states = run_scenario("distress", seconds=4)
    pipe = silent_pipeline()
    sim = PoolSimulator()
    sim.set_behavior("victim", "distress")
    import time
    dt = 1.0 / 25.0
    worst = 0.0
    for _ in range(100):
        obs = sim.step(dt)
        t0 = time.perf_counter()
        pipe.process(obs)
        worst = max(worst, (time.perf_counter() - t0) * 1000)
    assert worst < 40.0        # well under one frame budget at 25 fps


# --------------------------------------------------------------------- #
# Flood Guard: flow, features, engine

def silent_flood_pipeline(**kw) -> FloodPipeline:
    return FloodPipeline(alert_manager=AlertManager(
        AlertConfig(log_file="test_alerts.log",
                    event_label="person in distress in floodwater"),
        logger=SilentLogger()), **kw)


FLOW_RIGHT = FlowResult(vx_ms=2.0, vy_ms=0.0, speed_ms=2.0,
                        coherence=0.9, ready=True)


def flood_feats(**kw):
    base = {"drift_ratio": 0.0, "struggle": 0.0, "submersion": 0.0,
            "flow_exposure": 0.0, "exposure": 0.0,
            "speed_ms": 0.0, "flow_ms": 0.0}
    base.update(kw)
    return base


def test_flow_result_unit_direction():
    f = FlowResult(vx_ms=3.0, vy_ms=4.0, speed_ms=5.0, ready=True)
    assert f.unit == pytest.approx((0.6, 0.8))
    assert FlowResult().unit == (0.0, 0.0)


def test_flood_engine_swept_requires_sustained_carry():
    eng = FloodRiskEngine()
    feats = flood_feats(drift_ratio=1.0, struggle=0.4, flow_exposure=0.8,
                        speed_ms=2.0, flow_ms=2.0)
    st = eng.update(1, feats)
    assert st.phase == WADING and not st.alert_active      # needs persistence
    for _ in range(70):
        st = eng.update(1, feats)
    assert st.phase == SWEPT and st.alert_active


def test_flood_engine_stranded_gate():
    eng = FloodRiskEngine()
    feats = flood_feats(drift_ratio=0.0, flow_exposure=1.0,
                        speed_ms=0.1, flow_ms=2.6)         # stalled in current
    st = None
    for _ in range(200):
        st = eng.update(1, feats)
    assert st.phase == STRANDED and st.alert_active


def test_flood_engine_critical_on_repeated_submersion():
    eng = FloodRiskEngine()
    feats = flood_feats(submersion=1.0, flow_exposure=0.8, flow_ms=2.6)
    for _ in range(40):
        st = eng.update(1, feats)
    assert st.phase == CRITICAL and st.alert_active


def test_flood_engine_wading_stays_safe():
    eng = FloodRiskEngine()
    feats = flood_feats(drift_ratio=0.1, struggle=0.3, flow_exposure=0.6,
                        speed_ms=0.5, flow_ms=2.6)         # moving in control
    for _ in range(300):
        st = eng.update(1, feats)
    assert st.phase == WADING and not st.alert_active


def test_flood_extractor_counts_missing_frames_as_submersion():
    ex = FloodFeatureExtractor()
    for _ in range(ex.cfg.window):
        f = ex.update(None, FLOW_RIGHT)                    # lost under water
    assert f["submersion"] == pytest.approx(1.0)
    for i in range(ex.cfg.window):
        f = ex.update(obs_full_body(0.08, frame=i), FLOW_RIGHT)
    assert f["submersion"] == pytest.approx(0.0)


def test_flood_extractor_drift_ratio_tracks_current():
    ex = FloodFeatureExtractor()
    dx = FLOW_RIGHT.speed_ms * ex.cfg.pixels_per_meter / 25.0   # carried along
    f = None
    for i in range(25):
        x = 100 + dx * i
        obs = SwimObservation(frame_idx=i, bbox=(x, 200, x + 80, 300),
                              keypoints={n: None for n in KEYPOINT_NAMES},
                              visible=True)
        f = ex.update(obs, FLOW_RIGHT)
    assert f["drift_ratio"] > 0.8                          # water is in control


def test_flood_sim_hides_submerged_person():
    sim = FloodSimulator(scenario="clinging", flow_ms=2.6)
    assert sim.step()[0] is None            # head under: NO observation at all
    sim2 = FloodSimulator(scenario="wading")
    assert sim2.step()[0] is not None


# --------------------------------------------------------------------- #
# Flood Guard: end-to-end scenarios (real optical flow on rendered frames)

def run_flood(scenario: str, seconds: float, fps: float = 25.0,
              flow_ms: float = 2.6):
    sim = FloodSimulator(scenario=scenario, flow_ms=flow_ms)
    pipe = silent_flood_pipeline(fps=fps)
    dt = 1.0 / fps
    timeline = []                           # per-frame snapshots (engine mutates)
    for _ in range(int(seconds * fps)):
        obs = sim.step(dt)
        pf = pipe.process(obs, frame=sim.render(obs))
        timeline.append(
            {sid: (st.phase, st.alert_active) for sid, st in pf.states.items()})
    return timeline, pipe.alerts.events


def test_flood_wading_never_alerts():
    timeline, events = run_flood("wading", seconds=14)
    # timeline holds (phase, alert_active) tuples — check alert_active only
    assert not any(a[1] for fr in timeline for a in fr.values())
    assert not [e for e in events if e.kind == "fired"]


def test_flood_drift_alerts_after_being_swept():
    timeline, events = run_flood("drift", seconds=14)
    fired = [e for e in events if e.kind == "fired"]
    assert fired and fired[0].frame_idx < 25 * 10          # alarms within 10 s
    # swept away, dunked under repeatedly — CRITICAL (worst) is acceptable too
    assert fired[0].phase in (SWEPT, CRITICAL)
    assert timeline[-1][1][1]                              # still alarming


def test_flood_clinging_alerts_on_dunking():
    timeline, events = run_flood("clinging", seconds=12)
    fired = [e for e in events if e.kind == "fired"]
    assert fired and fired[0].frame_idx < 25 * 6           # fast: going under


def test_flood_stranded_alerts_against_rock():
    timeline, events = run_flood("stranded", seconds=11)
    fired = [e for e in events if e.kind == "fired"]
    assert fired                                           # pinned in current
    assert fired[0].phase in (STRANDED, CRITICAL)


def test_flood_pipeline_latency_is_realtime_capable():
    sim = FloodSimulator(scenario="drift")
    pipe = silent_flood_pipeline()
    import time
    dt = 1.0 / 25.0
    worst = 0.0
    for _ in range(80):
        obs = sim.step(dt)
        t0 = time.perf_counter()
        pipe.process(obs, frame=sim.render(obs))
        worst = max(worst, (time.perf_counter() - t0) * 1000)
    assert worst < 120.0       # incl. optical flow + rendering, at 25 fps


# --------------------------------------------------------------------- #
# Learned scorer + training pipeline

from lifestream.scorer import LearnedScorer
from lifestream.training import (Dataset, train_logistic)


def _tiny_model_dict():
    return {
        "kind": "pool", "version": 1,
        "feature_names": ["vertical_posture", "limb_distress"],
        "weights": [2.0, 1.0], "bias": -1.5,
        "mean": [0.3, 0.3], "std": [0.3, 0.3],
        "alert_threshold": 0.5,
        "metrics": {"f1": 0.95, "recall": 0.97, "fp_rate": 0.02},
    }


def test_scorer_scores_monotonically_in_danger():
    sc = LearnedScorer(_tiny_model_dict())
    safe = sc.score({"vertical_posture": 0.0, "limb_distress": 0.0})
    mid = sc.score({"vertical_posture": 0.5, "limb_distress": 0.3})
    danger = sc.score({"vertical_posture": 1.0, "limb_distress": 1.0})
    assert safe < mid < danger
    assert danger > 0.9 and safe < 0.1


def test_scorer_contributions_and_describe():
    sc = LearnedScorer(_tiny_model_dict())
    contrib = sc.contributions({"vertical_posture": 0.6,
                                "limb_distress": 0.0})
    assert contrib["vertical_posture"] > 0.0
    assert "val_f1=0.95" in sc.describe()


def test_scorer_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "m.json")
    LearnedScorer(_tiny_model_dict()).save(p)
    sc = LearnedScorer.load(p)
    assert sc.kind == "pool"
    assert sc.weights == [2.0, 1.0]


def test_scorer_rejects_bad_version(tmp_path):
    bad = _tiny_model_dict()
    bad["version"] = 99
    p = str(tmp_path / "bad.json")
    import json
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    with pytest.raises(ValueError):
        LearnedScorer.load(p)


def test_train_logistic_learns_a_separable_boundary():
    import numpy as np
    rng = np.random.default_rng(0)
    # class 0 around origin, class 1 in the upper corner — separable
    n = 400
    x0 = rng.uniform(0.0, 0.35, size=(n, 2))
    x1 = rng.uniform(0.65, 1.0, size=(n, 2))
    X = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    ds = Dataset("pool", ["a", "b"], X, y)
    res = train_logistic(ds, epochs=1500, lr=0.5, val_fraction=0.25)
    assert res.metrics["f1"] > 0.9
    assert res.metrics["fp_rate"] < 0.1
    # both features must carry positive weight
    assert all(w > 0 for w in res.model["weights"])
    # model round-trips through LearnedScorer
    sc = LearnedScorer(res.model)
    assert sc.score({"a": 1.0, "b": 1.0}) > sc.score({"a": 0.0, "b": 0.0})


def test_pool_engine_uses_learned_score_when_given():
    class AlwaysDangerous:
        def score(self, f):
            return 0.95

        def contributions(self, f):
            return {"learned": 0.95}

    eng = DrowningRiskEngine(scorer=AlwaysDangerous())
    st = None
    for _ in range(20):
        st = eng.update(1, FEATS_SAFE, True, True)
    assert st.alert_active and "learned" in st.terms

    # without a scorer the same safe features never alert
    eng2 = DrowningRiskEngine()
    for _ in range(20):
        st2 = eng2.update(1, FEATS_SAFE, True, True)
    assert not st2.alert_active


def test_cli_scorer_falls_back_when_model_missing(monkeypatch, tmp_path):
    from lifestream import cli

    # neither the working directory nor the project root has a models/ dir
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "PROJECT_ROOT", str(tmp_path / "nowhere"))
    assert cli._load_scorer("pool", "rules") is None
    assert cli._load_scorer("pool", "learned") is None      # warn + fallback


# --------------------------------------------------------------------- #
# ML pipeline integrity: model JSON schema, seed stability, independent
# revalidation on a split the shipped model never saw.

import json as _json
import os as _os

REQUIRED_MODEL_FIELDS = ("kind", "version", "feature_names", "weights",
                         "bias", "mean", "std", "alert_threshold",
                         "trained_at", "n_train", "n_val", "seed")
RATE_METRIC_FIELDS = ("precision", "recall", "f1", "fp_rate")


@pytest.mark.skipif(not _os.path.isdir("models"),
                    reason="models/ not built (python scripts/train.py)")
def test_cli_model_lookup_falls_back_to_project_root(monkeypatch, tmp_path):
    """models/ next to the cwd wins; the packaged install is a fallback."""
    from lifestream import cli

    assert cli._model_path("pool") is not None            # run from project root
    monkeypatch.chdir(tmp_path)
    assert cli._model_path("pool") is not None            # -> packaged path
    monkeypatch.setattr(cli, "PROJECT_ROOT", str(tmp_path / "nowhere"))
    assert cli._model_path("pool") is None


@pytest.mark.parametrize("kind", ["pool", "flood"])
@pytest.mark.skipif(not _os.path.isdir("models"),
                    reason="models/ not built (python scripts/train.py)")
def test_shipped_model_json_schema(kind):
    """Trained model files must carry every field the runtime relies on,
    with consistent shapes — a malformed model must fail loudly here,
    not silently at the poolside."""
    path = _os.path.join("models", f"{kind}_scorer.json")
    with open(path, "r", encoding="utf-8") as fh:
        m = _json.load(fh)
    for field in REQUIRED_MODEL_FIELDS:
        assert field in m, f"{path}: missing '{field}'"
    assert m["kind"] == kind
    assert m["version"] == 1
    names, w = m["feature_names"], m["weights"]
    assert len(names) == len(set(names)), "duplicate feature names"
    assert len(w) == len(names) == len(m["mean"]) == len(m["std"])
    assert all(v > 1e-9 for v in m["std"]), "zero std would divide by zero"
    assert 0.0 < m["alert_threshold"] < 1.0
    assert m["n_train"] > 0 and m["n_val"] > 0
    assert isinstance(m["seed"], int)
    mets = m["metrics"]
    for field in RATE_METRIC_FIELDS:
        assert 0.0 <= mets[field] <= 1.0, f"{field} out of range"
    # it must load through the real runtime loader (version + shape checks)
    sc = LearnedScorer.load(path)
    assert sc.kind == kind and sc.alert_threshold == m["alert_threshold"]


def test_train_logistic_seed_stability():
    """Same dataset + same seed -> bit-identical weights, bias and metrics;
    retraining a deployed model must never change what it says."""
    from lifestream.training import build_pool_dataset

    ds = build_pool_dataset(episodes=5, episode_s=6.0, seed=41)
    r1 = train_logistic(ds, seed=7)
    r2 = train_logistic(ds, seed=7)
    assert r1.model["weights"] == r2.model["weights"]
    assert r1.model["bias"] == r2.model["bias"]
    assert r1.model["mean"] == r2.model["mean"]
    assert r1.model["std"] == r2.model["std"]
    assert r1.model["metrics"] == r2.model["metrics"]


def test_dataset_randomization_depends_on_seed():
    """Different dataset seeds must roll genuinely different episodes —
    otherwise 'independent' validation would be an illusion."""
    import numpy as np
    from lifestream.training import build_flood_dataset

    d1 = build_flood_dataset(episodes=4, episode_s=8.0, seed=1)
    d2 = build_flood_dataset(episodes=4, episode_s=8.0, seed=2)
    # same nominal size (withheld warm-up rows vary slightly by episode)
    assert min(d1.X.shape[0], d2.X.shape[0]) > 500
    assert abs(d1.X.shape[0] - d2.X.shape[0]) <= 0.05 * d1.X.shape[0]
    assert d1.X.shape[1] == d2.X.shape[1]
    n = min(len(d1.y), len(d2.y))
    assert not np.allclose(d1.X[:n], d2.X[:n])
    assert not np.array_equal(d1.y[:n], d2.y[:n])


@pytest.mark.parametrize("kind", ["pool", "flood"])
@pytest.mark.skipif(not _os.path.isfile("models/flood_scorer.json")
                    or not _os.path.isfile("models/pool_scorer.json"),
                    reason="trained models not built (python scripts/train.py)")
def test_shipped_model_revalidated_on_independent_split(kind):
    """The honest audit: rebuild a FRESH dataset with an unseen seed,
    standardize it with the shipped model's own mean/std (exactly what the
    runtime does), and score the shipped weights on it. The stored metrics
    must survive contact with data the model was not selected on."""
    import numpy as np
    from lifestream.training import (build_flood_dataset,
                                     build_pool_dataset, evaluate)

    shipped = LearnedScorer.load(_os.path.join("models",
                                               f"{kind}_scorer.json"))
    if kind == "pool":
        ds = build_pool_dataset(episodes=8, episode_s=22.0, seed=909)
    else:
        ds = build_flood_dataset(episodes=8, episode_s=22.0, seed=908)
    # the shipped model must have been trained on the same feature contract
    assert list(ds.names) == list(shipped.names), (
        f"feature contract drift: dataset={ds.names} model={shipped.names}")
    assert ds.X.shape[0] >= 1000, "independent set too small to mean anything"

    Xs = (ds.X - np.asarray(shipped.mean)) / np.asarray(shipped.std)
    mets = evaluate(shipped.model, Xs, ds.y)

    # regression gates — loose enough to not flake, tight enough to catch
    # the kind of collapse we fixed before (flood recall 0.68 era)
    if kind == "pool":
        assert mets["f1"] >= 0.85, mets
        assert mets["recall"] >= 0.80, mets
        assert mets["precision"] >= 0.90, mets
    else:
        assert mets["f1"] >= 0.85, mets
        assert mets["recall"] >= 0.75, mets
        assert mets["precision"] >= 0.95, mets
    assert mets["fp_rate"] <= 0.05, mets
    # and the numbers must be in the same league as the stored ones
    stored = shipped.model["metrics"]
    assert abs(mets["f1"] - stored["f1"]) <= 0.10, (mets, stored)


@pytest.mark.skipif(
    not __import__("os").path.isfile("models/pool_scorer.json"),
    reason="trained model not built yet (python scripts/train.py)")
def test_shipped_pool_model_loads_and_behaves():
    sc = LearnedScorer.load("models/pool_scorer.json")
    assert sc.kind == "pool"
    safe = sc.score({"vertical_posture": 0.05, "limb_distress": 0.05,
                     "stillness": 0.0, "submersion": 0.0, "head_dip": 0.0})
    danger = sc.score({"vertical_posture": 0.9, "limb_distress": 0.8,
                       "stillness": 0.3, "submersion": 0.5, "head_dip": 0.6})
    assert safe < sc.alert_threshold < danger
