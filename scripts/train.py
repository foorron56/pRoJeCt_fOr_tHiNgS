# -*- coding: utf-8 -*-
"""Train the LifeStream risk scorers (pool + flood) and save them to models/.

Usage:
    python scripts/train.py                 # train both, print metrics
    python scripts/train.py --episodes 400  # more data, slower
    python scripts/train.py --kind flood    # just one mode

What happens:
  1. roll randomized swim/flood episodes through the SAME feature extractors
     the runtime uses (lifestream/features.py, flood physics mirror),
  2. label every frame with the physics heuristic of danger in progress,
  3. fit a standardized, class-balanced L2 logistic regression (numpy only),
  4. validate on held-out episodes and report precision/recall/F1/fp-rate,
  5. save models/pool_scorer.json and models/flood_scorer.json, then smoke-
     test them end-to-end on the four demo scenarios in each mode.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lifestream.flood_pipeline import FloodPipeline        # noqa: E402
from lifestream.pipeline import LifeStreamPipeline        # noqa: E402
from lifestream.scorer import LearnedScorer               # noqa: E402
from lifestream.training import (build_flood_dataset,     # noqa: E402
                                 build_pool_dataset, train_logistic)


class _Silent:
    def info(self, line: str) -> None:
        pass


def train_pool(episodes: int, seed: int) -> LearnedScorer:
    print(f"[pool] building dataset ({episodes} randomized episodes)...")
    ds = build_pool_dataset(episodes=episodes, seed=seed)
    print(f"[pool] dataset: {len(ds.y)} frames, "
          f"danger={int(ds.y.sum())} ({ds.y.mean():.1%})")
    res = train_logistic(ds, alert_threshold=0.5)
    print(f"[pool] train n={res.model['n_train']}  val n={res.model['n_val']}")
    print(f"[pool] val metrics: {res.metrics}")
    weights = dict(zip(res.model["feature_names"], res.model["weights"]))
    print(f"[pool] learned weights: "
          f"{ {k: round(v, 2) for k, v in weights.items()} }")
    path = os.path.join("models", "pool_scorer.json")
    LearnedScorer(res.model).save(path)
    print(f"[pool] saved -> {path}")
    return LearnedScorer(res.model)


def train_flood(episodes: int, seed: int) -> LearnedScorer:
    print(f"[flood] building dataset ({episodes} randomized episodes)...")
    ds = build_flood_dataset(episodes=episodes, seed=seed + 1)
    print(f"[flood] dataset: {len(ds.y)} frames, "
          f"danger={int(ds.y.sum())} ({ds.y.mean():.1%})")
    res = train_logistic(ds, alert_threshold=0.5)
    print(f"[flood] train n={res.model['n_train']}  val n={res.model['n_val']}")
    print(f"[flood] val metrics: {res.metrics}")
    weights = dict(zip(res.model["feature_names"], res.model["weights"]))
    print(f"[flood] learned weights: "
          f"{ {k: round(v, 2) for k, v in weights.items()} }")
    path = os.path.join("models", "flood_scorer.json")
    LearnedScorer(res.model).save(path)
    print(f"[flood] saved -> {path}")
    return LearnedScorer(res.model)


def smoke_test_pool(scorer: LearnedScorer) -> None:
    """The demo scenarios must still behave with the learned scorer."""
    from lifestream.alerts import AlertConfig, AlertManager
    from lifestream.simulator import PoolSimulator

    print("[pool] end-to-end smoke test with the trained scorer:")
    for behavior, expect_alert in (("laps", False), ("playing", False),
                                   ("distress", True), ("still", True)):
        sim = PoolSimulator()
        sim.set_behavior("victim", behavior)
        alerts = AlertManager(AlertConfig(log_file="train_alerts.log"),
                              logger=_Silent())
        pipe = LifeStreamPipeline(alert_manager=alerts, scorer=scorer)
        dt = 1.0 / 25.0
        fired = False
        first_fire = None
        for i in range(int(24 * 25)):
            pf = pipe.process(sim.step(dt))
            st = pf.states[1]
            if st.alert_active and not fired:
                fired, first_fire = True, i / 25.0
        ok = fired == expect_alert
        mark = "OK " if ok else "FAIL"
        when = f" at {first_fire:.1f}s" if fired else ""
        print(f"  [{mark}] {behavior:9s} alert={fired}{when}")
    print("[pool] smoke test done.")


def smoke_test_flood(scorer: LearnedScorer) -> None:
    from lifestream.alerts import AlertConfig, AlertManager
    from lifestream.flood_sim import FloodSimulator

    print("[flood] end-to-end smoke test with the trained scorer:")
    for scenario, expect_alert in (("wading", False), ("drift", True),
                                   ("clinging", True), ("stranded", True)):
        sim = FloodSimulator(scenario=scenario)
        alerts = AlertManager(AlertConfig(log_file="train_alerts.log"),
                              logger=_Silent())
        pipe = FloodPipeline(alert_manager=alerts, scorer=scorer)
        dt = 1.0 / 25.0
        fired = False
        first_fire = None
        for i in range(int(14 * 25)):
            obs = sim.step(dt)
            pf = pipe.process(obs, frame=sim.render(obs))
            for st in pf.states.values():
                if st.alert_active and not fired:
                    fired, first_fire = True, i / 25.0
        ok = fired == expect_alert
        mark = "OK " if ok else "FAIL"
        when = f" at {first_fire:.1f}s" if fired else ""
        print(f"  [{mark}] {scenario:9s} alert={fired}{when}")
    print("[flood] smoke test done.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=240,
                    help="randomized episodes per mode (default 240)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--kind", default="both", choices=["both", "pool", "flood"])
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the end-to-end demo smoke test")
    args = ap.parse_args()

    if args.kind in ("both", "pool"):
        sc = train_pool(args.episodes, args.seed)
        if not args.no_smoke:
            smoke_test_pool(sc)
    if args.kind in ("both", "flood"):
        sc = train_flood(args.episodes, args.seed)
        if not args.no_smoke:
            smoke_test_flood(sc)
    print("\nAll done. Run the demos with:  "
          "python -m lifestream --scorer learned --simulate ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
