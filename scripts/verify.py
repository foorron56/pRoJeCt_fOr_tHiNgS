# -*- coding: utf-8 -*-
"""LifeStream AI — quick verification script (docs/examples/demo.md)."""
from __future__ import annotations

import time

from lifestream.alerts import AlertConfig, AlertManager
from lifestream.pipeline import LifeStreamPipeline
from lifestream.risk_engine import DISTRESS, SINKING, STILL_INACTIVE
from lifestream.simulator import PoolSimulator


def silent_logger():
    class L:
        def info(self, line):
            pass
    return L()


def run(behavior: str, seconds: float = 26.0) -> None:
    sim = PoolSimulator()
    sim.set_behavior("victim", behavior)
    alerts = AlertManager(AlertConfig(log_file="verify_alerts.log"),
                          logger=silent_logger())
    pipe = LifeStreamPipeline(alert_manager=alerts)
    dt = 1.0 / 25.0
    print(f"\n=== scenario: {behavior} ({seconds:.0f}s) ===")
    fired = False
    for i in range(int(seconds * 25)):
        pf = pipe.process(sim.step(dt))
        st = pf.states[1]                      # victim = first spawned track
        if i % 50 == 0 or (st.alert_active and not fired):
            terms = ", ".join(f"{k}={v:.2f}" for k, v in st.terms.items())
            print(f"t={i/25:5.1f}s  phase={st.phase:<14} score={st.score:.2f}"
                  f"  alert={st.alert_active}  [{terms}]")
        fired = fired or st.alert_active
    print(f"-> alert fired: {fired}; events: "
          f"{[e.kind for e in alerts.events]}")


if __name__ == "__main__":
    for scenario in ("laps", "playing", "distress", "still"):
        run(scenario)
    print("\nAll verification scenarios completed.")
