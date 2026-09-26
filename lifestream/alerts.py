"""Alert outputs: audit log + optional webhook (Slack/Discord-compatible).

Only *transitions* are emitted: one "fired" event when a swimmer crosses into
alert state (subject to cooldown), one "cleared" event when they recover. The
audit log is the evidence trail for lifeguards and judges.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .config import AlertConfig, DEFAULT_ALERT
from .risk_engine import RiskState

try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # noqa: BLE001
    requests = None


@dataclass
class AlertEvent:
    swimmer_id: int
    score: float
    phase: str
    frame_idx: int
    kind: str                     # "fired" | "cleared"
    ts: float


class AlertManager:
    """Deduplicates alerts per swimmer and fans them out to sinks."""

    def __init__(self, config: AlertConfig = DEFAULT_ALERT,
                 webhook_url: Optional[str] = None,
                 logger=None) -> None:
        import os

        self.cfg = config
        self.webhook_url = webhook_url or os.environ.get("LIFESTREAM_WEBHOOK")
        self.logger = logger or _default_logger(config.log_file)
        self._was_active: Dict[int, bool] = {}
        self._last_fired: Dict[int, float] = {}
        self.events = []                      # in-memory record (tests/demo)

    def notify(self, swimmer_id: int, state: RiskState,
               frame_idx: int, source: str = "camera") -> None:
        previously = self._was_active.get(swimmer_id, False)

        if state.alert_active and not previously:
            last = self._last_fired.get(swimmer_id, 0.0)
            if time.time() - last < self.cfg.cooldown_s:
                self._was_active[swimmer_id] = True   # suppress but remember
                return
            self._last_fired[swimmer_id] = time.time()
            self._emit(AlertEvent(swimmer_id, state.score, state.phase,
                                  frame_idx, "fired", time.time()), source)
        elif not state.alert_active and previously:
            self._emit(AlertEvent(swimmer_id, state.score, state.phase,
                                  frame_idx, "cleared", time.time()), source)
        self._was_active[swimmer_id] = state.alert_active

    def _emit(self, e: AlertEvent, source: str) -> None:
        self.events.append(e)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.ts))
        self.logger.info(
            f"[{stamp}] ALERT {e.kind.upper()} swimmer=#{e.swimmer_id} "
            f"phase={e.phase} score={e.score:.2f} frame={e.frame_idx} "
            f"source={source}"
        )
        if self.webhook_url:
            self._webhook(e, source)

    def _webhook(self, e: AlertEvent, source: str) -> None:
        if requests is None:
            return
        emoji = "\U0001F6A8" if e.kind == "fired" else "\u2705"
        text = (f"{emoji} LifeStream AI — {self.cfg.event_label} (person "
                f"#{e.swimmer_id}, {e.phase}, score {e.score:.2f}, "
                f"{source}). Immediate attention required!")
        try:
            requests.post(self.webhook_url, json={"text": text}, timeout=2.5)
        except Exception:  # noqa: BLE001 — never let alerting kill the loop
            pass


def _default_logger(path: str):
    import logging

    lg = logging.getLogger("lifestream.alerts")
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(fh)
    return lg


def event_to_dict(e: AlertEvent) -> dict:
    return {
        "kind": e.kind, "swimmer_id": e.swimmer_id, "phase": e.phase,
        "score": round(e.score, 3), "frame": e.frame_idx, "ts": e.ts,
    }


def events_to_jsonl(events) -> str:
    return "\n".join(json.dumps(event_to_dict(e)) for e in events)
