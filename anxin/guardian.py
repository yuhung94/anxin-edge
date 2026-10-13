"""Guardian: the closed loop around the detection cascade (T3 + escalation).

MONITORING --(VLM confirms fall)--> CHECK-IN: speak, listen, Whisper, intent
   intent HELP / NO_RESPONSE / UNCLEAR --> send URGENT ---------------------> ALERTED
   intent OK                           --> send FYI, start long-lie timer ---> WATCHING
WATCHING --(person upright again)--> send FYI "back on their feet" --> MONITORING
WATCHING --(still down after long_lie_s; occlusion counts as down)--> URGENT --> ALERTED
ALERTED  --(person upright again)--> MONITORING

Timers use the stream clock `rec.t` (wall clock for a live camera). Every alert carries
measured wall-clock timings from the T1 suspicion to the receiver's delivery ack.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Callable

from .checkin import CHECKIN_PROMPT, caregiver_message, run_checkin, urgent_template
from .fall import SUSPECT, UPRIGHT

MONITORING, CHECKIN, WATCHING, ALERTED = "MONITORING", "CHECK-IN", "WATCHING", "ALERTED"


@dataclass
class Guardian:
    cascade: object
    asr: object
    llm: object
    notifier: object
    audio: object
    room: str = "living room"
    long_lie_s: float = 120.0
    listen_s: float = 6.0
    ui: Callable[[dict], None] | None = None   # redraw hook, called before blocking steps
    phase: str = MONITORING
    alerts: list = field(default_factory=list)
    status: dict = field(default_factory=dict)

    def warmup(self):
        """Load and exercise every model once at start-up (first calls are slow on any backend)."""
        import numpy as np

        from .backends.vlm import verify_fall
        verify_fall(self.cascade.vlm, np.zeros((240, 320, 3), np.uint8))
        self.cascade.pose.infer(np.zeros((240, 320, 3), np.uint8))
        self.asr.transcribe(np.zeros(16000, np.float32))
        self.llm.chat("Reply OK.", "Hi", max_new_tokens=4)
        caregiver_message({"time": "00:00", "room": self.room, "observation": "test", "person_said": "test",
                           "action": "ALERT_WATCH"}, self.llm)
        if hasattr(self.audio, "prepare"):
            self.audio.prepare(CHECKIN_PROMPT)

    def __post_init__(self):
        self.cascade.on_vlm_start = lambda rec: self._ui(rec, step="vlm")
        self._t1_wall = None
        self._watch_start = None
        self._last_rec = None

    # ------------------------------------------------------------------ helpers
    def _ui(self, rec, **kw):
        self.status.update(kw)
        if self.ui:
            self.ui({"rec": rec, "phase": self.phase, "alerts": self.alerts, **self.status})

    def _span(self) -> str:
        s = self.long_lie_s
        return f"{s / 60:g} minute{'s' * (s != 60)}" if s >= 60 else f"{s:g} second{'s' * (s != 1)}"

    def _send(self, rec, level: str, kind: str, message: str, timings: dict) -> dict:
        alert = {"level": level, "kind": kind, "room": self.room, "message": message,
                 "stream_t": round(rec.t, 2), "sent_at": time.time()}
        t0 = time.monotonic()
        receipt = self.notifier.send(alert)
        timings["deliver_s"] = round(time.monotonic() - t0, 3)
        if self._t1_wall is not None:
            timings["t1_to_delivered_s"] = round(time.monotonic() - self._t1_wall, 2)
        alert.update(receipt=receipt, timings=dict(timings))
        self.alerts.append(alert)
        return alert

    # ------------------------------------------------------------------ check-in
    def _checkin(self, rec):
        self.phase = CHECKIN
        timings = {"t1_to_vlm_confirmed_s": round(time.monotonic() - self._t1_wall, 2)}
        self._ui(rec, step="ask", transcript=None, intent=None)
        t = time.monotonic()
        self.audio.say(CHECKIN_PROMPT)
        timings["ask_s"] = round(time.monotonic() - t, 2)
        self._ui(rec, step="listen")
        t = time.monotonic()
        reply = self.audio.listen(self.listen_s)
        timings["listen_s"] = round(time.monotonic() - t, 2)
        self._ui(rec, step="understand")
        t = time.monotonic()
        ck = run_checkin(reply, self.asr, self.llm)
        timings["asr_intent_s"] = round(time.monotonic() - t, 2)
        self._ui(rec, step="send" if ck.action == "ALERT_URGENT" else "write",
                 transcript=ck.transcript, intent=ck.intent, decided_by=ck.decided_by)
        clock = dt.datetime.now().strftime("%H:%M")
        facts = {"time": clock, "room": self.room, "observation": rec.vlm["label"],
                 "person_said": ck.transcript or "(no reply)", "action": ck.action}
        if ck.action == "ALERT_URGENT":
            # 1) deterministic alert first, 2) LLM-written details as a follow-up
            self._send(rec, "URGENT", f"fall_{ck.intent.lower()}", urgent_template(clock, self.room, ck.transcript), timings)
            self.phase = ALERTED
            self._ui(rec, step="write")
            t = time.monotonic()
            msg = caregiver_message(facts, self.llm, label="Details")
            self._send(rec, "INFO", "details", msg, {"llm_message_s": round(time.monotonic() - t, 2)})
        else:
            t = time.monotonic()
            msg = caregiver_message(facts, self.llm, next_step="keeps watching; sends an urgent alert if they are still on "
                                                              f"the floor after {self._span()}")
            timings["llm_message_s"] = round(time.monotonic() - t, 2)
            self._ui(rec, step="send")
            self._send(rec, "FYI", f"fall_{ck.intent.lower()}", msg, timings)
            self.phase, self._watch_start = WATCHING, rec.t
        self._ui(rec, step="sent")

    # ------------------------------------------------------------------ main loop
    def run(self, frames):
        """Yields (FrameRecord, status) for every processed frame."""
        prev_state = UPRIGHT
        for rec in self.cascade.run(frames):
            if rec.state == SUSPECT and prev_state != SUSPECT and not rec.end_of_stream:
                self._t1_wall = getattr(self.cascade, "t1_wall", None) or time.monotonic()
                if self.phase == MONITORING:  # new incident: clear the previous one from the display
                    for k in ("step", "last_vlm", "transcript", "intent", "decided_by"):
                        self.status.pop(k, None)
            if rec.vlm:
                self.status["last_vlm"] = rec.vlm
            prev_state = rec.state
            if rec.verdict == "CONFIRMED" and self.phase == MONITORING:
                self._checkin(rec)
            elif rec.verdict == "DISMISSED" and self.phase == MONITORING:
                self.status.update(step="dismissed")
            elif self.phase == WATCHING:
                if rec.state == UPRIGHT:
                    self._send(rec, "FYI", "recovered", "Your relative is back on their feet. AnXin keeps watching.", {})
                    self.phase = MONITORING
                elif rec.t - self._watch_start >= self.long_lie_s:
                    self._send(rec, "URGENT", "long_lie",
                               f"URGENT: Your relative said they were okay, but they have now been on the floor in the "
                               f"{self.room} for more than {self._span()} and may not be able to get up. "
                               f"Please call or go to them now.", {})
                    self.phase = ALERTED
            elif self.phase == ALERTED and rec.state == UPRIGHT:
                self.phase = MONITORING
            if self.phase == WATCHING:
                self.status["watch_left_s"] = max(0.0, self.long_lie_s - (rec.t - self._watch_start))
            self._last_rec = rec
            yield rec, {"phase": self.phase, "alerts": self.alerts, **self.status}
