"""Closed loop: check-in -> delivery -> long-lie timer / recovery."""
import numpy as np

from anxin.guardian import ALERTED, MONITORING, Guardian
from anxin.pipeline import Cascade
from conftest import FixedVLM, ScriptedPose, fall_sequence


class StubAudio:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)

    def listen(self, seconds):
        return np.ones(16000, np.float32)


class StubASR:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class StubLLM:
    def chat(self, system, user, max_new_tokens=0):
        return "OK" if max_new_tokens <= 4 else "At 10:00 AnXin detected a fall."


class StubNotifier:
    def __init__(self):
        self.sent = []

    def send(self, alert):
        self.sent.append(alert)
        return {"delivered": True, "attempts": 1, "rtt_ms": 1.0, "channel": "stub", "ack": {"id": len(self.sent)}}


def run(seq, reply, long_lie_s):
    n = StubNotifier()
    g = Guardian(Cascade(ScriptedPose(seq), FixedVLM()), StubASR(reply), StubLLM(), n, StubAudio(), long_lie_s=long_lie_s)
    list(g.run((i, i / 30.0, np.zeros((240, 320, 3), np.uint8)) for i in range(len(seq))))
    return g, n.sent


def test_help_reply_sends_urgent():
    g, sent = run(fall_sequence(down_s=3), "My hip hurts", 120)
    assert [a["level"] for a in sent] == ["URGENT", "INFO"]          # template first, LLM details second
    assert sent[0]["kind"] == "fall_help" and g.phase == ALERTED
    assert sent[0]["message"].startswith("URGENT: Possible fall") and "My hip hurts" in sent[0]["message"]
    assert sent[0]["timings"]["t1_to_delivered_s"] >= 0


def test_ok_reply_then_still_down_escalates_long_lie():
    g, sent = run(fall_sequence(down_s=6), "I'm fine, I just slipped", long_lie_s=2.0)
    assert [(a["level"], a["kind"]) for a in sent] == [("FYI", "fall_ok"), ("URGENT", "long_lie")]


def test_ok_reply_then_gets_up_sends_recovered_and_resets():
    g, sent = run(fall_sequence(down_s=2, getup_s=2), "I'm fine, I just slipped", long_lie_s=60)
    assert [(a["level"], a["kind"]) for a in sent] == [("FYI", "fall_ok"), ("FYI", "recovered")]
    assert g.phase == MONITORING


def test_no_reply_is_urgent():
    g, sent = run(fall_sequence(down_s=3), "", 120)
    assert sent[0]["level"] == "URGENT" and sent[0]["kind"] == "fall_no_response" and "did not answer" in sent[0]["message"]
