"""Tier-3: voice check-in, intent understanding and escalation.

After the VLM confirms a fall, AnXin speaks to the person, listens with Whisper and
lets a 1.5B LLM decide what to do. Safety rules are deterministic and wrap the LLM:
  * a distress keyword ALWAYS escalates (the LLM may escalate, never de-escalate it);
  * silence escalates;
  * "I'm OK" is accepted but the person must be back up within LONG_LIE_S, otherwise
    a "long-lie" alert is sent (people who cannot get up are the highest-risk group).
Only text leaves the device: the caregiver gets an LLM-written message, never video.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
LONG_LIE_S = 120

CHECKIN_PROMPT = "Hello, this is AnXin. It looks like you may have fallen. Are you okay? Please tell me if you need help."

DISTRESS = re.compile(r"\b(help|hurt|hurts|pain|can'?t (get|stand) up|cannot (get|stand) up|ambulance|bleeding|dizzy)\b|救命|好痛|起不來|幫忙",
                      re.I)

INTENT_SYSTEM = ("You classify what an elderly person said right after a possible fall. "
                 "Reply with exactly one word:\n"
                 "OK = they clearly say they are fine / not hurt\n"
                 "HELP = they are hurt, in pain, cannot get up, or ask for help\n"
                 "UNCLEAR = anything else, including confused or off-topic answers\n"
                 "Examples:\n"
                 "\"I'm fine, just tripped.\" -> OK\n"
                 '"My arm hurts." -> HELP\n'
                 '"What time is dinner?" -> UNCLEAR\n'
                 '"Who are you?" -> UNCLEAR')

REPORT_SYSTEM = (
    "You write a short text message to a family member about their elderly relative who lives alone. "
    "Report what happened using ONLY the facts in the JSON, in 2 or 3 complete sentences, in a calm tone. "
    "Mention the time, the room, and what the person said. Do not give orders. Do not write a heading.\n"
    "Example input: {\"time\": \"09:05\", \"room\": \"kitchen\", \"observation\": \"fallen and lying on the floor\", "
    "\"person_said\": \"I'm fine, I just tripped.\", \"anxin_next_step\": \"keeps watching; sends an urgent alert if still "
    "on the floor after 2 minutes\"}\n"
    "Example output: At 09:05 AnXin detected a fall in the kitchen. Your relative answered \"I'm fine, I just tripped.\" "
    "AnXin keeps watching and will send an urgent alert if they are still on the floor after 2 minutes.")

NEXT_STEP = {
    "ALERT_URGENT": "please call or go to them now; the AnXin speaker stays in contact with them",
    "ALERT_WATCH": "keeps watching; sends an urgent alert if still on the floor after 2 minutes",
}


def speak(text: str, out_wav: str | Path, voice: str = "Samantha", rate: int = 175) -> Path:
    """Text-to-speech to a 16 kHz mono WAV (macOS `say`; espeak-ng on Linux / Raspberry Pi)."""
    out_wav = Path(out_wav)
    if shutil.which("say"):
        with tempfile.TemporaryDirectory() as td:
            aiff = Path(td) / "t.aiff"
            subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ar", str(SR), "-ac", "1", str(out_wav)], check=True)
    else:
        subprocess.run(["espeak-ng", "-w", str(out_wav), text], check=True)
    return out_wav


def load_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1, "expect 16 kHz mono"
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (a.astype(np.float32) / 32768.0).copy()


@dataclass
class CheckinResult:
    transcript: str
    intent: str          # OK | HELP | NO_RESPONSE | UNCLEAR
    decided_by: str      # keyword | llm | silence
    action: str          # ALERT_URGENT | ALERT_WATCH | CONTINUE_MONITORING


def classify(transcript: str, llm) -> tuple[str, str]:
    if len(re.sub(r"[^\w]", "", transcript)) < 2:
        return "NO_RESPONSE", "silence"
    if DISTRESS.search(transcript):
        return "HELP", "keyword"
    out = llm.chat(INTENT_SYSTEM, f'The person said: "{transcript}"', max_new_tokens=4).upper()
    for k in ("HELP", "OK", "UNCLEAR"):
        if k in out:
            return k, "llm"
    return "UNCLEAR", "llm"


def decide(intent: str) -> str:
    return {"HELP": "ALERT_URGENT", "NO_RESPONSE": "ALERT_URGENT", "UNCLEAR": "ALERT_URGENT",
            "OK": "ALERT_WATCH"}[intent]


def run_checkin(reply_audio: np.ndarray | None, asr, llm) -> CheckinResult:
    transcript = asr.transcribe(reply_audio) if reply_audio is not None and len(reply_audio) else ""
    intent, by = classify(transcript, llm)
    return CheckinResult(transcript, intent, by, decide(intent))


URGENCY_LABEL = {"ALERT_URGENT": "URGENT", "ALERT_WATCH": "FYI"}


def caregiver_message(event: dict, llm) -> str:
    """event: time, room, observation, person_said, action.
    Urgency and next step come from the deterministic policy; the LLM only phrases the message."""
    facts = {k: event[k] for k in ("time", "room", "observation", "person_said") if k in event}
    facts["anxin_next_step"] = NEXT_STEP.get(event.get("action"), NEXT_STEP["ALERT_URGENT"])
    body = llm.chat(REPORT_SYSTEM, json.dumps(facts, ensure_ascii=False), max_new_tokens=120)
    body = re.sub(r"^\W*(urgent|urgency[^:]*|high|low|fyi)\W*[:\-–]\s*", "", body, flags=re.I).strip()
    return f"{URGENCY_LABEL.get(event.get('action'), 'ALERT')}: {body}"
