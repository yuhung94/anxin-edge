"""Speaker / microphone I/O for the voice check-in.

  say(text)          : TTS -> plays on the speaker (blocking) -> returns its duration
  listen(seconds)    : mic recording (sounddevice) OR, in dataset-replay mode, a reply
                       WAV file stands in for the microphone (the dataset has no audio)
Audio is held in RAM only and returned to the caller; nothing is written except the
TTS prompt cache. `events` logs (wall_time, wav) so a recording can include the sound.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .checkin import SR, load_wav, speak


class AudioIO:
    def __init__(self, voice: str = "Karen", mute: bool = False, reply_wav: str | None = None, use_mic: bool = False):
        self.voice, self.mute, self.reply_wav, self.use_mic = voice, mute, reply_wav, use_mic
        self.tmp = Path(tempfile.mkdtemp(prefix="anxin_tts_"))
        self.events: list[tuple[float, Path]] = []
        self._n = 0
        self._cache: dict[str, Path] = {}

    def _play(self, wav: Path, dur: float):
        self.events.append((time.monotonic(), wav))
        if not self.mute and shutil.which("afplay"):
            subprocess.run(["afplay", str(wav)], check=False)
        else:
            time.sleep(dur)  # keep real timing even when muted

    def prepare(self, text: str) -> Path:
        """Synthesise a fixed prompt once at start-up so speaking it later has no TTS delay."""
        if text not in self._cache:
            self._n += 1
            self._cache[text] = speak(text, self.tmp / f"say{self._n}.wav", self.voice)
        return self._cache[text]

    def say(self, text: str) -> float:
        wav = self.prepare(text)
        dur = len(load_wav(wav)) / SR
        self._play(wav, dur)
        return dur

    def listen(self, seconds: float = 6.0) -> np.ndarray:
        if self.reply_wav:  # dataset replay: the reply file plays as if spoken in the room
            a = load_wav(self.reply_wav)
            self._play(Path(self.reply_wav), len(a) / SR)
            return a
        if self.use_mic:
            import sounddevice as sd

            rec = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            return rec[:, 0].copy()
        time.sleep(seconds)  # no reply source: a silent listening window
        return np.zeros(0, np.float32)
