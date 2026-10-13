"""Wall-clock screen recorder for the dashboard.

Each pushed image is shown until the next push, so blocking steps (VLM, speech, LLM)
appear in the recording for exactly as long as they took. Audio played by AudioIO is
mixed in at the wall-clock time it was played. Output: H.264 + AAC MP4.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import cv2


class Recorder:
    def __init__(self, path: str | Path, fps: int = 30, size=(1280, 720)):
        self.path, self.fps, self.size = Path(path), fps, size
        self.tmp = Path(tempfile.mkdtemp()) / "v.mp4"
        self.vw = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.start = None
        self.n = 0
        self.last = None

    def push(self, img):
        now = time.monotonic()
        if self.start is None:
            self.start = now
        target = int((now - self.start) * self.fps)
        while self.last is not None and self.n < target:  # hold previous screen for elapsed time
            self.vw.write(self.last)
            self.n += 1
        self.vw.write(img)
        self.n += 1
        self.last = img

    def close(self, audio_events=(), tail_s: float = 1.5):
        for _ in range(int(tail_s * self.fps)):
            self.vw.write(self.last)
            self.n += 1
        self.vw.release()
        dur = self.n / self.fps
        ev = [(max(0.0, t - self.start), w) for t, w in audio_events]
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(self.tmp)]
        if ev:
            for _, w in ev:
                cmd += ["-i", str(w)]
            parts = ";".join(f"[{k + 1}:a]adelay={int(t * 1000)}|{int(t * 1000)}[a{k}]" for k, (t, _) in enumerate(ev))
            mix = "".join(f"[a{k}]" for k in range(len(ev)))
            cmd += ["-filter_complex", f"{parts};{mix}amix=inputs={len(ev)}:normalize=0,apad,atrim=0:{dur:.3f}[aout]",
                    "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-ar", "44100", "-ac", "2"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(self.path)]
        subprocess.run(cmd, check=True)
        return dur
