"""The AnXin detection cascade for one video stream (T0 -> T1 -> T2).

    T0 pose (every frame) -> T1 skeleton state machine (every frame)
      -> on suspicion: T2 VLM on up to 3 keyframes (t0, +0.5 s, +1.0 s)

T3 (voice check-in, alerting, long-lie timer) lives in `guardian.py`.

Frame retention (what this module holds, verified by tests/test_privacy.py):
  * the current frame, only while it is being processed;
  * the most recent frame (one), so a clip that ends mid-verification can still be
    checked; it is overwritten by the next frame;
  * skeleton features: <= 1 s (fall window) and <= 3 s (standing-height baseline).
`FrameRecord.frame` is None unless the caller asks for frames (display/recording only).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .backends.vlm import verify_fall
from .fall import SUSPECT, FallConfig, FallStateMachine, Tracker, features

VERIFY_OFFSETS_S = (0.0, 0.5, 1.0)


@dataclass
class FrameRecord:
    i: int
    t: float
    frame: np.ndarray | None          # only set when Cascade(keep_frames=True), for display
    kps: np.ndarray | None
    feats: dict | None
    state: str
    pose_ms: float
    vlm: dict | None = None           # set on frames where the VLM was consulted
    verdict: str | None = None        # "CONFIRMED" | "DISMISSED" when decided on this frame
    end_of_stream: bool = False


@dataclass
class Cascade:
    pose: object
    vlm: object
    cfg: FallConfig = field(default_factory=FallConfig)
    keep_frames: bool = False
    on_vlm_start: Callable[[FrameRecord], None] | None = None  # UI hook before the (blocking) VLM call

    def _verify(self, rec: FrameRecord, frame: np.ndarray):
        if self.on_vlm_start:
            self.on_vlm_start(rec)
        t1 = time.perf_counter()
        rec.vlm = verify_fall(self.vlm, frame)
        rec.vlm["ms"] = 1000 * (time.perf_counter() - t1)

    def run(self, frames):
        """frames: iterable of (i, t, frame_bgr). Yields one FrameRecord per frame (plus one
        final end_of_stream record if a verification was still pending when the stream ended)."""
        sm, tr = FallStateMachine(self.cfg), Tracker()
        self.t1_wall = None
        pending: list[int] = []
        last_frame, last_rec = None, None
        for i, t, frame in frames:
            t0 = time.perf_counter()
            people = self.pose.infer(frame)
            pose_ms = 1000 * (time.perf_counter() - t0)
            H, W = frame.shape[:2]
            k = tr.select(people, W)
            f = features(k, H)
            state, fired = sm.update(f)
            rec = FrameRecord(i, t, frame if self.keep_frames else None, k, f, state, pose_ms)
            if fired:
                self.t1_wall = time.monotonic()  # wall time of the T1 suspicion (for latency)
                pending = [i + int(o * self.cfg.fps) for o in VERIFY_OFFSETS_S]
            if pending and i == pending[0]:
                pending.pop(0)
                self._verify(rec, frame)
                if rec.vlm["is_fall"]:
                    rec.verdict, pending = "CONFIRMED", []
                elif not pending:
                    rec.verdict = "DISMISSED"
            last_frame, last_rec = frame, rec
            del frame
            yield rec
        if pending and last_rec is not None and last_rec.vlm is None and last_rec.state == SUSPECT:
            # Stream ended before every keyframe was checked: verify the last frame we have.
            end = FrameRecord(last_rec.i, last_rec.t, last_frame if self.keep_frames else None, last_rec.kps,
                              last_rec.feats, last_rec.state, 0.0, end_of_stream=True)
            self._verify(end, last_frame)
            end.verdict = "CONFIRMED" if end.vlm["is_fall"] else "DISMISSED"
            yield end
