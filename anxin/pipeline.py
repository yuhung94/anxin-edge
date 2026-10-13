"""The AnXin cascade, end to end, for one video stream.

    T0 pose (every frame) -> T1 skeleton state machine (every frame)
      -> on suspicion: T2 VLM on up to 3 keyframes (t0, +0.5 s, +1.0 s)
        -> on confirmation: T3 voice check-in -> caregiver alert (text only)

Frames are processed and discarded; a confirmed event keeps one keyframe in RAM for
the VLM and nothing else. Only the skeleton and the alert text are ever emitted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backends.vlm import verify_fall
from .fall import SUSPECT, FallConfig, FallStateMachine, Tracker, features

VERIFY_OFFSETS_S = (0.0, 0.5, 1.0)


@dataclass
class FrameRecord:
    i: int
    t: float
    frame: np.ndarray
    kps: np.ndarray | None
    feats: dict | None
    state: str
    pose_ms: float
    vlm: dict | None = None          # set on frames where the VLM was consulted
    verdict: str | None = None       # "CONFIRMED" | "DISMISSED" when decided on this frame


@dataclass
class Cascade:
    pose: object
    vlm: object
    cfg: FallConfig = field(default_factory=FallConfig)

    def run(self, frames):
        """frames: iterable of (i, t, frame_bgr). Yields FrameRecord per frame."""
        sm, tr = FallStateMachine(self.cfg), Tracker()
        pending: list[int] = []
        votes: list[bool] = []
        buffered = []
        for i, t, frame in frames:
            t0 = time.perf_counter()
            people = self.pose.infer(frame)
            pose_ms = 1000 * (time.perf_counter() - t0)
            H, W = frame.shape[:2]
            k = tr.select(people, W)
            f = features(k, H)
            state, fired = sm.update(f)
            rec = FrameRecord(i, t, frame, k, f, state, pose_ms)
            if fired:
                pending = [i + int(o * self.cfg.fps) for o in VERIFY_OFFSETS_S]
                votes = []
            if pending and i == pending[0]:
                pending.pop(0)
                t1 = time.perf_counter()
                rec.vlm = verify_fall(self.vlm, frame)
                rec.vlm["ms"] = 1000 * (time.perf_counter() - t1)
                votes.append(rec.vlm["is_fall"])
                if rec.vlm["is_fall"]:
                    rec.verdict, pending = "CONFIRMED", []
                elif not pending:
                    rec.verdict = "DISMISSED"
            buffered.append(rec)
            yield rec
        if pending and buffered and buffered[-1].vlm is None and buffered[-1].state == SUSPECT:
            # Clip ended before every keyframe was checked: verify the last frame we have.
            # (Callers that collect records into a list see this update.)
            last = buffered[-1]
            last.vlm = verify_fall(self.vlm, last.frame)
            last.vlm["ms"] = 0.0
            last.verdict = "CONFIRMED" if last.vlm["is_fall"] else "DISMISSED"
