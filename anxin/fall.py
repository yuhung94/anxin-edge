"""Tier-1: skeleton-only fall state machine.

Runs on 17 COCO keypoints per frame -- no pixels. Cheap enough for a Raspberry Pi
host CPU while the UGen300 does pose. A fall is *suspected* when the person goes
from upright to horizontal quickly AND stays low/horizontal for `dwell_s`.
Suspicions are then handed to Tier-2 (VLM) for verification.

Features (all scale-normalised by the person's standing height so they work at any
camera distance):
  torso_angle : angle of shoulder-mid -> hip-mid vector from vertical (deg)
  aspect      : skeleton bbox width / height
  hip_y       : hip-mid y position in frame-height units (0 top, 1 bottom)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
KP_CONF = 0.3

UPRIGHT, DESCENDING, DOWN, SUSPECT = "UPRIGHT", "DESCENDING", "DOWN", "SUSPECTED_FALL"


def features(kps: np.ndarray, frame_h: float) -> dict | None:
    """Return per-frame posture features, or None if the skeleton is unusable."""
    if kps is None or np.isnan(kps).all():
        return None
    ok = kps[:, 2] >= KP_CONF
    if ok.sum() < 5:
        return None
    pts = kps[ok, :2]
    w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
    out = {"aspect": float(w / max(h, 1.0)), "top_y": float(pts[:, 1].min() / frame_h)}
    sh = [kps[j, :2] for j in (L_SH, R_SH) if kps[j, 2] >= KP_CONF]
    hp = [kps[j, :2] for j in (L_HIP, R_HIP) if kps[j, 2] >= KP_CONF]
    if sh and hp:
        v = np.mean(hp, 0) - np.mean(sh, 0)
        out["torso_angle"] = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))
        out["hip_y"] = float(np.mean(hp, 0)[1] / frame_h)
    else:
        out["torso_angle"] = 90.0 if out["aspect"] > 1.2 else 0.0
        out["hip_y"] = float(pts[:, 1].mean() / frame_h)
    return out


def is_horizontal(f: dict, stand_top: float | None = None) -> bool:
    """Lying / collapsed posture. The head-drop cue catches falls toward the camera,
    where foreshortening keeps the torso looking vertical in the image."""
    if f["torso_angle"] > 55 or f["aspect"] > 1.3:
        return True
    return stand_top is not None and f["top_y"] - stand_top > 0.30


class Tracker:
    """Follow one person across frames (nearest skeleton centre); avoids jumping to
    a spurious detection such as a chair or a reflection."""

    def __init__(self, max_jump: float = 0.35, forget_frames: int = 30):
        self.prev = None
        self.max_jump = max_jump
        self.lost = 0
        self.forget_frames = forget_frames

    def select(self, people, frame_w: float):
        cands = [p for p in people if p is not None and not np.isnan(p).all() and (p[:, 2] >= KP_CONF).sum() >= 5]
        if not cands:
            self.lost += 1
            return None
        centres = [p[p[:, 2] >= KP_CONF, :2].mean(0) for p in cands]
        i = 0
        if self.prev is not None and self.lost < self.forget_frames:
            d = [np.linalg.norm(c - self.prev) / frame_w for c in centres]
            i = int(np.argmin(d))
            if d[i] > self.max_jump:  # nobody near the tracked person: treat as occluded
                self.lost += 1
                return None
        self.prev, self.lost = centres[i], 0
        return cands[i]


@dataclass
class FallConfig:
    fps: float = 30.0
    fast_window_s: float = 1.0   # upright -> horizontal must happen within this window
    min_drop: float = 0.12       # top-of-head must drop >= 12% of frame height in that window
    dwell_s: float = 0.5         # must stay horizontal this long (seconds) -> suspect
    recover_s: float = 1.0       # upright again this long -> reset


@dataclass
class FallStateMachine:
    cfg: FallConfig = field(default_factory=FallConfig)

    def __post_init__(self):
        n = int(self.cfg.fast_window_s * self.cfg.fps) + 1
        self.hist: deque = deque(maxlen=n)  # (upright?, top_y)
        self.stand_tops: deque = deque(maxlen=int(3 * self.cfg.fps))  # head height while upright
        self.state = UPRIGHT
        self.down_frames = 0
        self.up_frames = 0
        self.fired = False

    def update(self, f: dict | None) -> tuple[str, bool]:
        """Feed one frame's features. Returns (state, newly_suspected)."""
        if f is None:
            # Person lost. Occlusion behind furniture is common once someone is on the
            # floor, so time keeps counting toward the dwell while we are DOWN.
            if self.state == DOWN:
                self.down_frames += 1
                if self.down_frames >= self.cfg.dwell_s * self.cfg.fps and not self.fired:
                    self.state, self.fired = SUSPECT, True
                    return self.state, True
            return self.state, False
        stand_top = float(np.median(self.stand_tops)) if len(self.stand_tops) >= 5 else None
        horiz = is_horizontal(f, stand_top)
        if not horiz:
            self.stand_tops.append(f["top_y"])
        fast_drop = False
        if horiz and self.hist:
            was_upright = [ty for up, ty in self.hist if up]
            if was_upright and (f["top_y"] - min(was_upright)) >= self.cfg.min_drop:
                fast_drop = True
        self.hist.append((not horiz, f["top_y"]))

        fired = False
        if horiz:
            self.up_frames = 0
            if self.state == UPRIGHT and fast_drop:
                self.state, self.down_frames = DESCENDING, 0
            if self.state in (DESCENDING, DOWN):
                self.down_frames += 1
                self.state = DOWN
                if self.down_frames >= self.cfg.dwell_s * self.cfg.fps and not self.fired:
                    self.state, self.fired, fired = SUSPECT, True, True
        else:
            self.up_frames += 1
            need_s = self.cfg.recover_s if self.state == SUSPECT else 0.3  # tolerate keypoint flicker
            if self.state != UPRIGHT and self.up_frames >= need_s * self.cfg.fps:
                self.state, self.down_frames, self.fired = UPRIGHT, 0, False
        return self.state, fired


def run_sequence(people_seq: np.ndarray, frame_h: float, frame_w: float, cfg: FallConfig | None = None):
    """Run tracker + state machine over cached poses (T, P, 17, 3).
    Returns (first_fire_frame | None, states, tracked_kps)."""
    sm, tr = FallStateMachine(cfg or FallConfig()), Tracker()
    first, states, tracked = None, [], []
    for i, people in enumerate(people_seq):
        k = tr.select(list(people), frame_w)
        tracked.append(k)
        st, fired = sm.update(features(k, frame_h))
        states.append(st)
        if fired and first is None:
            first = i
    return first, states, tracked
