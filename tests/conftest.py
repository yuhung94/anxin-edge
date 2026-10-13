import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def skeleton(cx, top, upright=True, w=320):
    """Synthetic COCO-17 skeleton. upright: vertical body from `top` (px) down 150 px; else horizontal on the floor."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    if upright:
        ys = np.linspace(top, top + 150, 17)
        k[:, 0], k[:, 1] = cx + np.tile([-8, 8], 9)[:17], ys
        k[[5, 6], 1], k[[11, 12], 1] = top + 30, top + 80
    else:
        xs = np.linspace(cx - 75, cx + 75, 17)
        k[:, 0], k[:, 1] = xs, 215 + np.tile([-6, 6], 9)[:17]
        k[[5, 6], 0], k[[11, 12], 0] = cx - 45, cx + 5
        k[[5, 6], 1], k[[11, 12], 1] = 212, 218
    return k


class ScriptedPose:
    """Replays a list of skeletons (one per frame) instead of running a network."""

    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def infer(self, frame):
        k = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return [k] if k is not None else []


class FixedVLM:
    name = "stub"

    def __init__(self, answer="E) the person is on the floor"):
        self.answer, self.calls = answer, 0

    def ask(self, frame, *a, **k):
        self.calls += 1
        return self.answer


def fall_sequence(stand_s=1.0, down_s=5.0, fps=30, getup_s=0.0):
    seq = [skeleton(160, 40) for _ in range(int(stand_s * fps))]
    seq += [skeleton(160, 40 + 25 * j) for j in range(1, 5)]  # fast drop over ~4 frames
    seq += [skeleton(160, 0, upright=False) for _ in range(int(down_s * fps))]
    seq += [skeleton(160, 40) for _ in range(int(getup_s * fps))]
    return seq


@pytest.fixture
def frames_for():
    def make(n, h=240, w=320):
        for i in range(n):
            yield i, i / 30.0, np.zeros((h, w, 3), np.uint8)
    return make
