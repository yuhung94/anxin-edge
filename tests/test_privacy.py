"""The cascade must not retain raw frames (README / deck privacy claim)."""
import gc
import weakref

import numpy as np

from anxin.pipeline import Cascade
from conftest import FixedVLM, ScriptedPose, fall_sequence


class Frame(np.ndarray):  # ndarray subclass so we can hold weak references
    pass


def test_frames_are_not_retained_while_streaming():
    seq = fall_sequence(stand_s=1, down_s=3)
    refs, records = [], []

    def frames():
        for i in range(len(seq)):
            f = np.zeros((240, 320, 3), np.uint8).view(Frame)
            refs.append(weakref.ref(f))
            yield i, i / 30.0, f

    gen = Cascade(ScriptedPose(seq), FixedVLM()).run(frames())
    for rec in gen:
        records.append(rec)          # the caller keeps every record ...
        assert rec.frame is None     # ... but records carry no pixels
        gc.collect()
        alive = sum(r() is not None for r in refs)
        assert alive <= 2, f"{alive} frames alive"   # current + the single 'last frame'
    assert len(records) >= len(seq)


def test_keep_frames_is_opt_in_for_display():
    seq = fall_sequence(stand_s=0.5, down_s=1)
    recs = list(Cascade(ScriptedPose(seq), FixedVLM(), keep_frames=True).run(
        (i, i / 30, np.zeros((240, 320, 3), np.uint8)) for i in range(len(seq))))
    assert all(r.frame is not None for r in recs)
