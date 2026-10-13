"""Video helpers. URFD cam0 videos are 640x240 = [depth | RGB]; we only use the RGB half."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def iter_frames(path: str | Path, rgb_half: bool | None = None):
    """Yield (index, t_seconds, frame_bgr). Auto-crops URFD side-by-side videos."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rgb_half is None:
            rgb_half = frame.shape[1] == 2 * frame.shape[0] + 160  # 640x240 URFD layout
        if rgb_half:
            frame = frame[:, frame.shape[1] // 2 :]
        yield i, i / fps, np.ascontiguousarray(frame)
        i += 1
    cap.release()


def video_fps(path: str | Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps
