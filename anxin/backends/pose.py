"""Tier-0 pose estimation backends.

Both backends run the *same network family* (YOLOv8s-pose, COCO 17 keypoints),
so the rest of the pipeline is identical on a laptop and on the UGen300:

    cpu   -> Ultralytics yolov8s-pose.pt on CPU / Apple MPS  (Stage I prototype)
    hailo -> yolov8s_pose.hef on UGen300 (Hailo-10H) via HailoRT InferModel

Output contract: list of people, each a float32 array (17, 3) = (x, y, conf)
in original-frame pixel coordinates, sorted by detection score.
"""
from __future__ import annotations

import numpy as np


class UltralyticsPose:
    name = "cpu:yolov8s-pose"

    def __init__(self, weights: str = "yolov8s-pose.pt", device: str | None = None, conf: float = 0.35):
        from ultralytics import YOLO
        import torch

        self.model = YOLO(weights)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.conf = conf

    def infer(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        r = self.model.predict(frame_bgr, imgsz=640, conf=self.conf, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(kxy.shape[:2], np.float32)
        order = r.boxes.conf.cpu().numpy().argsort()[::-1]
        return [np.concatenate([kxy[i], kc[i][:, None]], axis=1).astype(np.float32) for i in order]


class HailoPose:
    name = "hailo:yolov8s_pose.hef"

    def __init__(self, hef_path: str = "models/yolov8s_pose.hef", conf: float = 0.5):
        from .hailo_vendor.hailo_pose import PoseEstimator

        self.est = PoseEstimator(hef_path, conf_threshold=conf)

    def infer(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(p, dtype=np.float32) for p in self.est.infer_multi(frame_bgr)]


def make_pose(backend: str = "cpu", **kw):
    return HailoPose(**kw) if backend == "hailo" else UltralyticsPose(**kw)
