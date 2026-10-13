"""Run Tier-0 pose on every URFD clip once and cache keypoints (+ per-frame latency)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anxin.backends.pose import make_pose  # noqa: E402
from anxin.video import iter_frames  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/urfd")
ap.add_argument("--out", default="data/pose_cache")
ap.add_argument("--backend", default="cpu", choices=["cpu", "hailo"])
args = ap.parse_args()

pose = make_pose(args.backend)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for vid in sorted(Path(args.data).glob("*.mp4")):
    if (out / f"{vid.stem}.npz").exists():
        continue
    kps, lat, shape = [], [], None
    for _, _, frame in iter_frames(vid):
        shape = frame.shape[:2]
        t0 = time.perf_counter()
        people = pose.infer(frame)
        lat.append(time.perf_counter() - t0)
        slot = np.full((3, 17, 3), np.nan, np.float32)  # keep up to 3 people; tracking happens later
        for j, pp in enumerate(people[:3]):
            slot[j] = pp
        kps.append(slot)
    np.savez_compressed(out / f"{vid.stem}.npz", kps=np.stack(kps), lat=np.array(lat), shape=np.array(shape))
    print(f"{vid.stem}: {len(kps)} frames, {np.median(lat)*1000:.1f} ms/frame", flush=True)
