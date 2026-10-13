"""End-to-end evaluation on the UR Fall Detection Dataset (30 falls / 40 ADL clips).

Tier 1 (skeleton state machine) runs on cached poses; every Tier-1 suspicion is sent
to the Tier-2 VLM with up to 3 keyframes (t0, +0.5 s, +1.0 s, clamped to clip end).
A suspicion is confirmed when ANY keyframe is judged "fallen on the floor".

Writes outputs/eval.json and outputs/eval_chart.png.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin.backends.vlm import make_vlm, verify_fall  # noqa: E402
from anxin.fall import FallConfig, run_sequence  # noqa: E402
from anxin.video import iter_frames  # noqa: E402

OFFSETS_S = (0.0, 0.5, 1.0)


def metrics(pred: dict[str, bool]) -> dict:
    tp = sum(v for k, v in pred.items() if k.startswith("fall"))
    fp = sum(v for k, v in pred.items() if k.startswith("adl"))
    nf = sum(k.startswith("fall") for k in pred)
    na = sum(k.startswith("adl") for k in pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / nf
    return {"TP": tp, "FN": nf - tp, "FP": fp, "TN": na - fp, "precision": round(prec, 3), "recall": round(rec, 3),
            "specificity": round((na - fp) / na, 3), "f1": round(2 * prec * rec / (prec + rec) if prec + rec else 0, 3)}


def main(backend: str = "cpu"):
    cfg = FallConfig()
    vlm = make_vlm(backend)
    clips, t1, t2, vlm_lat, pose_lat = {}, {}, {}, [], []
    for npz in sorted((ROOT / "data/pose_cache").glob("*.npz")):
        name = npz.stem
        d = np.load(npz)
        H, W = d["shape"]
        pose_lat += list(d["lat"])
        first, states, _ = run_sequence(d["kps"], H, W, cfg)
        t1[name] = first is not None
        rec = {"tier1_fire_frame": first, "n_frames": len(states), "vlm": []}
        confirmed = False
        if first is not None:
            want = sorted({min(first + int(o * cfg.fps), len(states) - 1) for o in OFFSETS_S})
            for i, _, frame in iter_frames(ROOT / "data/urfd" / f"{name}.mp4"):
                if i in want:
                    t0 = time.perf_counter()
                    r = verify_fall(vlm, frame)
                    vlm_lat.append(time.perf_counter() - t0)
                    rec["vlm"].append({"frame": i, **r})
                    confirmed |= r["is_fall"]
                if i >= want[-1]:
                    break
        t2[name] = confirmed
        rec["confirmed"] = confirmed
        clips[name] = rec
        print(name, "T1" if t1[name] else "--", "T2" if confirmed else "--",
              [v["choice"] for v in rec["vlm"]], flush=True)

    res = {
        "dataset": "UR Fall Detection Dataset, cam0 RGB (30 fall / 40 ADL)",
        "backend": {"pose": "yolov8s-pose", "vlm": vlm.name},
        "tier1_skeleton_only": metrics(t1),
        "tier1_plus_vlm": metrics(t2),
        "latency_ms": {"pose_median": round(1000 * float(np.median(pose_lat)), 1),
                       "vlm_median": round(1000 * float(np.median(vlm_lat)), 1) if vlm_lat else None,
                       "vlm_calls": len(vlm_lat)},
        "clips": clips,
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs/eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("tier1_skeleton_only", "tier1_plus_vlm", "latency_ms")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
