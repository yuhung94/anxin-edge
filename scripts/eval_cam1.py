"""Cross-view test: URFD cam1 (ceiling, top-down) falls, with rules and prompt UNCHANGED.
T1 recall on cached poses; T2 alone on the last frame of each clip (person on the floor)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin.backends.vlm import make_vlm, verify_fall
from anxin.fall import run_sequence
from anxin.video import iter_frames
vlm = make_vlm("cpu")
res = {}
for npz in sorted((ROOT / "data/pose_cache_cam1").glob("*.npz")):
    d = np.load(npz); H, W = d["shape"]
    first, _, _ = run_sequence(d["kps"], H, W)
    last = None
    for _, _, f in iter_frames(ROOT / "data/urfd_cam1" / f"{npz.stem}.mp4"):
        last = f
    v = verify_fall(vlm, last)
    detected = (~np.isnan(d["kps"][:, 0, 0, 0])).mean()
    res[npz.stem] = {"t1_fired": first is not None, "vlm_last_frame": v["choice"], "vlm_is_fall": v["is_fall"],
                     "person_detected_frac": round(float(detected), 2)}
    print(npz.stem, res[npz.stem], flush=True)
summ = {"n": len(res), "t1_recall": sum(r["t1_fired"] for r in res.values()) / len(res),
        "vlm_alone_recall_last_frame": sum(r["vlm_is_fall"] for r in res.values()) / len(res),
        "pose_detect_rate": round(float(np.mean([r["person_detected_frac"] for r in res.values()])), 2)}
print(summ)
(ROOT / "outputs/eval_cam1.json").write_text(json.dumps({"summary": summ, "clips": res}, indent=2))
