"""Run the AnXin cascade on a video file or webcam.

    python -m anxin --video data/urfd/fall-01-cam0.mp4            # Stage I (laptop)
    python -m anxin --video 0 --backend hailo --show              # Stage II (UGen300, live camera)
"""
import argparse

import cv2

from . import dashboard as dash
from .backends.pose import make_pose
from .backends.vlm import make_vlm
from .pipeline import Cascade
from .video import iter_frames


def webcam(index: int):
    cap = cv2.VideoCapture(index)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, i / fps, frame
        i += 1


def main():
    ap = argparse.ArgumentParser(description="AnXin Edge fall guardian")
    ap.add_argument("--video", default="0", help="video file, or camera index")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "hailo"])
    ap.add_argument("--show", action="store_true", help="show the dashboard window")
    a = ap.parse_args()
    frames = webcam(int(a.video)) if a.video.isdigit() else iter_frames(a.video)
    cascade = Cascade(make_pose(a.backend), make_vlm(a.backend))
    banner = {"text": "Monitoring…  only the skeleton on the right is kept."}
    for r in cascade.run(frames):
        if r.vlm:
            print(f"[{r.t:6.2f}s] VLM: {r.vlm['choice']}) {r.vlm['label']}" + (f" -> {r.verdict}" if r.verdict else ""))
            if r.verdict == "CONFIRMED":
                banner = {"text": f"VLM: “{r.vlm['label']}” › fall CONFIRMED › voice check-in + family alert", "color": dash.RED}
            elif r.verdict == "DISMISSED":
                banner = {"text": f"VLM: “{r.vlm['label']}” › false alarm DISMISSED", "color": dash.GREEN}
        if a.show:
            img = dash.render({"frame": r.frame, "kps": r.kps, "feats": r.feats, "t": r.t, "state": r.state, "banner": banner})
            cv2.imshow("AnXin Edge  (Esc to quit)", img)
            # play at roughly real time; pause 3 s on a VLM decision so it can be read
            if cv2.waitKey(3000 if r.verdict else 15) == 27:
                break
    if a.show:
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
