"""Run AnXin Edge end to end: detect -> verify -> voice check-in -> deliver alert -> long-lie timer.

Start the receiver first (separate terminal):   python scripts/family_phone.py
Dataset replay (reply audio from a file, because the dataset has no audio):
    python -m anxin --video data/urfd/fall-01-cam0.mp4 --reply-wav outputs/audio/reply_help.wav --show
Live camera + microphone:
    python -m anxin --video 0 --mic --show
Stage II on UGen300:  add  --backend hailo
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from . import dashboard as dash
from .audio import AudioIO
from .backends.genai import make_asr, make_llm
from .backends.pose import make_pose
from .backends.vlm import make_vlm
from .guardian import Guardian
from .notify import Notifier
from .pipeline import Cascade
from .video import iter_frames


def webcam(index: int):
    cap = cv2.VideoCapture(index)
    start, i = time.monotonic(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, time.monotonic() - start, frame
        i += 1


def paced(frames):
    """Replay a file at real-time speed. After a blocking step the schedule is re-based
    (like a camera that kept running while the single-threaded prototype was busy)."""
    base_wall = base_t = None
    for i, t, f in frames:
        now = time.monotonic()
        if base_wall is None or now - base_wall > (t - base_t) + 0.25:
            base_wall, base_t = now, t
        time.sleep(max(0.0, (t - base_t) - (now - base_wall)))
        yield i, t, f


def main():
    ap = argparse.ArgumentParser(description="AnXin Edge fall guardian")
    ap.add_argument("--video", default="0", help="video file, or camera index")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "hailo"])
    ap.add_argument("--show", action="store_true", help="show the dashboard window")
    ap.add_argument("--record", help="write a wall-clock screen recording (MP4) of the dashboard")
    ap.add_argument("--reply-wav", help="dataset replay: WAV that stands in for the person's spoken reply")
    ap.add_argument("--mic", action="store_true", help="listen with the microphone (live use)")
    ap.add_argument("--mute", action="store_true", help="do not play audio on the speaker")
    ap.add_argument("--long-lie", type=float, default=120.0, help="seconds on the floor after 'I'm OK' before URGENT")
    ap.add_argument("--room", default="living room")
    ap.add_argument("--webhook", help="alert endpoint (default: local family-phone receiver)")
    ap.add_argument("--log", help="write alerts + measured timings as JSON")
    a = ap.parse_args()

    live = a.video.isdigit()
    frames = webcam(int(a.video)) if live else paced(iter_frames(a.video))
    audio = AudioIO(mute=a.mute, reply_wav=a.reply_wav, use_mic=a.mic)
    cascade = Cascade(make_pose(a.backend), make_vlm(a.backend), keep_frames=bool(a.show or a.record))
    disclaimer = None if live else ("Offline replay of a public dataset clip · reply voice is synthesised (TTS)"
                                    if a.reply_wav else "Offline replay of a public dataset clip")
    if not live and a.long_lie < 60:
        disclaimer += f" · long-lie timer shortened to {a.long_lie:g} s for this demo"
    dash_rec = None
    if a.record:
        from .recorder import Recorder
        dash_rec = Recorder(a.record)
    pose_hist = []
    if a.backend == "hailo":
        where = "ASUS UGen300 (Hailo-10H)"
    else:
        import torch
        where = (f"{torch.cuda.get_device_name(0)} GPU, dev server (Stage I)" if torch.cuda.is_available()
                 else "laptop CPU/MPS (Stage I)")
    backend = f"backend: {a.backend}  ·  {where}"

    def draw(rec, st):
        img = dash.render(dash.guardian_view(rec, st, disclaimer, sorted(pose_hist)[len(pose_hist) // 2] if pose_hist else None, backend))
        if dash_rec:
            dash_rec.push(img)
        if a.show:
            cv2.imshow("AnXin Edge  (Esc to quit)", img)
            return cv2.waitKey(1) == 27
        return False

    g = Guardian(cascade, make_asr(a.backend), make_llm(a.backend),
                 Notifier(a.webhook, local_siren=audio.say), audio, room=a.room, long_lie_s=a.long_lie,
                 ui=lambda v: draw(v["rec"], v))
    print("warming up models …", flush=True)
    g.warmup()
    for rec, st in g.run(frames):
        if rec.pose_ms:
            pose_hist.append(rec.pose_ms)
        if rec.vlm:
            print(f"[{rec.t:6.2f}s] VLM: {rec.vlm['choice']}) {rec.vlm['label']}" + (f" -> {rec.verdict}" if rec.verdict else ""), flush=True)
        if (a.show or a.record) and draw(rec, st):
            break
    for al in g.alerts:
        r = al["receipt"]
        print(f"ALERT {al['level']} [{al['kind']}] delivered={r['delivered']} timings={al['timings']}\n  {al['message']}", flush=True)
    if dash_rec:
        dash_rec.close(audio.events)
        print("recording:", a.record)
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(json.dumps({"video": a.video, "reply_wav": a.reply_wav, "long_lie_s": a.long_lie,
                                           "alerts": g.alerts}, indent=2, ensure_ascii=False))
    if a.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
