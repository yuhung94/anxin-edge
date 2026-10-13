"""Render demo scenarios: run the REAL cascade on public URFD clips and draw the dashboard.

Each scenario -> outputs/video/<name>.mp4 (H.264 + AAC). All model outputs shown on
screen (pose, state machine, VLM answer, Whisper transcript, LLM intent and alert text)
are produced live by the models; only the elderly person's spoken reply is synthesised
with TTS because the dataset has no audio.
Also writes outputs/video/<name>.json with the raw results.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin import dashboard as dash  # noqa: E402
from anxin.backends.genai import make_asr, make_llm  # noqa: E402
from anxin.backends.pose import make_pose  # noqa: E402
from anxin.backends.vlm import make_vlm  # noqa: E402
from anxin.checkin import CHECKIN_PROMPT, caregiver_message, load_wav, run_checkin, speak  # noqa: E402
from anxin.pipeline import Cascade  # noqa: E402
from anxin.video import iter_frames  # noqa: E402

FPS = 30
OUT = ROOT / "outputs/video"
NARRATOR = "Samantha"
ANXIN_VOICE = "Karen"

SCENARIOS = {
    "demo_fall_help": {
        "clip": "fall-01-cam0", "reply": ("Grandma (English (US))", "Oh... I can't get up. My hip really hurts."),
        "room": "living room", "clock": "03:12",
        "intro": "Here is a real fall from the public U R Fall Detection dataset. The skeleton drops fast and stays down, "
                 "so tier one raises a suspicion. The vision language model then looks at a single keyframe, and confirms the fall.",
        "outro": "Whisper transcribes the reply, the distress rule fires, and the language model writes a text-only alert "
                 "for the family. No video ever leaves the home.",
    },
    "demo_fall_ok": {
        "clip": "fall-05-cam0", "reply": ("Grandpa (English (US))", "I'm alright, I just slipped. Give me a minute."),
        "room": "study", "clock": "14:40",
        "intro": "In this second fall, the person answers that they are fine.",
        "outro": "AnXin does not cry wolf. It sends a low-priority note, and keeps watching. If the person is still on the floor "
                 "after two minutes, it escalates to a long-lie alert, because not being able to get up is the real danger.",
    },
    "demo_bed": {
        "clip": "adl-36-cam0",
        "intro": "Everyday life often looks like falling. Lying down on a bed triggers tier one, "
                 "but the vision language model sees a bed, and the alarm is dismissed.",
    },
    "demo_bend": {
        "clip": "adl-04-cam0",
        "intro": "Bending down to pick something up is dismissed the same way. Nobody is disturbed.",
    },
}


def wav_len(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def tiers_for(state: str, phase: str, vlm: dict | None, checkin: dict | None, pose_ms: float, vlm_ms: float | None):
    t0 = {"name": "T0  Pose  ·  YOLOv8s-pose", "status": "ok", "detail": "17 keypoints per frame", "right": f"{pose_ms:.0f} ms/frame"}
    t1 = {"name": "T1  Skeleton fall state machine", "status": "idle", "detail": "watching for fast drop + stay down"}
    if state in ("DESCENDING", "DOWN"):
        t1.update(status="run", detail="person is low / horizontal …")
    if state == "SUSPECTED_FALL" or phase != "play":
        t1.update(status="alert", detail="suspected fall › ask the VLM")
    t2 = {"name": "T2  VLM verify  ·  Qwen2-VL-2B", "status": "idle", "detail": "sleeps until T1 fires (saves power)"}
    if phase == "vlm_wait":
        t2.update(status="run", detail="looking at ONE keyframe held in RAM …")
    elif vlm is not None and phase != "play":
        t2.update(status="alert" if vlm["is_fall"] else "ok",
                  detail=f"“{vlm['choice']}) {vlm['label']}”" + ("  › confirmed" if vlm["is_fall"] else "  › dismissed"),
                  right=f"{vlm_ms:.0f} ms" if vlm_ms else None)
    t3 = {"name": "T3  Voice check-in  ·  Whisper-Base + Qwen2.5-1.5B", "status": "idle", "detail": "talks to the person, decides escalation"}
    if phase in ("ask", "listen"):
        t3.update(status="run", detail="speaking / listening …")
    elif checkin is not None and phase in ("intent", "notify"):
        t3.update(status="alert" if checkin["intent"] != "OK" else "ok",
                  detail=f"intent = {checkin['intent']} (by {checkin['decided_by']})  ›  {checkin['action']}")
    return [t0, t1, t2, t3]


def render_scenario(name: str, sc: dict, cascade: Cascade, asr, llm) -> dict:
    work = Path(tempfile.mkdtemp())
    recs = list(cascade.run(iter_frames(ROOT / "data/urfd" / f"{sc['clip']}.mp4")))
    vrec = next((r for r in recs if r.verdict), None)
    assert vrec is not None, f"{name}: cascade made no decision"
    vlm = vrec.vlm
    pose_ms = sorted(r.pose_ms for r in recs)[len(recs) // 2]
    result = {"clip": sc["clip"], "verdict": vrec.verdict, "vlm": vlm, "verdict_frame": vrec.i}

    frames, audio = [], []  # audio: (t_start_s, wav)

    def emit(state_dict, n=1):
        img = dash.render(state_dict)
        frames.extend([img] * n)

    def t_now():
        return len(frames) / FPS

    intro = speak(sc["intro"], work / "intro.wav", NARRATOR)
    audio.append((0.3, intro))
    intro_end = 0.3 + wav_len(intro) + 0.4

    base = lambda r, **kw: {"frame": r.frame, "kps": r.kps, "feats": r.feats, "t": r.t, "state": r.state, **kw}  # noqa: E731
    first = recs[0]
    emit(base(first, tiers=tiers_for("UPRIGHT", "play", None, None, pose_ms, None),
              banner={"text": f"Public dataset clip {sc['clip']} (UR Fall Detection Dataset) — processed live by the cascade"}), FPS)
    for r in recs[: vrec.i + 1]:
        emit(base(r, tiers=tiers_for(r.state, "play", None, None, pose_ms, None),
                  banner={"text": "Monitoring…  only the skeleton on the right is kept."}))
    emit(base(vrec, tiers=tiers_for(vrec.state, "vlm_wait", vlm, None, pose_ms, None),
              banner={"text": "Suspected fall — asking the on-device VLM to look at this one frame.", "color": dash.AMBER},
              caption="T1: suspected fall"), int(1.5 * FPS))
    confirmed = vrec.verdict == "CONFIRMED"
    vlm_banner = (f"VLM: “{vlm['label']}” › fall CONFIRMED" if confirmed
                  else f"VLM: “{vlm['label']}” › false alarm DISMISSED. No alert, nobody disturbed.")
    hold = max(2.5, intro_end - t_now()) if confirmed else max(3.0, intro_end - t_now())
    emit(base(vrec, tiers=tiers_for(vrec.state, "vlm", vlm, None, pose_ms, vlm.get("ms")),
              banner={"text": vlm_banner, "color": dash.RED if confirmed else dash.GREEN},
              caption="T2: confirmed" if confirmed else "T2: dismissed"), int(hold * FPS))

    if confirmed:
        ask = speak(CHECKIN_PROMPT, work / "ask.wav", ANXIN_VOICE)
        voice, text = sc["reply"]
        reply = speak(text, work / "reply.wav", voice, rate=150)
        bubbles = [("AnXin (speaker)", CHECKIN_PROMPT, dash.TEAL)]
        audio.append((t_now() + 0.2, ask))
        emit(base(vrec, tiers=tiers_for(vrec.state, "ask", vlm, None, pose_ms, None), bubbles=bubbles),
             int((wav_len(ask) + 0.6) * FPS))
        ck = run_checkin(load_wav(reply), asr, llm)
        checkin = {"transcript": ck.transcript, "intent": ck.intent, "decided_by": ck.decided_by, "action": ck.action}
        bubbles = bubbles + [("Person (mic)", "…", dash.AMBER)]
        audio.append((t_now() + 0.1, reply))
        emit(base(vrec, tiers=tiers_for(vrec.state, "listen", vlm, None, pose_ms, None), bubbles=bubbles),
             int((wav_len(reply) + 0.4) * FPS))
        bubbles[-1] = ("Whisper heard", f"“{ck.transcript}”", dash.AMBER)
        emit(base(vrec, tiers=tiers_for(vrec.state, "intent", vlm, checkin, pose_ms, None), bubbles=bubbles), int(2.5 * FPS))
        event = {"time": sc["clock"], "room": sc["room"], "observation": vlm["label"],
                 "person_said": ck.transcript, "action": ck.action}
        msg = caregiver_message(event, llm)
        checkin["caregiver_message"] = msg
        result["checkin"] = checkin
        outro = speak(sc["outro"], work / "outro.wav", NARRATOR)
        audio.append((t_now() + 0.5, outro))
        urgent = ck.action == "ALERT_URGENT"
        emit(base(vrec, tiers=tiers_for(vrec.state, "notify", vlm, checkin, pose_ms, None),
                  notify={"title": ("URGENT ALERT" if urgent else "FYI — low priority") +
                          "  ·  text sent to family phone (written on-device by Qwen2.5-1.5B)",
                          "text": msg, "color": dash.RED if urgent else dash.AMBER}),
             int((wav_len(outro) + 1.5) * FPS))

    # write video + mux audio
    OUT.mkdir(parents=True, exist_ok=True)
    silent = work / "v.mp4"
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (dash.W, dash.H))
    for f in frames:
        vw.write(f)
    vw.release()
    dur = len(frames) / FPS
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(silent)]
    for _, w in audio:
        cmd += ["-i", str(w)]
    parts = [f"[{k + 1}:a]adelay={int(t * 1000)}|{int(t * 1000)}[a{k}]" for k, (t, _) in enumerate(audio)]
    mix = "".join(f"[a{k}]" for k in range(len(audio)))
    filt = ";".join(parts) + f";{mix}amix=inputs={len(audio)}:normalize=0,apad,atrim=0:{dur:.3f}[aout]"
    cmd += ["-filter_complex", filt, "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-c:a", "aac", "-ar", "44100", "-ac", "2", str(OUT / f"{name}.mp4")]
    subprocess.run(cmd, check=True)
    result["duration_s"] = round(dur, 2)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(name, json.dumps({k: v for k, v in result.items() if k != "vlm"}, ensure_ascii=False), flush=True)
    return result


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    cascade = Cascade(make_pose("cpu"), make_vlm("cpu"))
    asr, llm = make_asr("cpu"), make_llm("cpu")
    for n in names:
        render_scenario(n, SCENARIOS[n], cascade, asr, llm)


if __name__ == "__main__":
    main()
