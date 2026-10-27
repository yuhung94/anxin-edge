"""Assemble the Stage I demo video (≈3 min) from narrated slides + UNEDITED live recordings.

Recordings come from `python -m anxin ... --record outputs/runs/<name>.mp4` (wall-clock screen
recordings of the running program; see scripts/record_runs.sh). They are inserted as-is: no
cuts, no speed-up, no overlays. Narration is only placed on slide segments.

Usage:  python scripts/make_video.py <slide-png-dir>
Narration: Kokoro-82M neural TTS (Linux/GPU) when installed, else macOS `say`.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin.checkin import speak  # noqa: E402

SLIDES = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "deck/png")
VOICE = "Samantha"
_KOKORO = None


def narrate(text: str, out: Path) -> Path:
    global _KOKORO
    if shutil.which("say"):
        return speak(text, out, VOICE, rate=185)
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    _KOKORO = _KOKORO or KPipeline(lang_code="a")
    audio = np.concatenate([a for _, _, a in _KOKORO(text, voice="af_heart", speed=1.05)])
    sf.write(out, audio, 24000)
    return out
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-crf", "20", "-c:a", "aac", "-ar", "44100", "-ac", "2"]

TIMELINE = [
    ("slide", 1, "AnXin Edge guards older adults who live alone. Here is a real fall from a public dataset, processed "
                 "end to end. This is an unedited, real-time recording."),
    ("rec", "fall01_help"),
    ("slide", 3, "What is real today: the whole loop runs on a laptop and a GPU server, and is tested. The UGen 300 "
                 "version is written but not yet run on the hardware. That is stage two."),
    ("slide", 7, "Pose and skeleton rules run on every frame. The vision language model only checks suspected falls. "
                 "Escalation uses fixed rules, so urgent alerts never wait for the language model."),
    ("slide", 12, "Lying down on a bed looks like a fall to the skeleton tier. The vision language model dismisses it."),
    ("rec", "adl36_bed"),
    ("slide", 6, "If the person says they are okay but stays down, a long-lie timer escalates. Default two minutes; "
                 "shortened to one second here because dataset clips are short."),
    ("rec", "fall10_ok_longlie"),
    ("slide", 13, "On seventy public clips, false alarms drop from twenty to two, with twenty-nine of thirty falls detected. "
                  "We also show where it fails, including a ceiling camera, where the skeleton tier catches six of thirty."),
    ("slide", 17, "Stage two moves the same code to the UGen 300, and measures accuracy, latency and power on the "
                  "real hardware."),
    ("slide", 20, "AnXin Edge. Thank you."),
]


def main():
    work = Path(tempfile.mkdtemp())
    parts = []
    for k, item in enumerate(TIMELINE):
        seg = work / f"seg{k:02d}.mp4"
        if item[0] == "slide":
            _, ref, text = item
            wav = narrate(text, work / f"n{k}.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(SLIDES / f"slide-{ref:02d}.png"), "-i", str(wav),
                            "-filter_complex", "[0:v]scale=1280:720,format=yuv420p[v];[1:a]adelay=300|300,apad=pad_dur=0.6[a]",
                            "-map", "[v]", "-map", "[a]", "-shortest", *ENC, str(seg)], check=True)
        else:
            src = ROOT / "outputs/runs" / f"{item[1]}.mp4"
            # add a silent track if the recording has no audio, so concat stays aligned
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                            "-filter_complex", "[0:v]scale=1280:720[v]", "-map", "[v]", "-map", "0:a?", "-map", "1:a",
                            "-shortest", *ENC, str(seg.with_suffix(".tmp.mp4"))], check=True)
            has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                                        "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip()
            amap = "0:a:0"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(seg.with_suffix(".tmp.mp4")), "-map", "0:v", "-map", amap,
                            "-c", "copy", str(seg)], check=True)
            if not has_audio:
                print(f"note: {item[1]} has no audio track (silence added)")
        parts.append(seg)
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = ROOT / "outputs/AnXin_Edge_Stage1_Demo.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out}  ({float(dur):.1f} s)")


if __name__ == "__main__":
    main()
