"""Assemble the ~3-minute Stage I demo video: narrated slides + live-cascade demo segments.

Usage: python scripts/make_video.py ../deck/png   (slide PNGs from deck/build.sh)
Requires outputs/video/demo_*.mp4 from scripts/render_scenarios.py.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anxin.checkin import speak  # noqa: E402

SLIDES = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "deck/png")
VOICE = "Samantha"
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-crf", "20", "-c:a", "aac", "-ar", "44100", "-ac", "2"]

TIMELINE = [
    ("slide", 1, "This is AnXin Edge. AnXin means peace of mind in Mandarin. It is a privacy-first fall guardian for older "
                 "adults who live alone, running four AI models on one ASUS UGen 300 accelerator, at about two and a half watts."),
    ("slide", 3, "Falls kill around six hundred eighty thousand people a year. For people living alone, the real danger is "
                 "the long lie: hours on the floor before anyone knows. Pendants are often not worn, and few people want a "
                 "cloud camera in their bedroom or bathroom."),
    ("slide", 6, "AnXin is a four-tier cascade. Tier zero, YOLOv8 pose, turns every frame into a skeleton. Tier one, a "
                 "skeleton state machine, flags fast falls. Only then does tier two, the Qwen2 V L vision language model, "
                 "verify one keyframe. Tier three talks to the person, using Whisper and a small Qwen language model."),
    ("video", "demo_fall_help", None),
    ("video", "demo_bed", None),
    ("video", "demo_bend", None),
    ("video", "demo_fall_ok", None),
    ("slide", 10, "On all seventy clips of the U R Fall dataset, the skeleton tier alone catches twenty-nine of thirty falls, "
                  "but raises twenty false alarms. Adding the VLM keeps the same recall, and cuts false alarms to two, "
                  "while the VLM runs on only about one percent of frames."),
    ("slide", 17, "AnXin Edge. Camera-level accuracy, radar-level privacy. Thank you."),
]


def main():
    work = Path(tempfile.mkdtemp())
    parts = []
    for k, (kind, ref, text) in enumerate(TIMELINE):
        seg = work / f"seg{k:02d}.mp4"
        if kind == "slide":
            png = SLIDES / f"slide-{ref:02d}.png"
            wav = speak(text, work / f"n{k}.wav", VOICE)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", str(png), "-i", str(wav),
                            "-filter_complex", "[0:v]scale=1280:720,format=yuv420p[v];[1:a]adelay=400|400,apad=pad_dur=0.8[a]",
                            "-map", "[v]", "-map", "[a]", "-shortest", *ENC, str(seg)], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ROOT / "outputs/video" / f"{ref}.mp4"), *ENC, str(seg)],
                           check=True)
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
