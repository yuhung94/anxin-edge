#!/usr/bin/env bash
# End-to-end latency benchmark (headless: no display, no recording).
# Needs the receiver running: python scripts/family_phone.py
cd "$(dirname "$0")/.."
for k in 1 2 3; do
  .venv/bin/python -m anxin --mute --video data/urfd/fall-01-cam0.mp4 --reply-wav outputs/audio/reply_help.wav --log outputs/bench/help_$k.json 2>&1 | grep "ALERT URGENT"
  .venv/bin/python -m anxin --mute --video data/urfd/fall-01-cam0.mp4 --log outputs/bench/noreply_$k.json 2>&1 | grep "ALERT URGENT"
done
