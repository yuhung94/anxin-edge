#!/usr/bin/env bash
# Record the three demo runs (unedited, wall-clock screen recordings of the running program).
# Starts the stand-in family-phone receiver, then replays public URFD clips.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p outputs/runs
$PY scripts/family_phone.py > outputs/family_phone.log 2>&1 &
RX=$!; sleep 2
$PY -m anxin --mute --video data/urfd/fall-01-cam0.mp4 --reply-wav outputs/audio/reply_help.wav \
    --record outputs/runs/fall01_help.mp4 --log outputs/runs/fall01_help.json
$PY -m anxin --mute --video data/urfd/adl-36-cam0.mp4 \
    --record outputs/runs/adl36_bed.mp4 --log outputs/runs/adl36_bed.json
$PY -m anxin --mute --video data/urfd/fall-10-cam0.mp4 --reply-wav outputs/audio/reply_ok.wav --long-lie 1 \
    --record outputs/runs/fall10_ok_longlie.mp4 --log outputs/runs/fall10_ok_longlie.json
kill $RX
echo RECORD_DONE
