#!/usr/bin/env bash
# Download the UR Fall Detection Dataset (Kwolek & Kepski, 2014) RGB videos (camera 0).
# 30 fall sequences + 40 activities-of-daily-living (ADL) sequences, ~80 MB total.
# Source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html  (research use; cite the paper)
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/urfd"
mkdir -p "$DIR"
BASE=https://fenix.ur.edu.pl/~mkepski/ds/data
for i in $(seq -w 1 30); do f=fall-$i-cam0.mp4; [ -s "$DIR/$f" ] || curl -sfL -o "$DIR/$f" "$BASE/$f"; done
for i in $(seq -w 1 40); do f=adl-$i-cam0.mp4;  [ -s "$DIR/$f" ] || curl -sfL -o "$DIR/$f" "$BASE/$f"; done
echo "done: $(ls "$DIR" | wc -l) files in $DIR"
# Second viewpoint (cam1) exists for the 30 falls only: used as an untouched cross-view test set.
mkdir -p "$DIR/../urfd_cam1"
for i in $(seq -w 1 30); do f=fall-$i-cam1.mp4; [ -s "$DIR/../urfd_cam1/$f" ] || curl -sfL -o "$DIR/../urfd_cam1/$f" "$BASE/$f"; done
echo "cam1: $(ls "$DIR/../urfd_cam1" | wc -l) files"
