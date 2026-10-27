#!/usr/bin/env bash
# One-shot on the GPU server: setup -> record 3 live runs -> narrate -> assemble the demo video.
set -e
cd "$(dirname "$0")"
source ./env.sh
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.11 .venv
fi
uv pip install -q --python .venv/bin/python -r requirements.txt pytest kokoro soundfile
./scripts/download_urfd.sh
.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest -q tests
./scripts/record_runs.sh
.venv/bin/python scripts/make_video.py slides_png
echo ALL_DONE
