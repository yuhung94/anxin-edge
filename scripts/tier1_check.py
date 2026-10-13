import glob, os, sys
import numpy as np
sys.path.insert(0, '.')
from anxin.fall import run_sequence, FallConfig
cfg = FallConfig(**{k: float(v) for k, v in (a.split('=') for a in sys.argv[1:])})
tp = fp = 0; miss = []; fps_ = []
for f in sorted(glob.glob('data/pose_cache/*.npz')):
    d = np.load(f); n = os.path.basename(f)[:-4]; H, W = d['shape']
    first, _, _ = run_sequence(d['kps'], H, W, cfg)
    if n.startswith('fall'):
        tp += first is not None
        if first is None: miss.append(n)
    elif first is not None:
        fp += 1; fps_.append(n)
print('TP', tp, '/30  FP', fp, '/40'); print('miss', miss); print('fp', fps_)
