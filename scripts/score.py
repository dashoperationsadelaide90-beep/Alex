import sys, json, torch
sys.path.insert(0, sys.argv[1])
from mms import MMS
m = MMS(); e = torch.load(sys.argv[2]); m.fps = e.shape[0] / 129.4733
tests = json.load(open(sys.argv[3]))
for t0, t1, cands in tests:
    sc = sorted((m.score(e, t0, t1, c), c) for c in cands)
    print(f"[{t0}-{t1}]", " | ".join(f"{s:.2f} {c}" for s, c in sc))
