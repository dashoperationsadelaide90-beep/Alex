import sys, json, torch
sys.path.insert(0, sys.argv[1])
from mms import MMS
m = MMS(); e = torch.load(sys.argv[2]); m.fps = e.shape[0] / 129.4733
L = json.load(open(sys.argv[3]))
out = []
for t0, t1, keys in L["windows"]:
    words, owner = [], []
    for k in keys:
        for w in L["cards"][k][0].split():
            words.append(w); owner.append(k)
    res = m.align_words(e, t0, t1, words)
    # group consecutive words by card occurrence
    i = 0
    for k in keys:
        n = len(L["cards"][k][0].split())
        seg = res[i:i + n]; i += n
        st = min(s for _, s, _ in seg if s is not None); en = max(x for _, _, x in seg if x is not None)
        out.append({"key": k, "si": L["cards"][k][0], "en": L["cards"][k][1], "start": round(st, 2), "end": round(en, 2),
                    "words": [(w, round(s, 2), round(x, 2)) for w, s, x in seg]})
        print(f"{st:7.2f} {en:7.2f}  {k}  {L['cards'][k][0]}   " + " ".join(f"{w}@{s:.1f}" for w, s, _ in seg))
json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False, indent=1)
