"""MMS CTC helper: emissions, greedy romanized decode, hypothesis scoring, forced alignment."""
import sys, json, wave, re
import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForCTC
import uroman as ur

MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
SR = 16000
_u = ur.Uroman()


def load_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def romanize(txt):
    r = _u.romanize_string(txt).lower()
    r = re.sub(r"[^a-z' ]", " ", r)
    return re.sub(r"\s+", " ", r).strip()


class MMS:
    def __init__(self):
        torch.set_num_threads(4)
        self.m = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
        from huggingface_hub import hf_hub_download
        self.vocab = json.load(open(hf_hub_download(MODEL, "vocab.json")))
        self.inv = {v: k for k, v in self.vocab.items()}
        self.star = len(self.vocab)  # extra "star" token = any garbage

    def emissions(self, a, win=30.0, ctx=2.0):
        out = []
        step = int(win * SR)
        c = int(ctx * SR)
        for s in range(0, len(a), step):
            lo, hi = max(0, s - c), min(len(a), s + step + c)
            x = torch.from_numpy(a[lo:hi])[None]
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.no_grad():
                lg = self.m(x).logits[0]
            lp = torch.log_softmax(lg, -1)
            fps = lp.shape[0] / ((hi - lo) / SR)
            a0 = int(round((s - lo) / SR * fps))
            a1 = a0 + int(round(min(step, len(a) - s) / SR * fps))
            out.append(lp[a0:a1])
        e = torch.cat(out)
        self.fps = e.shape[0] / (len(a) / SR)
        return e

    def greedy(self, e, t0, t1):
        f0, f1 = int(t0 * self.fps), int(t1 * self.fps)
        ids = e[f0:f1].argmax(-1).tolist()
        s, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                s.append(self.inv.get(i, "?"))
            prev = i
        return "".join(s).replace("|", " ")

    def tokens(self, rom):
        return [self.vocab[ch] if ch in self.vocab else self.vocab["<unk>"] for ch in rom.replace(" ", "")]

    def score(self, e, t0, t1, text):
        """mean per-char CTC negative log-likelihood of text over [t0,t1] (lower is better)."""
        f0, f1 = int(t0 * self.fps), int(t1 * self.fps)
        toks = self.tokens(romanize(text))
        loss = torch.nn.functional.ctc_loss(e[f0:f1][:, None], torch.tensor([toks]), torch.tensor([f1 - f0]),
                                            torch.tensor([len(toks)]), blank=0, reduction="sum", zero_infinity=True)
        return float(loss) / max(1, len(toks))

    def align_words(self, e, t0, t1, words):
        """Force-align list of (sinhala) words inside [t0,t1]; returns [(word, start, end)]."""
        f0, f1 = int(t0 * self.fps), int(t1 * self.fps)
        roms = [romanize(w).replace(" ", "") for w in words]
        toks, owner = [], []
        for i, r in enumerate(roms):
            for ch in r:
                toks.append(self.vocab.get(ch, self.vocab["<unk>"]))
                owner.append(i)
        em = e[f0:f1][None]
        ali, sc = AF.forced_align(em, torch.tensor([toks], dtype=torch.int32), blank=0)
        ali = ali[0].tolist()
        spans = {}
        k = -1
        prev = 0
        for fi, tk in enumerate(ali):
            if tk != 0 and (tk != prev or prev == 0):
                k += 1
            if tk != 0:
                w = owner[k]
                st, en = spans.get(w, (fi, fi))
                spans[w] = (min(st, fi), max(en, fi))
            prev = tk
        res = []
        for i, w in enumerate(words):
            st, en = spans.get(i, (None, None))
            res.append((w, None if st is None else t0 + st / self.fps, None if en is None else t0 + (en + 1) / self.fps))
        return res


if __name__ == "__main__":
    mms = MMS()
    a = load_wav(sys.argv[1])
    e = mms.emissions(a)
    torch.save(e, sys.argv[2])
    print("fps", mms.fps)
    step = 4.0
    t = 0.0
    while t < len(a) / SR:
        print(f"[{t:6.1f}] {mms.greedy(e, t, t + step)}")
        t += step
