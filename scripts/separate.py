import sys, wave, numpy as np, torch
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
src, repo, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
torch.set_num_threads(4)
model = get_model("htdemucs_ft", repo=repo)
model.eval()
with wave.open(src) as w:
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32).reshape(-1, ch).T / 32768.0
import torchaudio.functional as F
x = torch.from_numpy(a)
if sr != model.samplerate:
    x = F.resample(x, sr, model.samplerate)
ref = x.mean(0)
x = (x - ref.mean()) / ref.std()
with torch.no_grad():
    s = apply_model(model, x[None], shifts=1, split=True, overlap=0.25, progress=True)[0]
s = s * ref.std() + ref.mean()
vi = model.sources.index("vocals")
for name, idx in [("vocals", vi)]:
    v = s[idx].clamp(-1, 1).numpy()
    with wave.open(out, "wb") as w:
        w.setnchannels(v.shape[0]); w.setsampwidth(2); w.setframerate(model.samplerate)
        w.writeframes((v.T * 32767).astype(np.int16).tobytes())
print("ok", model.sources)
