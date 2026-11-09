"""Transcribe isolated vocals chunk-by-chunk (split at breaths).

usage: chunked_asr.py VOCALS16K.wav OUT.json BACKEND MODEL
BACKEND: fw (faster-whisper) | hf (transformers whisper)
"""
import sys, json, wave, time
import numpy as np

src, out, backend, model_name = sys.argv[1:5]
LANG = sys.argv[5] if len(sys.argv) > 5 else "si"
with wave.open(src) as w:
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0

# --- breath-point chunking ---
hop = int(sr * 0.1)
db = np.array([20 * np.log10(np.sqrt(np.mean(a[i:i + hop] ** 2)) + 1e-9) for i in range(0, len(a), hop)])
MIN_LEN, MAX_LEN = float(sys.argv[6]) if len(sys.argv)>6 else 4.0, float(sys.argv[7]) if len(sys.argv)>7 else 12.0
bounds = [0.0]
i = 0
n = len(db)
while True:
    start = bounds[-1]
    lo, hi = int((start + MIN_LEN) / 0.1), int((start + MAX_LEN) / 0.1)
    if lo >= n:
        break
    hi = min(hi, n - 1)
    seg = db[lo:hi + 1]
    cut = (lo + int(np.argmin(seg))) * 0.1
    if hi == n - 1 and (n * 0.1 - start) <= MAX_LEN:
        break
    bounds.append(round(cut, 2))
bounds.append(len(a) / sr)
chunks = list(zip(bounds[:-1], bounds[1:]))

t0 = time.time()
res = []
if backend == "fw":
    from faster_whisper import WhisperModel
    m = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    for s, e in chunks:
        segs, _ = m.transcribe(a[int(s * sr):int(e * sr)], language=LANG, beam_size=5,
                               condition_on_previous_text=False, compression_ratio_threshold=1.8,
                               repetition_penalty=1.2, no_repeat_ngram_size=3,
                               temperature=[0.0, 0.2, 0.4], vad_filter=False)
        txt = " ".join(x.text.strip() for x in segs)
        res.append({"start": s, "end": e, "text": txt})
        print(f"[{s:6.1f}-{e:6.1f}] {txt}", flush=True)
else:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(4)
    proc = WhisperProcessor.from_pretrained(model_name)
    m = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32).eval()
    from transformers import GenerationConfig
    if "large" in model_name:
        m.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v3")
    for k in ("lang_to_id","task_to_id","is_multilingual"):
        pass
    for s, e in chunks:
        feats = proc(a[int(s * sr):int(e * sr)], sampling_rate=sr, return_tensors="pt").input_features
        with torch.no_grad():
            ids = m.generate(feats, language=LANG, task="transcribe", num_beams=4,
                             no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=120)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        res.append({"start": s, "end": e, "text": txt})
        print(f"[{s:6.1f}-{e:6.1f}] {txt}", flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("done", round(time.time() - t0, 1), "s")
