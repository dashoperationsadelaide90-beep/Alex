"""Procedural worship background: dawn gradient, soft rays, drifting bokeh.

Writes raw rgb24 frames to stdout at the given (half) resolution.
usage: background.py W H FPS DURATION AUDIO_WAV SEED
"""
import sys
import wave
import numpy as np

W, H, FPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
DUR = float(sys.argv[4])
AUDIO = sys.argv[5]
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 7
N = int(round(DUR * FPS))
rng = np.random.default_rng(SEED)

# ---------- audio envelope (smoothed RMS per frame) ----------
with wave.open(AUDIO) as w:
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
a = a.reshape(-1, ch).mean(axis=1) / 32768.0
hop = sr / FPS
env = np.array([np.sqrt(np.mean(a[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-9) if int(i * hop) + 1 < len(a) else 0.0 for i in range(N)])
k = np.exp(-np.linspace(-2, 2, int(FPS * 0.8)) ** 2)
env = np.convolve(env, k / k.sum(), mode="same")
env = (env - np.percentile(env, 5)) / (np.percentile(env, 95) - np.percentile(env, 5) + 1e-9)
env = np.clip(env, 0, 1.2)

# ---------- static fields ----------
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
u, v = x / W, y / H                       # 0..1
aspect = W / H
portrait = H > W


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


top, mid, low = hexrgb("0a0d26"), hexrgb("24164a"), hexrgb("63302c")
g = v[..., None]
base = np.where(g < 0.55, top + (mid - top) * (g / 0.55), mid + (low - mid) * ((g - 0.55) / 0.45))

# horizon glow centre (lower middle)
gx, gy = 0.5, (0.86 if not portrait else 0.80)
dx, dy = (u - gx) * (aspect if aspect > 1 else 1), (v - gy) * (1 if aspect > 1 else 1 / aspect)
rglow = np.sqrt(dx * dx + dy * dy * 2.2)
glow_shape = np.exp(-(rglow / 0.42) ** 2)
glow_col = hexrgb("ffb25c")
halo_shape = np.exp(-(rglow / 0.16) ** 2)
halo_col = hexrgb("ffe2b0")

# rays from above top centre
rx, ry = 0.5, -0.25
ang = np.arctan2((u - rx) * aspect, (v - ry))
rdist = np.sqrt(((u - rx) * aspect) ** 2 + (v - ry) ** 2)
ray_fall = np.clip(1.0 - (v / 0.95), 0, 1) ** 1.6 * np.exp(-((u - 0.5) * aspect / 0.9) ** 2)
ray_col = hexrgb("cfd8ff")

# vignette
vd = np.sqrt(((u - 0.5) * 1.15) ** 2 + ((v - 0.5) * 1.15) ** 2)
vign = np.clip(1.0 - 0.75 * np.clip(vd - 0.25, 0, 1) ** 1.4 * 1.6, 0.25, 1)[..., None]

# ---------- bokeh particles ----------
NP = 46 if not portrait else 52
scale = min(W, H)
P = {
    "x": rng.uniform(0, W, NP),
    "y": rng.uniform(0, H, NP),
    "r": rng.uniform(0.012, 0.055, NP) * scale,
    "vy": rng.uniform(4, 16, NP) * (scale / 540),          # px/s upward
    "sway": rng.uniform(4, 20, NP) * (scale / 540),
    "sf": rng.uniform(0.05, 0.25, NP),
    "ph": rng.uniform(0, 2 * np.pi, NP),
    "tw": rng.uniform(0.15, 0.6, NP),
    "a": rng.uniform(0.10, 0.38, NP),
    "warm": rng.uniform(0, 1, NP),
}
cw, cc = hexrgb("ffc878"), hexrgb("bcd2ff")


def sprite(r):
    s = int(np.ceil(r * 2.4))
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / r
    disc = 1.0 / (1.0 + np.exp((d - 0.85) * 9))        # soft-edged disc
    ring = np.exp(-((d - 0.82) / 0.09) ** 2) * 0.35    # faint bokeh rim
    return (disc * 0.75 + ring), s


sprites = [sprite(r) for r in P["r"]]

out = sys.stdout.buffer
for f in range(N):
    t = f / FPS
    e = env[f]
    img = base.copy()
    breathe = 0.85 + 0.10 * np.sin(t * 0.35) + 0.22 * e
    img += glow_shape[..., None] * glow_col * (0.44 * breathe)
    img += halo_shape[..., None] * halo_col * (0.10 * breathe)
    rays = (0.5 + 0.5 * np.sin(ang * 9 + t * 0.10)) ** 6 + 0.6 * (0.5 + 0.5 * np.sin(ang * 5 - t * 0.07 + 1.3)) ** 8
    img += (rays * ray_fall)[..., None] * ray_col * (0.07 + 0.03 * e)

    layer = np.zeros((H, W, 3), np.float32)
    for i in range(NP):
        r = P["r"][i]
        cy = (P["y"][i] - P["vy"][i] * t) % (H + 4 * r) - 2 * r
        cx = P["x"][i] + P["sway"][i] * np.sin(2 * np.pi * P["sf"][i] * t + P["ph"][i])
        alpha = P["a"][i] * (1 - P["tw"][i] * (0.5 + 0.5 * np.sin(t * 0.9 + P["ph"][i] * 3))) * (0.85 + 0.35 * e)
        spr, s = sprites[i]
        x0, y0 = int(cx) - s, int(cy) - s
        x1, y1 = x0 + spr.shape[1], y0 + spr.shape[0]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if X1 <= X0 or Y1 <= Y0:
            continue
        col = cw * P["warm"][i] + cc * (1 - P["warm"][i])
        patch = spr[sy0:sy0 + (Y1 - Y0), sx0:sx0 + (X1 - X0)]
        layer[Y0:Y1, X0:X1] += patch[..., None] * col * alpha
    img = img + layer - img * layer * 0.5           # gentle screen-ish blend
    img *= vign
    # fade in / out
    fade = min(1.0, t / 1.5, (DUR - t) / 2.0)
    img *= max(0.0, fade)
    out.write((np.clip(img, 0, 1) ** (1 / 1.05) * 255).astype(np.uint8).tobytes())
