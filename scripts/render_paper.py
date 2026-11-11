"""'Lyrics written on paper' renderer.

A pen writes each lyric line (Sinhala, then English) on ruled notebook paper in sync
with the song; repeats are followed by the pen pointing under the sung words.
Raw RGB frames go to stdout.

usage: render_paper.py ALIGNED.json FONTS_DIR WORK_DIR FMT(916|169) TOTAL_DUR [--preview t1,t2,...]
"""
import sys, json, os, subprocess, math
import numpy as np
import cv2

ALIGNED, FONTS, WORK, FMT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
TOTAL = float(sys.argv[5])
PREVIEW = None
if len(sys.argv) > 7 and sys.argv[6] == "--preview":
    PREVIEW = [float(x) for x in sys.argv[7].split(",")]
FPS = 30
OW, OH = (1080, 1920) if FMT == "916" else (1920, 1080)
rng = np.random.default_rng(3)
os.makedirs(WORK, exist_ok=True)

cards = json.load(open(ALIGNED))

# ---------------------------------------------------------------- page layout
PW = 1200            # page width (page px)
TOP = 210            # y of first ruled line
RS = 96              # row spacing
MARGIN_X = 128       # red margin line
X0 = 156             # text start x
SI_SIZE, EN_SIZE = 80, 54
N_ROWS = 35
PH = TOP + N_ROWS * RS + 90
DESK = 460 if FMT == "916" else 760   # desk margin around page in world
WW, WH = PW + 2 * DESK, PH + 2 * DESK
PX, PY = DESK, DESK  # page origin in world

first_seen, rows = {}, {}
order = []
for i, c in enumerate(cards):
    if c["key"] not in first_seen:
        first_seen[c["key"]] = i
        order.append(c["key"])
# row assignment: title 0, subtitle 1, blank, verse, blank, bridge, blank, chorus
r = 3
for k in order:
    if k == "b1" or k == "c1":
        r += 1
    rows[k] = r
    r += 2
assert r <= N_ROWS, r


def row_y(rw):
    return PY + TOP + rw * RS


# ---------------------------------------------------------------- text masks via libass
items = {}   # name -> dict(text, font, size)
for k in order:
    c = cards[first_seen[k]]
    items[f"si_{k}"] = (c["si"], "Yaldevi Medium", SI_SIZE)
    items[f"en_{k}"] = (c["en"], "Patrick Hand", EN_SIZE)
items["title"] = ("කුණාටු මධ්‍යයේ", "Yaldevi Medium", 104)
items["subtitle"] = ("ඔබ මා පර්වතයයි", "Yaldevi", 56)
items["x2"] = ("x 2", "Patrick Hand", 92)
items["x3"] = ("x 3", "Patrick Hand", 92)
items["ref_si"] = ("කමනවප", "Yaldevi Medium", SI_SIZE)
items["ref_en"] = ("xmnaco", "Patrick Hand", EN_SIZE)
names = list(items)
MW, MHt, BASE = 1300, 280, 190   # mask canvas; text bottom-left anchored at (20, BASE)


def ts(t):
    return f"0:00:{t:05.2f}" if t < 60 else f"0:{int(t//60):02d}:{t%60:05.2f}"


ass = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {MW}", f"PlayResY: {MHt}", "",
       "[V4+ Styles]",
       "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
       "Style: D,Arial,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,1,0,0,0,1",
       "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
for i, n in enumerate(names):
    txt, font, size = items[n]
    ass.append(f"Dialogue: 0,{ts(i*2)},{ts(i*2+1.5)},D,,0,0,0,,{{\\an1\\pos(20,{BASE})\\fn{font}\\fs{size}}}{txt}")
open(f"{WORK}/masks.ass", "w", encoding="utf-8").write("\n".join(ass) + "\n")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={MW}x{MHt}:r=1:d={len(names)*2}",
                "-vf", f"select='not(mod(n\\,2))',ass={WORK}/masks.ass:fontsdir={FONTS},format=gray",
                "-vsync", "0", f"{WORK}/m_%03d.png"], check=True)

raw = {}
for i, n in enumerate(names):
    m = cv2.imread(f"{WORK}/m_{i+1:03d}.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    raw[n] = m


def baseline_of(m):
    ys = np.where(m.max(axis=1) > 0.5)[0]
    return ys.max()


BL = {"si": baseline_of(raw["ref_si"]), "en": baseline_of(raw["ref_en"])}


def wobble(m, amp=1.3, rot=0.6):
    """hand-written feel: small smooth displacement + slight rotation."""
    h, w = m.shape
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 14)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 14)
    dx *= amp / (dx.std() + 1e-6); dy *= amp / (dy.std() + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = cv2.remap(m, gx + dx, gy + dy, cv2.INTER_LINEAR)
    ang = rng.uniform(-rot, rot)
    M = cv2.getRotationMatrix2D((20, BASE), ang, 1.0)
    return cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR)


class Ink:
    """A placed piece of handwriting on the page."""

    def __init__(self, name, x, base_y, kind, nwords=None, alpha=0.94, color=(26, 36, 108)):
        m = wobble(raw[name]) if kind != "ref" else raw[name]
        cols = np.where(m.max(axis=0) > 0.05)[0]
        rws = np.where(m.max(axis=1) > 0.05)[0]
        x0, x1 = max(0, cols.min() - 3), min(MW, cols.max() + 4)
        y0, y1 = max(0, rws.min() - 3), min(MHt, rws.max() + 4)
        self.m = m[y0:y1, x0:x1]
        bl = BL["si"] if kind == "si" else BL["en"] if kind in ("en", "lat") else BASE
        self.x = int(round(x - (20 - x0)))
        self.y = int(round(base_y - (bl - y0)))
        h, w = self.m.shape
        tex = 0.86 + 0.14 * cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.2) / 0.5
        self.a = np.clip(self.m * alpha * np.clip(tex, 0.75, 1.05), 0, 1)
        self.color = np.array(color, np.float32)
        # column profile -> pen y while writing
        colmass = self.m.sum(axis=0)
        yy = np.arange(h, dtype=np.float32)[:, None]
        cy = (self.m * yy).sum(axis=0) / np.maximum(colmass, 1e-3)
        mid = (h * 0.55)
        cy = np.where(colmass > 0.3, cy, np.nan)
        # fill gaps by interpolation
        idx = np.arange(w)
        ok = ~np.isnan(cy)
        cy = np.interp(idx, idx[ok], cy[ok]) if ok.any() else np.full(w, mid)
        self.pen_y = cv2.GaussianBlur(cy.astype(np.float32)[None], (0, 0), 3)[0]
        self.w, self.h = w, h
        # word boundaries (n-1 widest empty-column gaps)
        self.bounds = [0, w]
        if nwords and nwords > 1:
            empty = colmass < 0.05
            gaps, s = [], None
            for xx in range(w):
                if empty[xx] and s is None:
                    s = xx
                if not empty[xx] and s is not None:
                    gaps.append((xx - s, (s + xx) / 2)); s = None
            gaps = sorted(gaps, reverse=True)[:nwords - 1]
            self.bounds = [0] + sorted(g[1] for g in gaps) + [w]

    def draw(self, canvas, frac_cols=None, off=(0, 0)):
        """alpha-blend into canvas whose top-left is world (off); frac_cols = reveal front column."""
        a = self.a
        if frac_cols is not None:
            cols = np.arange(self.w, dtype=np.float32)
            a = a * np.clip((frac_cols - cols) / 5.0, 0, 1)[None]
        blend(canvas, off, self.x, self.y, a, self.color)


def blend(canvas, off, x, y, a, color):
    H, W = canvas.shape[:2]
    x0, y0 = x - off[0], y - off[1]
    h, w = a.shape
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if X1 <= X0 or Y1 <= Y0:
        return
    sub = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    reg = canvas[Y0:Y1, X0:X1]
    if np.ndim(color) == 1:
        reg += (color - reg) * sub[..., None]
    else:
        reg += (color[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] - reg) * sub[..., None]


# place inks
inks = {}
for k in order:
    nw = len(cards[first_seen[k]]["words"])
    inks[f"si_{k}"] = Ink(f"si_{k}", PX + X0, row_y(rows[k]) - 10, "si", nwords=nw)
    inks[f"en_{k}"] = Ink(f"en_{k}", PX + X0 + 30, row_y(rows[k] + 1) - 12, "en", alpha=0.82, color=(40, 52, 120))
t_title = Ink("title", 0, row_y(0) - 10, "si")
t_title.x = PX + (PW - t_title.w) // 2
t_sub = Ink("subtitle", 0, row_y(1) - 10, "si", alpha=0.85)
t_sub.x = PX + (PW - t_sub.w) // 2
vr = [rows[k] for k in ("v1", "v2", "v3", "v4")]
cr = [rows[k] for k in ("c1", "c2", "c3", "c4")]
inks["x2"] = Ink("x2", PX + PW - 270, row_y((min(vr) + max(vr) + 1) / 2 + 0.5) - 10, "lat")
inks["x3"] = Ink("x3", PX + PW - 270, row_y((min(cr) + max(cr) + 1) / 2 + 0.5) - 10, "lat")

# ---------------------------------------------------------------- static world (desk + paper)
def make_world():
    # wood desk
    n = rng.standard_normal((WH // 4, WW // 4)).astype(np.float32)
    grain = cv2.GaussianBlur(n, (0, 0), sigmaX=14, sigmaY=1.0)
    grain = cv2.resize(grain, (WW, WH), interpolation=cv2.INTER_CUBIC)
    grain = (grain - grain.mean()) / (grain.std() + 1e-6)
    fine = cv2.GaussianBlur(rng.standard_normal((WH, WW)).astype(np.float32), (0, 0), sigmaX=6, sigmaY=0.6)
    wood = np.zeros((WH, WW, 3), np.float32)
    rgb = np.array([118, 80, 52], np.float32)
    wood[:] = rgb
    yy_, xx_ = np.mgrid[0:WH, 0:WW].astype(np.float32)
    warp = cv2.resize(cv2.GaussianBlur(rng.standard_normal((WH // 16, WW // 16)).astype(np.float32), (0, 0), 4), (WW, WH))
    rings = np.sin((yy_ + 60 * warp / (warp.std() + 1e-6) + 0.02 * xx_) * 0.11) ** 8
    planks = (np.abs(((yy_ + 37) % 230) - 0) < 2.5).astype(np.float32)
    wood *= (1 + 0.07 * grain[..., None] + 0.06 * fine[..., None] - 0.045 * rings[..., None] - 0.35 * planks[..., None])
    # paper
    paper_rgb = np.array([246, 243, 235], np.float32)
    paper = np.zeros((PH, PW, 3), np.float32) + paper_rgb
    blot = cv2.GaussianBlur(rng.standard_normal((PH // 8, PW // 8)).astype(np.float32), (0, 0), 6)
    blot = cv2.resize(blot, (PW, PH)) / (blot.std() + 1e-6)
    paper *= (1 + 0.012 * blot[..., None])
    paper += rng.normal(0, 2.2, (PH, PW, 1)).astype(np.float32)
    # rules
    rule = np.array([150, 185, 222], np.float32)
    for i in range(N_ROWS + 1):
        y = TOP + i * RS
        for dy, a in ((-1, 0.18), (0, 0.55), (1, 0.18)):
            paper[y + dy] += (rule - paper[y + dy]) * a
    red = np.array([226, 128, 128], np.float32)
    for dx, a in ((-1, 0.2), (0, 0.6), (1, 0.2)):
        paper[:, MARGIN_X + dx] += (red - paper[:, MARGIN_X + dx]) * a
    # faint show-through of the previous page (mirrored writing)
    for nm, (gx, gy) in (("en_c3", (420, 7)), ("en_b2", (300, 15)), ("si_b5", (260, 22)), ("en_v2", (380, 29))):
        m = cv2.flip(inks[nm].m, 1) if nm in inks else None
        if m is None:
            continue
        y = TOP + gy * RS - m.shape[0] + 6; x = gx
        reg = paper[y:y + m.shape[0], x:x + m.shape[1]]
        reg -= (reg - np.array([200, 200, 205], np.float32)) * (m[..., None] * 0.05)
    # page edge darkening
    yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
    edge = np.minimum(np.minimum(xx, PW - 1 - xx), np.minimum(yy, PH - 1 - yy))
    paper *= (0.94 + 0.06 * np.clip(edge / 40, 0, 1))[..., None]
    # shadow under paper
    sh = np.zeros((WH, WW), np.float32)
    sh[PY + 10:PY + PH + 18, PX + 6:PX + PW + 14] = 1
    sh = cv2.GaussianBlur(sh, (0, 0), 14)
    wood *= (1 - 0.55 * sh[..., None])
    wood[PY:PY + PH, PX:PX + PW] = paper
    return np.clip(wood, 0, 255)


world0 = make_world()
for ink in (t_title, t_sub):
    ink.draw(world0)

# ---------------------------------------------------------------- pen sprite (tip at origin)
def make_pen():
    S = 3
    L, ang = 760, math.radians(52)          # body goes to lower-right
    pts_len = [(0, 1.0), (24, 5.5), (44, 8.0), (56, 13), (176, 15), (188, 14), (680, 14), (720, 12), (L, 10)]
    W = H = 2 * (L + 40)
    img = np.zeros((H * S, W * S, 4), np.float32)
    ox, oy = (W // 2) * S, (H // 2) * S
    d = np.array([math.cos(ang), math.sin(ang)]); nrm = np.array([-d[1], d[0]])

    def quad(a, b, wa, wb, col, alpha=1.0):
        p = [np.array([ox, oy]) + S * (a * d + wa * nrm), np.array([ox, oy]) + S * (b * d + wb * nrm),
             np.array([ox, oy]) + S * (b * d - wb * nrm), np.array([ox, oy]) + S * (a * d - wa * nrm)]
        poly = np.array(p, np.int32)
        layer = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(layer, [poly], 255, lineType=cv2.LINE_AA)
        al = layer.astype(np.float32) / 255 * alpha
        img[..., :3] += (np.array(col, np.float32) - img[..., :3]) * al[..., None]
        img[..., 3] = np.maximum(img[..., 3], al)

    quad(0, 24, 1.0, 5.5, (175, 175, 180))            # metal tip
    quad(24, 44, 5.5, 8.0, (205, 205, 210))
    quad(44, 56, 8.0, 13, (150, 150, 158))             # ferrule
    quad(56, 176, 13, 15, (34, 36, 44))                # grip
    for g in range(64, 172, 9):                         # grip ridges
        quad(g, g + 3, 15, 15, (58, 60, 70))
    quad(176, 188, 15, 14, (190, 190, 196))            # metal ring
    quad(188, 680, 14, 14, (20, 34, 92))               # body (navy)
    quad(680, 720, 14, 12, (190, 190, 196))
    quad(720, L, 12, 10, (20, 34, 92))
    quad(560, 700, 4, 4, (205, 205, 212))              # clip
    # highlight stripe and shade
    hl = np.zeros(img.shape[:2], np.float32)
    a0 = np.array([ox, oy]) + S * (60 * d - 7 * nrm); a1 = np.array([ox, oy]) + S * (740 * d - 6 * nrm)
    cv2.line(hl, tuple(a0.astype(int)), tuple(a1.astype(int)), 1.0, int(4 * S), cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), 3 * S) * img[..., 3]
    img[..., :3] += (255 - img[..., :3]) * (hl[..., None] * 0.35)
    sd = np.zeros(img.shape[:2], np.float32)
    b0 = np.array([ox, oy]) + S * (60 * d + 10 * nrm); b1 = np.array([ox, oy]) + S * (740 * d + 9 * nrm)
    cv2.line(sd, tuple(b0.astype(int)), tuple(b1.astype(int)), 1.0, int(6 * S), cv2.LINE_AA)
    sd = cv2.GaussianBlur(sd, (0, 0), 3 * S) * img[..., 3]
    img[..., :3] *= (1 - 0.45 * sd[..., None])
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, (W // 2, H // 2)


PEN0, PEN_O = make_pen()
PEN_BASE_ANG = 52.0
_pen_cache = {}


def pen_at(angle):
    """pen sprite + shadow rotated so the body points at `angle` deg (image coords), tip fixed."""
    key = round(angle * 2) / 2
    if key not in _pen_cache:
        if abs(key - PEN_BASE_ANG) < 0.25:
            spr = PEN0
        else:
            M = cv2.getRotationMatrix2D(PEN_O, PEN_BASE_ANG - key, 1.0)
            spr = cv2.warpAffine(PEN0, M, (PEN0.shape[1], PEN0.shape[0]), flags=cv2.INTER_LINEAR)
        # crop to content (keep tip offset)
        ys, xs = np.where(spr[..., 3] > 0.01)
        y0, y1, x0, x1 = ys.min() - 30, ys.max() + 30, xs.min() - 30, xs.max() + 30
        y0, x0 = max(0, y0), max(0, x0)
        spr = spr[y0:y1, x0:x1].copy()
        sh = cv2.GaussianBlur(spr[..., 3], (0, 0), 5)
        _pen_cache[key] = (spr, sh, (int(PEN_O[0] - x0), int(PEN_O[1] - y0)))
    return _pen_cache[key]


def blit_pen(canvas, tx, ty, lift, off=(0, 0), angle=PEN_BASE_ANG):
    """draw pen with shadow, tip at (tx,ty) in world coords; lift 0..1."""
    tx -= off[0]; ty -= off[1]
    CH, CW = canvas.shape[:2]
    PEN, PEN_SH, PO = pen_at(angle)
    h, w = PEN.shape[:2]
    # shadow: pivots at the tip, offset grows with lift (light from upper-left)
    ang = 4 + 5 * lift
    M = cv2.getRotationMatrix2D(PO, -ang, 1.0)
    M[0, 2] += 6 + 26 * lift; M[1, 2] += 4 + 30 * lift
    sh = cv2.warpAffine(PEN_SH, M, (w + 80, h + 80))
    x0, y0 = int(tx) - PO[0], int(ty) - PO[1]
    for layer, isshadow in ((sh, True), (PEN, False)):
        lh, lw = layer.shape[:2]
        X0, Y0 = max(0, x0), max(0, y0)
        X1, Y1 = min(CW, x0 + lw), min(CH, y0 + lh)
        if X1 <= X0 or Y1 <= Y0:
            continue
        sub = layer[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        reg = canvas[Y0:Y1, X0:X1]
        if isshadow:
            reg *= (1 - 0.32 * (1 - 0.35 * lift) * sub[..., None])
        else:
            reg += (sub[..., :3] - reg) * sub[..., 3:4]


# ---------------------------------------------------------------- timeline
writes = []   # dict(ink, t0, t1, tmap=[(t, col)])
points = []   # dict(ink, words=[(t0,t1,c0,c1)])
for i, c in enumerate(cards):
    k = c["key"]
    nxt = cards[i + 1]["start"] if i + 1 < len(cards) else TOTAL
    if first_seen[k] == i:
        si, en = inks[f"si_{k}"], inks[f"en_{k}"]
        W0 = c["start"] - 0.28
        W1 = nxt - 0.42
        De = float(np.clip(0.045 * len(c["en"]), 0.9, 1.5))
        S1 = c["end"] + 0.05
        if S1 + 0.15 + De > W1:
            De = max(0.75, min(De, 0.36 * (W1 - W0)))
            S1 = W1 - De - 0.12
        # word-synced Sinhala reveal, compressed into [W0, S1]
        wt = [w[1] for w in c["words"]] + [c["end"]]
        span = (c["end"] - W0)
        f = lambda t: W0 + (t - W0) * (S1 - W0) / span
        tmap = [(W0, 0.0)]
        for j in range(len(c["words"])):
            tmap.append((f(wt[j + 1]), si.bounds[j + 1]))
        tmap[1:] = [(max(tmap[0][0] + 0.15, t), cc) for t, cc in tmap[1:]]
        writes.append(dict(ink=si, t0=W0, t1=S1, tmap=tmap, row=rows[k]))
        writes.append(dict(ink=en, t0=S1 + 0.12, t1=S1 + 0.12 + De, tmap=[(S1 + 0.12, 0), (S1 + 0.12 + De, en.w)], row=rows[k] + 1))
    else:
        si = inks[f"si_{k}"]
        ws = []
        for j, w in enumerate(c["words"]):
            t0 = w[1]; t1 = c["words"][j + 1][1] if j + 1 < len(c["words"]) else c["end"]
            ws.append((t0, t1, si.bounds[j], si.bounds[j + 1]))
        points.append(dict(ink=si, t0=c["start"] - 0.2, t1=c["end"] + 0.3, words=ws, row=rows[k]))

# repeat annotations, written just before the first repeat of each block
def first_repeat(key):
    for i, c in enumerate(cards):
        if c["key"] == key and first_seen[key] != i:
            return c["start"]


for nm, key in (("x2", "v1"), ("x3", "c1")):
    tr = first_repeat(key)
    prev_end = max(w["t1"] for w in writes if w["t1"] < tr)
    t0 = prev_end + 0.1
    t1 = min(t0 + 0.55, tr + 0.3)
    ink = inks[nm]
    writes.append(dict(ink=ink, t0=t0, t1=t1, tmap=[(t0, 0), (t1, ink.w)], row=None))
writes.sort(key=lambda w: w["t0"])


def interp_map(tmap, t):
    ts_ = [a for a, _ in tmap]; cs = [b for _, b in tmap]
    return float(np.interp(t, ts_, cs))


def pen_target(t):
    """returns (x, y, lift, mode, cam_row_y)"""
    for w in writes:
        if w["t0"] <= t <= w["t1"]:
            ink = w["ink"]; col = interp_map(w["tmap"], t)
            ci = int(np.clip(col, 0, ink.w - 1))
            wig = 2.2 * math.sin(t * 52) + 1.4 * math.sin(t * 31 + 1)
            return ink.x + col, ink.y + ink.pen_y[ci] + wig, 0.0, "write", ink.y + ink.h
    for p in points:
        if p["t0"] <= t <= p["t1"]:
            ink = p["ink"]
            col = None
            for (a, b, c0, c1) in p["words"]:
                if t < a:
                    col = c0; break
                if a <= t <= b:
                    col = c0 + (c1 - c0) * (t - a) / max(1e-3, b - a); break
            if col is None:
                col = p["words"][-1][3]
            return ink.x + col, ink.y + 2, 0.55, "point", ink.y + ink.h
    # idle: hover near the next thing to do
    nxt = [w for w in writes if w["t0"] > t] + [p for p in points if p["t0"] > t]
    if nxt:
        n = min(nxt, key=lambda z: z["t0"])
        ink = n["ink"]
        if "words" in n:
            return ink.x - 4, ink.y + 2, 1.0, "idle_point", ink.y + ink.h
        return ink.x - 4, ink.y + ink.h * 0.6, 1.0, "idle", ink.y + ink.h
    return WW * 0.95, WH * 0.9, 1.0, "away", None


# precompute pen path and camera with smoothing
N = int(round(TOTAL * FPS))
pen_xy = np.zeros((N, 2)); pen_lift = np.zeros(N); cam_y = np.zeros(N); pen_ang = np.zeros(N)
ang_now = PEN_BASE_ANG
POINT_ANG = -58.0
p = np.array(pen_target(0)[:2], float); v = np.zeros(2); lift = 1.0
scale_fit = 1.0
VIEW_H = OH / scale_fit
cy = row_y(0) + VIEW_H * 0.40
cv_ = 0.0
last_row_y = row_y(3)
for f in range(N):
    t = f / FPS
    x, y, lt, mode, ry = pen_target(t)
    tgt = np.array([x, y])
    if mode == "write":
        p = tgt.copy(); v[:] = 0
    else:
        w0 = 16.0 if mode == "point" else 11.0
        a = w0 * w0 * (tgt - p) - 2 * w0 * v
        v += a / FPS; p += v / FPS
    lift += (lt - lift) * (1 - math.exp(-14 / FPS))
    ang_t = POINT_ANG if mode in ("point", "idle_point") else PEN_BASE_ANG
    ang_now += (ang_t - ang_now) * (1 - math.exp(-7 / FPS))
    pen_xy[f] = p; pen_lift[f] = lift; pen_ang[f] = ang_now
    if ry is not None:
        last_row_y = ry
    want = last_row_y - VIEW_H * (0.08 if FMT == "916" else 0.05)
    lo = PY + VIEW_H / 2 - (40 if FMT == "169" else 0)
    hi = PY + PH - VIEW_H / 2 + 40
    want = float(np.clip(want, lo, hi))
    a = 9.0 * (want - cy) - 2 * 3.0 * cv_
    cv_ += a / FPS; cy += cv_ / FPS
    cam_y[f] = cy

# ---------------------------------------------------------------- lighting
yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
light = 1.04 - 0.16 * np.sqrt(((xx - OW * 0.3) / OW) ** 2 + ((yy - OH * 0.2) / OH) ** 2)
vd = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2)
light *= np.clip(1 - 0.22 * np.clip(vd - 0.55, 0, 1) ** 1.5, 0, 1)
warm = np.array([1.02, 1.0, 0.95], np.float32)
light3 = (light[..., None] * warm).astype(np.float32)

# ---------------------------------------------------------------- render
world = world0.copy()
done = set()
out = sys.stdout.buffer
frames = range(N) if PREVIEW is None else [int(t * FPS) for t in PREVIEW]
END_ZOOM_T = TOTAL - 4.6
for f in frames:
    t = f / FPS
    if PREVIEW is not None:
        world = world0.copy(); done = set()
    # bake finished writes
    for i, w in enumerate(writes):
        if i not in done and w["t1"] < t:
            w["ink"].draw(world); done.add(i)
    # camera: follow + handheld drift + end pull-back
    s = 1.0
    ccx = PX + PW / 2 + 6 * math.sin(t * 0.37) + 3 * math.sin(t * 0.91)
    ccy = cam_y[f] + 5 * math.sin(t * 0.29 + 1) + 2.5 * math.sin(t * 1.13)
    rot = 0.35 * math.sin(t * 0.21) + 0.15 * math.sin(t * 0.63 + 2)
    s *= 1 + 0.008 * math.sin(t * 0.17)
    if t > END_ZOOM_T:
        k = min(1, (t - END_ZOOM_T) / 3.2)
        k = k * k * (3 - 2 * k)
        s_end = min(OH / (PH + 120), OW / (PW + 160)) if FMT == "916" else 0.74
        s = s * (1 - k) + s_end * k
        ccy = ccy * (1 - k) + (PY + PH / 2) * k
    # world region the camera sees (+margin for rotation)
    hw, hh = OW / s / 2 * 1.03 + 30, OH / s / 2 * 1.03 + 30
    x0, y0 = int(max(0, ccx - hw)), int(max(0, ccy - hh))
    x1, y1 = int(min(WW, ccx + hw)), int(min(WH, ccy + hh))
    canvas = world[y0:y1, x0:x1].copy()
    off = (x0, y0)
    for i, w in enumerate(writes):
        if i not in done and w["t0"] <= t <= w["t1"]:
            w["ink"].draw(canvas, interp_map(w["tmap"], t), off)
    px, py = pen_xy[f]
    if t > END_ZOOM_T:
        k = min(1, (t - END_ZOOM_T) / 1.6)
        k = k * k * (3 - 2 * k)
        a_ = math.radians(pen_ang[f])
        px += k * 1300 * math.cos(a_); py += k * 1300 * math.sin(a_)
    blit_pen(canvas, px, py, pen_lift[f], off, pen_ang[f])
    M = cv2.getRotationMatrix2D((ccx - x0, ccy - y0), rot, s)
    M[0, 2] += OW / 2 - (ccx - x0); M[1, 2] += OH / 2 - (ccy - y0)
    frame = cv2.warpAffine(canvas, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    frame *= light3
    fade = min(1.0, t / 0.5, (TOTAL - t) / 0.9)
    frame *= max(0.0, fade)
    img = np.clip(frame, 0, 255).astype(np.uint8)
    if PREVIEW is not None:
        cv2.imwrite(f"{WORK}/prev_{FMT}_{t:06.2f}.png", img[..., ::-1])
    else:
        out.write(img.tobytes())
