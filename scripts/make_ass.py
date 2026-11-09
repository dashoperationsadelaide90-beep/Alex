"""Build an ASS lyric file (Sinhala karaoke-fill + English line) from aligned.json.

usage: make_ass.py aligned.json out.ass W H AUDIO_END TOTAL_END
"""
import sys, json

src, out = sys.argv[1], sys.argv[2]
W, H = int(sys.argv[3]), int(sys.argv[4])
AUDIO_END, TOTAL_END = float(sys.argv[5]), float(sys.argv[6])
cards = json.load(open(src))
portrait = H > W

si_size = 96 if not portrait else 84
en_size = 46 if not portrait else 44
cx = W // 2
si_y = int(H * (0.52 if not portrait else 0.50))
en_y = si_y + (22 if not portrait else 26)
margin = 90 if not portrait else 70


def ts(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Si,Noto Sans Sinhala,{si_size},&H00D6F4FF,&H70FFFFFF,&H50140A10,&H80000000,1,0,0,0,100,100,0,0,1,2.2,2,5,{margin},{margin},0,1
Style: En,Noto Sans,{en_size},&H10F2E6DC,&H10F2E6DC,&H50140A10,&H80000000,0,1,0,0,100,100,0,0,1,1.4,1.5,5,{margin},{margin},0,1
Style: Title,Noto Serif Sinhala,{int(si_size*1.05)},&H00D6F4FF,&H00D6F4FF,&H50140A10,&H80000000,1,0,0,0,100,100,0,0,1,2,2,5,{margin},{margin},0,1
Style: Sub,Noto Sans,{int(en_size*0.85)},&H20E8D8CC,&H20E8D8CC,&H50140A10,&H80000000,0,0,0,0,100,100,0,0,1,1,1,5,{margin},{margin},0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

ev = []
LEAD = 0.30
for i, c in enumerate(cards):
    start = c["start"] - LEAD
    nxt = cards[i + 1]["start"] - LEAD if i + 1 < len(cards) else None
    if nxt is not None and nxt - c["end"] < 3.5:
        end = nxt - 0.05
    else:
        end = min(c["end"] + 1.6, nxt - 0.05 if nxt else AUDIO_END)
    if nxt is None:
        end = min(AUDIO_END, c["end"] + 1.0)
    # karaoke fill per word
    parts = [f"{{\\k{int(round(LEAD*100))}}}"]
    words = c["words"]
    for j, (w, ws, we) in enumerate(words):
        nxt_w = words[j + 1][1] if j + 1 < len(words) else we
        dur = max(0.12, nxt_w - ws)
        sep = " " if j + 1 < len(words) else ""
        parts.append(f"{{\\kf{int(round(dur*100))}}}{w}{sep}")
    fade = "\\fad(260,220)"
    rise = f"\\move({cx},{si_y+14},{cx},{si_y},0,320)"
    glow = "\\blur1.2"
    ev.append(f"Dialogue: 1,{ts(start)},{ts(end)},Si,,0,0,0,,{{\\an2{rise}{fade}{glow}}}" + "".join(parts))
    erise = f"\\move({cx},{en_y+10},{cx},{en_y},60,400)"
    ev.append(f"Dialogue: 1,{ts(start+0.08)},{ts(end)},En,,0,0,0,,{{\\an8{erise}\\fad(380,220)\\blur0.6}}{c['en']}")

# closing title card after the music
t0 = AUDIO_END - 0.4
ty = int(H * (0.44 if not portrait else 0.44))
ev.append(f"Dialogue: 2,{ts(t0)},{ts(TOTAL_END)},Title,,0,0,0,,{{\\an5\\pos({cx},{ty})\\fad(700,900)\\blur1}}කුණාටු මධ්‍යයේ")
ev.append(f"Dialogue: 2,{ts(t0+0.25)},{ts(TOTAL_END)},Sub,,0,0,0,,{{\\an5\\pos({cx},{ty+int(si_size*1.05)})\\fad(800,900)}}ඔබ මා පර්වතයයි  ·  YOU ARE MY ROCK")

open(out, "w", encoding="utf-8").write(hdr + "\n".join(ev) + "\n")
print("events", len(ev))
