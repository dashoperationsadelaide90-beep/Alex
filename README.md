# Worship lyric video: කුණාටු මධ්‍යයේ (ඔබ මා පර්වතයයි)

This lyric video was built from a live recording of the song sung at *Territorial Command Sunday*. The original footage is removed and only the soundtrack is kept. The new visuals are an animated background, with Sinhala lyrics that fill in word by word as they're sung and an English translation underneath.

## Deliverables
| File | Use |
|---|---|
| `output/kunatu-madhyaye_16x9_1080p.mp4` | YouTube, Facebook, church screens (1920×1080) |
| `output/kunatu-madhyaye_9x16_1080p.mp4` | Reels, Shorts, TikTok, Status (1080×1920) |
| `subtitles/kunatu-madhyaye.si.srt` / `.en.srt` | Separate caption tracks (e.g. YouTube CC) |
| `subtitles/*.ass` | The styled, animated subtitles that are burned into the videos |
| `lyrics/LYRICS_REVIEW.md` | **Lyrics with confidence notes. Please review before publishing.** |

| `output/kunatu-madhyaye_paper_9x16_1080p.mp4` | **"Written on paper" style**, vertical (1080×1920) |
| `output/kunatu-madhyaye_paper_16x9_1080p.mp4` | **"Written on paper" style**, landscape (1920×1080), on a wooden desk |

### "Written on paper" style
A pen writes each Sinhala line on ruled notebook paper as it is sung, followed by the English translation in smaller handwriting. The camera follows the pen down the page. For repeated sections, the pen writes "x 2" or "x 3" beside the block (like the reference video), then points at each word from above as it is sung again. At the end, the camera pulls back to show the finished page.

- Sinhala handwriting: **Yaldevi** (a monoline, ballpoint-like font). English: **Patrick Hand**. Both are under the SIL Open Font License (see `fonts/`).
- Each line gets a slight random wobble, tilt and ink texture, so it doesn't look typeset.
- Renderer: `scripts/render_paper.py`. It uses the same word timings (`lyrics/aligned.json`) as the first version.

Every video is 2:13.5 long: the full 2:09 of audio, plus a short ending. Audio is normalised to −14 LUFS, with a fade-out where the original recording cuts off.

## How it was made (scripts/)
1. `separate.py`: isolates the vocals from the band using Demucs `htdemucs_ft`.
2. `chunked_asr.py`: transcribes the vocals in breath-length chunks, using Whisper models fine-tuned for Sinhala.
3. `mms.py` / `score.py`: Meta MMS CTC model. It gives a romanised third transcript and scores competing lyric versions against the audio.
4. `align.py`: force-aligns the final lyrics (`lyrics/lyrics.json`) word by word, producing `lyrics/aligned.json`.
5. `background.py`: procedural dawn gradient, light rays and drifting bokeh, gently following the music's loudness.
6. `make_ass.py`: builds the karaoke-style subtitles. The final videos are then encoded with ffmpeg and libass, using the Noto Sans Sinhala font.

**To correct a lyric:** edit the text in `lyrics/lyrics.json`, then re-run `align.py`. After that, re-run `make_ass.py` and the ffmpeg encode for the first style, or `render_paper.py` for the paper style.
