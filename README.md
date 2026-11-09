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

Both videos are 2:13.5 long: the full 2:09 of audio, plus a short closing title card. Audio is normalised to −14 LUFS, with a fade-out where the original recording cuts off.

## How it was made (scripts/)
1. `separate.py`: isolates the vocals from the band using Demucs `htdemucs_ft`.
2. `chunked_asr.py`: transcribes the vocals in breath-length chunks, using Whisper models fine-tuned for Sinhala.
3. `mms.py` / `score.py`: Meta MMS CTC model. It gives a romanised third transcript and scores competing lyric versions against the audio.
4. `align.py`: force-aligns the final lyrics (`lyrics/lyrics.json`) word by word, producing `lyrics/aligned.json`.
5. `background.py`: procedural dawn gradient, light rays and drifting bokeh, gently following the music's loudness.
6. `make_ass.py`: builds the karaoke-style subtitles. The final videos are then encoded with ffmpeg and libass, using the Noto Sans Sinhala font.

**To correct a lyric:** edit the text in `lyrics/lyrics.json`, then re-run `align.py`, `make_ass.py` and the ffmpeg encode.
