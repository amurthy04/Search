# Search, explained (video)

A narrated, 3Blue1Brown-style explainer of the whole pipeline, rendered with [Manim](https://www.manim.community).
Seven chapters, ~6 minutes; every number on screen comes from `bench/results/`.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python manim edge-tts soundfile
brew install cairo pango pkgconf ffmpeg                          # Manim's native deps
.venv/bin/python gen_audio.py --backend edge --out audio_edge      # narration (en-US-AndrewMultilingualNeural) + durations.json
AUDIO_DIR=audio_edge .venv/bin/python -m manim -qh --fps 30 video.py C1Intro C2Crawl C3Passages C4Embed C5Index C6Query C7Outro
printf "file '%s'\n" media/videos/video/1080p30/C*.mp4 > list.txt && ffmpeg -f concat -safe 0 -i list.txt -c copy search_explained.mp4
```

- `narration.py`: the script, one entry per narrated segment. Scene timing is driven by the measured length of each
  segment's audio, so a different voice or edit re-times the animation automatically.
- `gen_audio.py`: `--backend edge` (Microsoft neural voices, needs network) or `--backend chatterbox` (local, needs
  its own Python 3.11 env with `chatterbox-tts`).
- `video.py`: one `Scene` per chapter. `Base.narrate(key)` starts a segment's audio; `self.t(frac)` returns a run time
  as a fraction of that segment; `finish()` pads to the end of the narration.
