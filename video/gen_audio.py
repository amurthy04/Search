"""Generate narration audio for every segment and write durations.json.
usage: python gen_audio.py --backend edge|chatterbox --out audio_edge
"""
import argparse, asyncio, json, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
from narration import SEGMENTS

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="edge")
ap.add_argument("--out", default="audio_edge")
ap.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
ap.add_argument("--exaggeration", type=float, default=0.5)
ap.add_argument("--cfg", type=float, default=0.5)
ap.add_argument("--only", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SR = 24000
PAUSE = int(0.28 * SR)

def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", t) if s.strip()]

def norm(x):
    x = x.astype(np.float32)
    peak = np.abs(x).max() or 1.0
    return x / peak * 0.85

if a.backend == "edge":
    import edge_tts, subprocess, tempfile
    async def one(text, path):
        tmp = path + ".mp3"
        for attempt in range(6):
            try:
                await edge_tts.Communicate(text, a.voice, rate="-3%").save(tmp); break
            except Exception as e:
                print("retry", attempt, type(e).__name__, flush=True); await asyncio.sleep(2 + 3 * attempt)
        else:
            raise SystemExit("edge-tts kept failing for: " + text[:60])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ar", str(SR), "-ac", "1", path], check=True)
        os.remove(tmp)
    async def main():
        for k, t in SEGMENTS.items():
            if a.only and k not in a.only.split(","): continue
            if os.path.exists(f"{a.out}/{k}.wav"): continue
            await one(t, f"{a.out}/{k}.wav"); print("edge", k, flush=True)
    asyncio.run(main())
else:
    import torch, torchaudio as ta
    _load = torch.load
    def _patched(*ar, **k):
        k.setdefault("map_location", torch.device("cpu")); return _load(*ar, **k)
    torch.load = _patched
    import perth
    class _NoWM:
        def apply_watermark(self, wav, sample_rate=None): return wav
    perth.PerthImplicitWatermarker = _NoWM
    from chatterbox.tts import ChatterboxTTS
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = ChatterboxTTS.from_pretrained(device=dev)
    torch.manual_seed(7)
    for k, t in SEGMENTS.items():
        if a.only and k not in a.only.split(","): continue
        parts = []
        for s in sentences(t):
            wav = m.generate(s, exaggeration=a.exaggeration, cfg_weight=a.cfg).squeeze(0).cpu().numpy()
            # trim leading/trailing near-silence
            idx = np.where(np.abs(wav) > 0.01)[0]
            if len(idx): wav = wav[max(0, idx[0] - int(0.05 * m.sr)): idx[-1] + int(0.12 * m.sr)]
            parts.append(wav); parts.append(np.zeros(PAUSE, dtype=np.float32))
        audio = norm(np.concatenate(parts[:-1]))
        if m.sr != SR:
            audio = ta.functional.resample(torch.from_numpy(audio), m.sr, SR).numpy()
        sf.write(f"{a.out}/{k}.wav", audio, SR); print("chatterbox", k, round(len(audio) / SR, 1), "s", flush=True)

dur = {}
for k in SEGMENTS:
    p = f"{a.out}/{k}.wav"
    if os.path.exists(p):
        info = sf.info(p); dur[k] = round(info.frames / info.samplerate, 3)
json.dump(dur, open(f"{a.out}/durations.json", "w"), indent=1)
print("total narration", round(sum(dur.values()), 1), "s over", len(dur), "segments")
