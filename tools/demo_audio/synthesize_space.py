"""Synthesize the demo speech with the Breeze TTS 2 Hugging Face Space.

The Space is the only synthesiser used for the shipped demo audio. It needs a
Hugging Face token in ``HF_TOKEN`` (ZeroGPU quota); the token is read from the
environment and never written anywhere.

    $env:HF_TOKEN = ...                 # PowerShell
    python synthesize_space.py --out <workspace>/tts/speech-space

Every track is written twice: the raw Space output (24 kHz) under
``<out>/raw-24k`` and a 16 kHz mono PCM16 conversion next to it. A JSON file
records the durations, the detected speech regions, the seed, the instruction
and the wall time for the reproducibility record.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from gradio_client import Client
from scipy.signal import resample_poly

SR = 16_000
SPACE = "BreezeBlue/breeze-tts-2-demo"
CFG_SCALE = 4.0
SEED = 42

JOBS = [
    (
        "zh",
        "喂，你好，能听清我说话吗？我现在在路上，周围有点吵。"
        "如果信号不太好，我们晚一点再聊。好，那就这样，回头见。",
        "一位温柔自信的年轻女性，声音清晰，语气亲切，像在安静地打电话，语速自然，停顿自然。",
    ),
    (
        "en",
        "Hi, can you hear me clearly? I'm walking down a busy street right now. "
        "If the line is bad, let's talk again later. Alright, take care, and talk soon.",
        "A warm, clear young woman speaking calmly on a phone call, natural pace, "
        "friendly and unhurried, with natural pauses.",
    ),
    (
        "zh-alt1",
        "今天下午三点开会，你记得把材料带上。路上小心，到了给我发个消息。晚饭我就不回来吃了。",
        "一位沉稳的中年男性，声音偏低，语速稍慢，语气平和。",
    ),
    (
        "zh-alt2",
        "这家店的面条味道不错，下次我们一起去试试吧。人有点多，我们换个时间再来。听说旁边还有一家甜品店。",
        "一位活泼的年轻女性，音色明亮，语速偏快，语气轻快。",
    ),
    (
        "en-alt1",
        "The weather is nice today, so I think we should walk to the park. "
        "Let me know when you are ready to leave. I will bring some water for the walk.",
        "A calm middle-aged man with a low, steady voice, speaking slowly.",
    ),
]


def speech_regions(track: np.ndarray, frame: int = 320, hop: int = 160) -> list[list[float]]:
    """Energy-based speech regions, in seconds (used for the pause metrics).

    Gaps shorter than ``MERGE_GAP`` are treated as intra-sentence pauses and
    folded into the surrounding region, so a "pause" in the metrics means a real
    conversational pause, not a stop consonant.
    """
    merge_gap = 0.35
    count = max(0, 1 + (len(track) - frame) // hop)
    levels = np.array(
        [10 * np.log10(max(float(np.mean(track[i * hop:i * hop + frame] ** 2)), 1e-14)) for i in range(count)]
    )
    active = levels > -55.0
    raw = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            raw.append([start * hop / SR, index * hop / SR])
            start = None
    if start is not None:
        raw.append([start * hop / SR, len(track) / SR])

    merged: list[list[float]] = []
    for region in raw:
        if merged and region[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = region[1]
        else:
            merged.append(region)
    return [
        [round(a, 3), round(b, 3)]
        for a, b in merged
        if b - a >= 0.12
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(r"<workspace>\tts\speech-space"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; the Space needs it for ZeroGPU quota")

    raw_dir = args.out / "raw-24k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE, verbose=False, token=token)

    meta = {}
    for name, text, instruction in JOBS:
        started = time.time()
        path = client.predict(
            text=text,
            instruction=instruction,
            cfg_scale=CFG_SCALE,
            seed=SEED,
            api_name="/voice_design",
        )
        wall = time.time() - started
        raw_path = raw_dir / f"{name}.wav"
        shutil.copy(path, raw_path)

        audio, rate = sf.read(str(raw_path), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        track = resample_poly(mono, SR, rate).astype(np.float32) if rate != SR else mono
        peak = float(np.max(np.abs(track)))
        track = (track / max(peak, 1e-9) * 0.89).astype(np.float32)
        sf.write(str(args.out / f"{name}.wav"), track, SR, subtype="PCM_16")

        meta[name] = {
            "space": SPACE,
            "api_name": "/voice_design",
            "cfg_scale": CFG_SCALE,
            "seed": SEED,
            "instruction": instruction,
            "text": text,
            "raw_sample_rate": int(rate),
            "duration_s": round(len(track) / SR, 3),
            "sentences": speech_regions(track),
            "peak": round(float(np.max(np.abs(track))), 4),
            "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(track**2)) + 1e-12)), 2),
            "wall_s": round(wall, 1),
        }
        print(f"{name}: {meta[name]['duration_s']} s, wall {wall:.1f} s, regions {meta[name]['sentences']}")

    (args.out / "speech_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(item["wall_s"] for item in meta.values())
    print(f"total wall {total:.1f} s")


if __name__ == "__main__":
    main()

