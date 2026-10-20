"""Build the CleanField demo audio set.

Pipeline
--------
1. Read the 24 kHz Kokoro speech tracks, resample to 16 kHz mono.
2. Pick an eight-second window per language that contains speech *and* pauses.
3. Synthesize five noise scenes (white, pink, babble, cafe, street) at a
   realistic SNR and mix them with the speech.
4. Run the offline denoiser (`local-denoiser/process_wav.py`, Rust backend) on
   every noisy clip.
5. Build the echo scene: the Chinese clip played through a synthetic room
   impulse response plus a gated near-end talker, processed by
   `cf-echo-pipeline`.
6. Measure RMS before/after, pause suppression and speech preservation.

Run from anywhere with the Python that has numpy/scipy/soundfile installed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

SR = 16_000
CLIP_SECONDS = 8.0
CLIP = int(SR * CLIP_SECONDS)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSETS = REPO / "assets" / "audio"
WORK = Path(os.environ.get("CF_DEMO_WORK", r"<workspace>\tts\build"))
SPEECH = Path(os.environ.get("CF_DEMO_SPEECH", r"<workspace>\tts\speech-space"))
DENOISER = Path(r"<workspace>\local-denoiser")
ECHO_DRIVER = Path(os.environ.get("CF_ECHO_DRIVER", r"<workspace>\tts\cargo-target\release\echo_driver.exe"))
ECHO_MODEL = REPO / "models" / "echo-default.cfwm"

WORK.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261020)

# SNR per scene, in dB, referenced to the active-speech RMS of the clip.
SNR_DB = {"white": 12.0, "pink": 10.0, "babble": 9.0, "cafe": 8.0, "street": 10.0}


# --------------------------------------------------------------------------- IO
def read_16k(path: Path) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SR:
        mono = resample_poly(mono, SR, rate).astype(np.float32)
    return mono


def write_wav(path: Path, samples: np.ndarray) -> dict:
    import soundfile as sf

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clipped = int(np.sum(np.abs(samples) >= 1.0))
    sf.write(str(path), np.clip(samples, -1.0, 1.0), SR, subtype="PCM_16")
    return {"peak": round(peak, 5), "clipped_samples": clipped}


def dbfs(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-12))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


# ----------------------------------------------------------------- speech clips
def load_speech():
    meta = json.loads((SPEECH / "speech_meta.json").read_text(encoding="utf-8"))
    tracks = {}
    for name in meta:
        tracks[name] = read_16k(SPEECH / f"{name}.wav")
        # Speech bounds were measured on the 24 kHz file: rescale to 16 kHz.
        meta[name]["sentences_16k"] = [
            [round(a * SR, 3), round(b * SR, 3)] for a, b in meta[name]["sentences"]
        ]
    return meta, tracks


def choose_window(sentences, total: int) -> tuple[int, list[tuple[int, int]]]:
    """Pick the eight-second window with the most pause time and enough speech."""
    best = None
    step = int(0.05 * SR)
    for start in range(0, max(1, total - CLIP + 1), step):
        end = start + CLIP
        pauses = []
        cursor = start
        for a, b in sentences:
            a, b = int(a), int(b)
            if b <= start or a >= end:
                continue
            if a > cursor:
                pauses.append((cursor, min(a, end)))
            cursor = max(cursor, min(b, end))
        if cursor < end:
            pauses.append((cursor, end))
        pauses = [(a, b) for a, b in pauses if b - a >= int(0.45 * SR)]
        if not pauses:
            continue
        speech_total = CLIP - sum(b - a for a, b in pauses)
        fraction = speech_total / CLIP
        if not (0.55 <= fraction <= 0.9):
            continue
        pause_total = sum(b - a for a, b in pauses)
        score = (pause_total, -start)
        if best is None or score > best[0]:
            best = (score, start, pauses)
    if best is None:
        raise RuntimeError("no eight-second window with speech and pauses")
    return best[1], best[2]


# ---------------------------------------------------------------------- noises
def pink_noise(n: int, rng) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    freqs[0] = freqs[1]
    shaped = spectrum / np.sqrt(freqs)
    out = np.fft.irfft(shaped, n)
    return (out / np.max(np.abs(out))).astype(np.float32)


def brown_noise(n: int, rng) -> np.ndarray:
    steps = rng.standard_normal(n)
    out = np.cumsum(steps)
    out -= lfilter([1 - 0.995], [1, -0.995], out)[0]
    return (out / np.max(np.abs(out))).astype(np.float32)


def bandpass(x: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def lowpass(x: np.ndarray, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, high, btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def highpass(x: np.ndarray, low: float, order: int = 4) -> np.ndarray:
    sos = butter(order, low, btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def peak_norm(x: np.ndarray) -> np.ndarray:
    return (x / max(float(np.max(np.abs(x))), 1e-9)).astype(np.float32)


def make_babble(tracks, rng, n: int) -> np.ndarray:
    """Eight overlapping talkers, each reverberant and band-limited."""
    names = ["zh-alt1", "zh-alt2", "en-alt1", "zh", "en"]
    mix = np.zeros(n, dtype=np.float32)
    for index in range(8):
        source = tracks[names[index % len(names)]]
        if len(source) <= n:
            source = np.pad(source, (0, n - len(source)))
        offset = int(rng.integers(0, max(1, len(source) - n)))
        segment = source[offset:offset + n].copy()
        rate = [92, 96, 100, 104, 108, 94, 102, 106][index]
        segment = resample_poly(segment, rate, 100).astype(np.float32)[:n]
        segment = np.pad(segment, (0, max(0, n - len(segment))))
        # Small room: smears intelligibility the way a real crowd does.
        taps = 220
        rir = rng.standard_normal(taps).astype(np.float32) * np.exp(-np.arange(taps) / 40.0)
        segment = fftconvolve(segment, rir)[:n].astype(np.float32)
        segment = bandpass(segment, 200, 3500)
        segment /= max(rms(segment), 1e-9)
        # Slow level drift, as tables of people lean in and out.
        drift = lowpass(rng.standard_normal(n).astype(np.float32), 0.8)
        drift = 0.55 + 0.45 * np.abs(drift) / max(float(np.max(np.abs(drift))), 1e-9)
        mix += segment * drift
    return peak_norm(mix)


def clatter(n: int, rng) -> np.ndarray:
    """Cup/silverware transients for the cafe scene."""
    out = np.zeros(n, dtype=np.float32)
    count = 14
    for _ in range(count):
        start = int(rng.integers(0, n - int(0.3 * SR)))
        length = int(rng.uniform(0.05, 0.22) * SR)
        burst = rng.standard_normal(length).astype(np.float32)
        burst *= np.exp(-np.linspace(0, 6, length)).astype(np.float32)
        burst = bandpass(burst, float(rng.uniform(1400, 2600)), float(rng.uniform(4200, 7200)))
        out[start:start + length] += burst * float(rng.uniform(0.3, 1.0))
    for _ in range(3):  # cup clinks: damped partials
        start = int(rng.integers(0, n - SR))
        t = np.arange(int(0.18 * SR)) / SR
        tone = sum(
            np.sin(2 * np.pi * f * t) * np.exp(-t * 30) * a
            for f, a in [(float(rng.uniform(2400, 3200)), 1.0), (float(rng.uniform(4200, 5200)), 0.6)]
        )
        out[start:start + len(t)] += 0.25 * tone.astype(np.float32)
    return out


def street_events(n: int, rng) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for centre in (int(2.4 * SR), int(6.1 * SR)):  # car pass-bys
        length = int(2.6 * SR)
        start = max(0, min(n - length, centre - length // 2))
        noise = bandpass(rng.standard_normal(length).astype(np.float32), 250, 2200)
        t = np.linspace(-1, 1, length)
        swell = np.exp(-(t**2) * 3.0)
        out[start:start + length] += noise * swell.astype(np.float32) * 0.9
    start = int(4.6 * SR)  # a short horn
    t = np.arange(int(0.42 * SR)) / SR
    horn = (
        np.sin(2 * np.pi * 415 * t) + 0.7 * np.sin(2 * np.pi * 554 * t) + 0.3 * np.sin(2 * np.pi * 830 * t)
    )
    horn *= np.minimum(1, np.minimum(t / 0.03, (0.42 - t) / 0.06))
    out[start:start + len(t)] += 0.22 * horn.astype(np.float32)
    return out


def scene_noise(kind: str, tracks, rng, n: int) -> np.ndarray:
    if kind == "white":
        return peak_norm(rng.standard_normal(n).astype(np.float32))
    if kind == "pink":
        return peak_norm(pink_noise(n, rng))
    if kind == "babble":
        return make_babble(tracks, rng, n)
    if kind == "cafe":
        base = make_babble(tracks, rng, n)
        room = lowpass(pink_noise(n, rng), 600)
        mix = 0.55 * base + 0.25 * peak_norm(room) + 0.9 * peak_norm(clatter(n, rng))
        return peak_norm(bandpass(mix, 120, 7200))
    if kind == "street":
        rumble = lowpass(brown_noise(n, rng), 150)
        hiss = 0.25 * peak_norm(lowpass(pink_noise(n, rng), 2500))
        mix = 1.0 * peak_norm(rumble) + hiss + peak_norm(street_events(n, rng))
        return peak_norm(mix)
    raise ValueError(kind)


# ------------------------------------------------------------------- denoiser
def run_denoiser(src: Path, dst: Path) -> dict:
    if dst.exists():
        dst.unlink()
    env = dict(os.environ)
    env["AINS_BACKEND"] = "rust"
    env.setdefault("PYTHONPATH", str(DENOISER))
    result = subprocess.run(
        [sys.executable, str(DENOISER / "process_wav.py"), str(src), str(dst)],
        cwd=str(DENOISER),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"denoiser failed on {src.name}: {result.stderr[-2000:]}")
    info = json.loads(result.stdout) if result.stdout.strip() else {}
    return info


# ---------------------------------------------------------------- measurements
def frame_db(x: np.ndarray, frame: int = 320, hop: int = 160) -> np.ndarray:
    count = max(0, 1 + (len(x) - frame) // hop)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        block = x[i * hop:i * hop + frame]
        out[i] = 10 * np.log10(max(float(np.mean(block.astype(np.float64) ** 2)), 1e-14))
    return out


def frame_index_to_mask(count: int, runs) -> np.ndarray:
    mask = np.zeros(count, dtype=bool)
    for a, b in runs:
        mask[max(0, a // 160):min(count, -(-b // 160))] = True
    return mask


def runs_from_mask(mask: np.ndarray, minimum: int, invert: bool = False) -> list[tuple[int, int]]:
    want = ~mask if invert else mask
    runs = []
    start = None
    for i, value in enumerate(want):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= minimum:
                runs.append((start, i))
            start = None
    if start is not None and len(want) - start >= minimum:
        runs.append((start, len(want)))
    return runs


def masked_db(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(10 * np.log10(np.mean(10 ** (values[mask] / 10))))


def measure(before: np.ndarray, after: np.ndarray, clean: np.ndarray, active: np.ndarray) -> dict:
    fb = frame_db(before)
    fa = frame_db(after)
    fc = frame_db(clean)
    count = min(len(fb), len(fa), len(fc))
    fb, fa, fc = fb[:count], fa[:count], fc[:count]
    active = active[:count]
    speech_mask = active
    pause_mask = ~active
    pause_runs = runs_from_mask(pause_mask, 5)
    pause_mask = np.zeros(count, dtype=bool)
    for a, b in pause_runs:
        pause_mask[a:b] = True
    result = {
        "rms_before_dbfs": round(dbfs(rms(before)), 2),
        "rms_after_dbfs": round(dbfs(rms(after)), 2),
        "overall_delta_db": round(dbfs(rms(after)) - dbfs(rms(before)), 2),
        "pause_seconds": round(float(pause_mask.sum()) * 0.01, 2),
        "speech_seconds": round(float(speech_mask.sum()) * 0.01, 2),
    }
    before_pause = masked_db(fb, pause_mask)
    after_pause = masked_db(fa, pause_mask)
    result["pause_before_dbfs"] = round(before_pause, 2) if before_pause is not None else None
    result["pause_after_dbfs"] = round(after_pause, 2) if after_pause is not None else None
    result["pause_suppression_db"] = (
        round(after_pause - before_pause, 2) if before_pause is not None and after_pause is not None else None
    )
    before_speech = masked_db(fb, speech_mask)
    after_speech = masked_db(fa, speech_mask)
    clean_speech = masked_db(fc, speech_mask)
    result["speech_before_dbfs"] = round(before_speech, 2) if before_speech is not None else None
    result["speech_after_dbfs"] = round(after_speech, 2) if after_speech is not None else None
    result["speech_preservation_db"] = (
        round(after_speech - clean_speech, 2)
        if after_speech is not None and clean_speech is not None
        else None
    )
    return result


# ------------------------------------------------------------------- echo set
def make_rir(taps: int = 512, direct: int = 160) -> np.ndarray:
    h = np.zeros(taps, dtype=np.float64)
    h[direct] = 1.0
    for delay, gain in [(23, 0.55), (41, -0.42), (77, 0.33), (113, -0.27), (157, 0.21), (211, -0.16)]:
        if direct + delay < taps:
            h[direct + delay] += gain
    state = 0x5EED1234
    for n in range(direct + 64, taps):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        noise = ((state >> 9) & 0x7FFFFF) / 4194304.0 - 1.0
        distance = n - direct
        h[n] += 0.12 * np.exp(-distance / 90.0) * noise
    return (h / np.sqrt(np.sum(h**2))).astype(np.float32)


def gate(length: int, intervals) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    fade = int(0.01 * SR)
    for start_s, end_s in intervals:
        start, end = int(start_s * SR), int(end_s * SR)
        for n in range(start, end):
            rising = n - start
            falling = end - 1 - n
            ramp_in = 0.5 - 0.5 * np.cos(np.pi * rising / fade) if rising < fade else 1.0
            ramp_out = 0.5 - 0.5 * np.cos(np.pi * falling / fade) if falling < fade else 1.0
            out[n] = min(ramp_in, ramp_out)
    return out


def build_echo(tracks, zh_clip: np.ndarray, near_source: np.ndarray) -> dict:
    reference = zh_clip.astype(np.float32)
    rir = make_rir()
    echo = fftconvolve(reference, rir)[:CLIP].astype(np.float32)
    gate_values = gate(CLIP, [(2.6, 4.5), (5.6, 7.4)])
    near = near_source[:CLIP].copy()
    if len(near) < CLIP:
        near = np.pad(near, (0, CLIP - len(near)))
    near *= gate_values

    echo_gain = 0.32
    near_gain = echo_gain * rms(echo) / max(rms(near), 1e-9) * 1.4
    echo_component = (echo * echo_gain).astype(np.float32)
    near_component = (near * near_gain).astype(np.float32)
    mic = (echo_component + near_component).astype(np.float32)
    scale = min(1.0, 0.85 / max(float(np.max(np.abs(mic))), 1e-9))
    mic = (mic * scale).astype(np.float32)
    reference = (reference * scale).astype(np.float32)
    echo_component = (echo_component * scale).astype(np.float32)
    near_component = (near_component * scale).astype(np.float32)

    before_path = ASSETS / "echo-zh-before.wav"
    ref_path = WORK / "echo-zh-reference.wav"
    mic_path = WORK / "echo-zh-mic.wav"
    info_before = write_wav(before_path, mic)
    write_wav(ref_path, reference)
    write_wav(mic_path, mic)

    after_path = ASSETS / "echo-zh-after.wav"
    if after_path.exists():
        after_path.unlink()
    result = subprocess.run(
        [str(ECHO_DRIVER), str(ECHO_MODEL), str(mic_path), str(ref_path), str(after_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"echo driver failed: {result.stderr[-2000:]}")
    print("echo driver:", result.stdout.strip())

    after = read_16k(after_path)[:CLIP]
    active = gate_values > 0.5
    count = len(frame_db(mic))
    speech_mask = frame_index_to_mask(count, runs_from_mask(active, 5))
    metrics = measure(mic, after, near_component, speech_mask)
    metrics.update(
        {
            "pair": "echo-zh",
            "noise": "echo",
            "language": "zh",
            "before_file": before_path.name,
            "after_file": after_path.name,
            "before_peak": info_before["peak"],
            "echo_driver": result.stdout.strip(),
        }
    )
    return metrics


# ---------------------------------------------------------------------- report
def markdown_table(rows: list[dict]) -> str:
    head = (
        "| Pair | SNR (dB) | RMS before (dBFS) | RMS after (dBFS) | Pause suppression (dB) "
        "| Speech change (dB) |"
    )
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [head, sep]
    for row in rows:
        lines.append(
            "| {pair} | {snr} | {rb} | {ra} | {ps} | {sp} |".format(
                pair=row["pair"],
                snr=row.get("snr_db", "n/a"),
                rb=row.get("rms_before_dbfs"),
                ra=row.get("rms_after_dbfs"),
                ps=row.get("pause_suppression_db"),
                sp=row.get("speech_preservation_db"),
            )
        )
    return "\n".join(lines)


def main() -> None:
    meta, tracks = load_speech()
    report: list[dict] = []

    clips = {}
    for lang in ("zh", "en"):
        start, pauses = choose_window(meta[lang]["sentences_16k"], len(tracks[lang]))
        clip = tracks[lang][start:start + CLIP]
        frame_count = len(frame_db(clip))
        speech_mask_frames = np.zeros(frame_count, dtype=bool)
        active_samples = np.zeros(CLIP, dtype=bool)
        for a, b in meta[lang]["sentences_16k"]:
            a, b = int(a), int(b)
            if b <= start or a >= start + CLIP:
                continue
            lo, hi = max(a, start) - start, min(b, start + CLIP) - start
            speech_mask_frames[frame_index_to_mask(frame_count, [(lo, hi)])] = True
            active_samples[lo:hi] = True
        clips[lang] = {
            "audio": clip.astype(np.float32),
            "active": speech_mask_frames,
            "active_samples": active_samples,
            "pauses": pauses,
            "start_s": round(start / SR, 2),
        }
        print(f"{lang}: window {start / SR:.2f}-{(start + CLIP) / SR:.2f} s, pauses {[(round(a / SR, 2), round(b / SR, 2)) for a, b in pauses]}")

    for lang in ("zh", "en"):
        clean = clips[lang]["audio"]
        active_samples = clips[lang]["active_samples"]
        for kind in ("white", "pink", "babble", "cafe", "street"):
            noise = scene_noise(kind, tracks, RNG, CLIP)
            speech_active = float(np.sqrt(np.mean(clean[active_samples] ** 2)))
            gain = speech_active / max(rms(noise), 1e-9) * 10 ** (-SNR_DB[kind] / 20)
            mix = clean + noise * gain
            peak = float(np.max(np.abs(mix)))
            if peak > 0.95:
                mix = mix * (0.95 / peak)
            pair = f"denoise-{lang}-{kind}"
            before_path = ASSETS / f"{pair}-before.wav"
            after_path = ASSETS / f"{pair}-after.wav"
            info = write_wav(before_path, mix)
            if after_path.exists():
                after_path.unlink()
            denoise_info = run_denoiser(before_path, after_path)
            after = read_16k(after_path)[:CLIP]
            metrics = measure(mix, after, clean, clips[lang]["active"])
            measured_snr = 10 * np.log10(
                float(np.mean(clean[active_samples] ** 2))
                / max(float(np.mean((noise * gain)[active_samples] ** 2)), 1e-12)
            )
            metrics.update(
                {
                    "pair": pair,
                    "language": lang,
                    "noise": kind,
                    "snr_db": round(measured_snr, 2),
                    "before_file": before_path.name,
                    "after_file": after_path.name,
                    "before_peak": info["peak"],
                    "before_clipped": info["clipped_samples"],
                    "rtf": round(float(denoise_info.get("realTimeFactor", 0)), 4),
                }
            )
            report.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False))

    # Near-end talker: a second Chinese voice, joined seamlessly to fill 8 s.
    near_source = np.concatenate(
        [tracks["zh-alt1"], np.zeros(int(0.3 * SR), dtype=np.float32), tracks["zh-alt2"]]
    )[:CLIP]
    echo_metrics = build_echo(tracks, clips["zh"]["audio"], near_source)
    report.append(echo_metrics)

    (WORK / "measurements.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + markdown_table(report))


if __name__ == "__main__":
    main()

