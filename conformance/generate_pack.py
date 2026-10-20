#!/usr/bin/env python3
"""Generate the CleanField conformance pack.

This is a build-time tool, not part of the runtime.  It creates three
deterministic 40-frame feature sequences, runs them through the Python
reference engine (`tools/engine-v64.py`, the blob-only numpy implementation of
the voice model), and writes the inputs and the expected outputs as raw
little-endian float32 binaries plus a `manifest.json` with the SHA-256 of every
file.

The engine boundary is the *feature frame*: one call consumes 513 f32 values
and produces 1026 f32 values.  Conformance is defined on those values, not on
PCM, because the STFT front-end is a separate adapter and is not bit-portable
across FFT and `log` implementations (see `CONTRACT.md`).

Determinism
-----------
All three sequences are pure functions of a documented seed and formula, and
the resulting bytes are pinned by SHA-256 in the manifest.  NumPy's
`default_rng` (PCG64) is used; the exact recipes are reproduced in
`manifest.json` under `sets[].sequence` and in `README.md`.  Re-running this
script on the same engine and NumPy version must yield the same SHA-256 values.

Usage
-----
    python conformance/generate_pack.py            # writes ./sets + manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                  # cleanfield/
REPO = ROOT.parent                  # repository root
DEFAULT_ENGINE = REPO / "tools" / "engine-v64.py"

INPUT_WIDTH = 513
OUTPUT_WIDTH = 1026
FRAMES = 40
SAMPLE_RATE = 16_000
HOP = 256
WINDOW = 768
LOOKAHEAD_HOPS = 2


# ---------------------------------------------------------------------------
# deterministic feature sequences
# ---------------------------------------------------------------------------
def quiet_speech(frames: int = FRAMES, seed: int = 20261020) -> np.ndarray:
    """Quiet voiced speech: harmonic comb + three formants, with pauses.

    The values are log-power features, the same quantity the front-end feeds
    the engine: ``ln(power + 1e-20)`` for bins 0..512 of a 1024-point FFT at
    16 kHz.  The recipe is deliberately written against the *feature* domain
    so it does not depend on an FFT implementation.  It produces values in the
    same range as real 16 kHz speech (about -18 .. +3), scaled down by 12 dB to
    sit in the "quiet" region.
    """
    rng = np.random.default_rng(seed)
    hz_per_bin = SAMPLE_RATE / 1024.0
    freqs = np.arange(INPUT_WIDTH, dtype=np.float64) * hz_per_bin
    out = np.empty((frames, INPUT_WIDTH), dtype=np.float32)

    # Fixed formant table (centre Hz, bandwidth Hz, gain).
    formants = ((520.0, 110.0, 1.00), (1480.0, 190.0, 0.62), (2520.0, 280.0, 0.36))
    formant = np.zeros(INPUT_WIDTH, dtype=np.float64)
    for fc, bw, gain in formants:
        formant += gain * np.exp(-0.5 * ((freqs - fc) / bw) ** 2)

    for t in range(frames):
        # Syllable envelope: 7 voiced frames, 3 quiet frames, repeated.
        gate = 1.0 if (t % 10) < 7 else 0.12
        level_db = -40.0 + 6.0 * np.sin(2.0 * np.pi * t / 13.0) + 2.0 * rng.standard_normal()
        amplitude = 10.0 ** (level_db / 20.0) * gate
        f0 = 115.0 + 30.0 * np.sin(2.0 * np.pi * t / 40.0)

        comb = np.zeros(INPUT_WIDTH, dtype=np.float64)
        for k in range(1, 41):
            fk = k * f0
            if fk >= SAMPLE_RATE / 2.0:
                break
            comb += (1.0 / k ** 0.8) * np.exp(-0.5 * ((freqs - fk) / 24.0) ** 2)

        # sqrt() converts the magnitude-shaped comb/formant product to the
        # power domain; the 1e-9 floor is the numerical noise floor.
        shape = (0.04 + formant) * (0.02 + comb) + 1e-6
        power = (amplitude * amplitude) * shape + 1e-9
        out[t] = np.log(power).astype(np.float32)
    return out


def steady_noise(frames: int = FRAMES, seed: int = 20261021) -> np.ndarray:
    """Stationary noise: fixed spectral tilt, small level wander, no transients.

    This is the steady-state case: every frame has the same distribution, so a
    streaming error shows up as a growing mismatch rather than a one-frame
    glitch.
    """
    rng = np.random.default_rng(seed)
    hz_per_bin = SAMPLE_RATE / 1024.0
    freqs = np.arange(INPUT_WIDTH, dtype=np.float64) * hz_per_bin
    tilt = -3.0 * np.log1p(freqs / 1000.0)          # about -9 dB at 8 kHz
    level = -9.0 + 0.35 * rng.standard_normal((frames, 1))
    texture = 0.8 * rng.standard_normal((frames, INPUT_WIDTH))
    return (level + tilt[None, :] + texture).astype(np.float32)


def extreme_amplitude(frames: int = FRAMES) -> np.ndarray:
    """Extreme amplitudes: the engine's own clamp-exercising sequence.

    This is byte-for-byte the sequence produced by
    ``tools/engine-v64.py`` ``clamp_frames(40)`` (seed 777): mixed scales from
    1 to 1000 with periodic full-scale frames.  It drives the int16 input
    quantiser, the LSTM state clamp and the int16 MatMul clamp into saturation,
    which is where a reimplementation is most likely to diverge.
    """
    rng = np.random.default_rng(777)
    base = rng.normal(0, 1, (frames, INPUT_WIDTH)).astype(np.float32)
    scales = np.array([1.0, 30.0, 300.0, 3.0, 1000.0, 10.0], np.float32)
    out = (base * scales[np.arange(frames) % len(scales)][:, None]).astype(np.float32)
    out[::9] = np.float32(500.0)
    return out


SETS = (
    {
        "name": "quiet-speech",
        "description": "Quiet voiced speech: harmonic comb with a three-formant "
                       "envelope, syllable gating and a -40 dB level; 40 frames.",
        "generator": quiet_speech,
        "sequence": {
            "function": "quiet_speech(frames=40, seed=20261020)",
            "rng": "numpy.random.default_rng(20261020) (PCG64), "
                   "draw order: one standard_normal() per frame for the level",
            "formula": "f0 = 115 + 30*sin(2*pi*t/40) Hz; 40 harmonics with "
                       "amplitude k**-0.8 and a 24 Hz Gaussian width; formants "
                       "(520,110,1.00), (1480,190,0.62), (2520,280,0.36); "
                       "amplitude = 10**(level_db/20), level_db = -40 + "
                       "6*sin(2*pi*t/13) + 2*N(0,1); gate 1.0 for t%10 < 7 "
                       "else 0.12; power = A^2*((0.04+formant)*(0.02+comb)+1e-6)"
                       "+1e-9; feature = ln(power)",
            "note": "Bins are 15.625 Hz apart (16 kHz / 1024); bin 512 is Nyquist.",
        },
    },
    {
        "name": "steady-noise",
        "description": "Stationary noise: -3*ln(1+f/1000) tilt, a slow overall "
                       "level, and independent per-bin texture; 40 frames.",
        "generator": steady_noise,
        "sequence": {
            "function": "steady_noise(frames=40, seed=20261021)",
            "rng": "numpy.random.default_rng(20261021) (PCG64), draw order: "
                   "40 level samples, then 40*513 texture samples",
            "formula": "feature[t,k] = -9 + 0.35*N(0,1)[t] "
                       "- 3*ln(1 + f_k/1000) + 0.8*N(0,1)[t,k], f_k = 15.625*k",
        },
    },
    {
        "name": "extreme-amplitude",
        "description": "Extreme amplitude: the engine clamp sequence (scales "
                       "1..1000, full-scale frames every 9th frame); 40 frames.",
        "generator": extreme_amplitude,
        "sequence": {
            "function": "extreme_amplitude(frames=40) == tools/engine-v64.py "
                        "clamp_frames(40)",
            "rng": "numpy.random.default_rng(777) (PCG64), 40*513 standard_normal",
            "formula": "base[t,k] = N(0,1); scale = [1,30,300,3,1000,10][t % 6]; "
                       "feature = base*scale; every 9th frame is the constant 500.0",
        },
    },
)


# ---------------------------------------------------------------------------
# engine plumbing
# ---------------------------------------------------------------------------
def load_engine(path: Path):
    spec = importlib.util.spec_from_file_location("cleanfield_engine_v64", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load the reference engine from %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, values: np.ndarray) -> int:
    """Write frame-major little-endian float32; return the byte count."""
    data = np.ascontiguousarray(values, dtype="<f4").tobytes()
    path.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the CleanField conformance pack.")
    parser.add_argument("--pack", type=Path, default=HERE, help="output directory")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE,
                        help="path to tools/engine-v64.py")
    args = parser.parse_args()

    engine_module = load_engine(args.engine)
    engine = engine_module.Engine()
    model_sha = engine_module.MODELS["v6_4"]["sha256"]

    pack = args.pack
    sets_dir = pack / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    manifest_sets = []
    for entry in SETS:
        frames = entry["generator"]()
        if frames.shape != (FRAMES, INPUT_WIDTH):
            raise SystemExit("%s: unexpected shape %r" % (entry["name"], frames.shape))
        if not np.all(np.isfinite(frames)):
            raise SystemExit("%s: non-finite input features" % entry["name"])

        engine.reset()
        outputs = np.empty((FRAMES, OUTPUT_WIDTH), dtype=np.float32)
        for t in range(FRAMES):
            prediction, _ = engine.process(frames[t], keep=False)
            outputs[t] = prediction
        if not np.all(np.isfinite(outputs)):
            raise SystemExit("%s: non-finite expected outputs" % entry["name"])

        out_dir = sets_dir / entry["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        input_path = out_dir / "input.f32.bin"
        output_path = out_dir / "expected.f32.bin"
        input_bytes = write_f32(input_path, frames)
        output_bytes = write_f32(output_path, outputs)

        manifest_sets.append({
            "name": entry["name"],
            "description": entry["description"],
            "frames": FRAMES,
            "input_width": INPUT_WIDTH,
            "output_width": OUTPUT_WIDTH,
            "input_file": "sets/%s/input.f32.bin" % entry["name"],
            "output_file": "sets/%s/expected.f32.bin" % entry["name"],
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "input_sha256": sha256(input_path),
            "output_sha256": sha256(output_path),
            "input_range": [float(frames.min()), float(frames.max())],
            "output_range": [float(outputs.min()), float(outputs.max())],
            "sequence": entry["sequence"],
        })
        print("%-18s %4d frames  %7d + %7d values  input[%8.3f, %8.3f]  "
              "output[%8.3f, %8.3f]"
              % (entry["name"], FRAMES, frames.size, outputs.size,
                 frames.min(), frames.max(), outputs.min(), outputs.max()))

    manifest = {
        "format": "cleanfield-conformance/1",
        "engine": {
            "implementation": "tools/engine-v64.py",
            "model": "v6_4",
            "model_sha256": model_sha,
            "reciprocal": "soft (portable table model of the hardware RCPPS)",
            "fma": "correctly-rounded f32 FMA (2Sum residual construction)",
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "contract": {
            "sample_rate_hz": SAMPLE_RATE,
            "hop_samples": HOP,
            "hop_ms": HOP * 1000.0 / SAMPLE_RATE,
            "window_samples": WINDOW,
            "lookahead_hops": LOOKAHEAD_HOPS,
            "latency_samples": WINDOW + LOOKAHEAD_HOPS * HOP,
            "latency_ms": (WINDOW + LOOKAHEAD_HOPS * HOP) * 1000.0 / SAMPLE_RATE,
            "input_width": INPUT_WIDTH,
            "output_width": OUTPUT_WIDTH,
            "input_layout": "frame-major float32, little-endian, 513 values per frame",
            "output_layout": "frame-major float32, little-endian, 1026 values per frame",
            "bandwidth_hz": [0, SAMPLE_RATE // 2],
        },
        "exactness": {
            "criterion": "bitwise equality of every expected output value",
            "tolerance": 0.0,
        },
        "sets": manifest_sets,
    }
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    print("wrote %s" % (pack / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
