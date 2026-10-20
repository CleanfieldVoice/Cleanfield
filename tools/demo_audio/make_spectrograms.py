"""Render the demo spectrograms (before/after, 0-8 kHz, purple/orange).

Usage:
    python make_spectrograms.py [--denoise-pair NAME] [--assets DIR]

The colours, panel layout and typography match the figures that shipped with
the repository: dark figure background, two stacked panels, one shared
colourbar labelled ``Power (dBFS)``.
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SR = 16_000
BAND_LIMIT_HZ = 8000.0

# Sampled from the previous figures' colourbar, dark purple -> pale orange.
_COLORS = [
    (0.125, 0.071, 0.204),
    (0.102, 0.063, 0.255),
    (0.172, 0.067, 0.373),
    (0.251, 0.059, 0.451),
    (0.333, 0.075, 0.490),
    (0.408, 0.106, 0.502),
    (0.475, 0.133, 0.506),
    (0.549, 0.161, 0.502),
    (0.620, 0.180, 0.494),
    (0.702, 0.208, 0.478),
    (0.773, 0.235, 0.455),
    (0.847, 0.271, 0.424),
    (0.910, 0.325, 0.384),
    (0.953, 0.396, 0.361),
    (0.976, 0.471, 0.361),
    (0.988, 0.573, 0.396),
    (0.996, 0.659, 0.451),
    (0.996, 0.722, 0.494),
    (0.992, 0.812, 0.569),
]
CMAP = LinearSegmentedColormap.from_list("cleanfield", _COLORS, N=256)
BG = "#120a1f"
FG = "#e8e2f0"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def spectrogram(samples: np.ndarray, window: int = 512, hop: int = 128):
    win = np.hanning(window).astype(np.float32)
    count = 1 + (len(samples) - window) // hop
    frames = np.stack([samples[i * hop:i * hop + window] * win for i in range(count)])
    spectrum = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    power = spectrum / (win.sum() / 2.0)
    db = 20.0 * np.log10(np.maximum(power, 1e-12))
    return db.T, count * hop / SR


def render(before: Path, after: Path, title: str, out: Path) -> None:
    a = read_wav(before)
    b = read_wav(after)
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    db_a, duration = spectrogram(a)
    db_b, _ = spectrogram(b)

    fig = plt.figure(figsize=(11.04, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    axes = [fig.add_axes([0.075, 0.545, 0.795, 0.375]), fig.add_axes([0.075, 0.105, 0.795, 0.375])]
    cax = fig.add_axes([0.895, 0.105, 0.022, 0.815])
    for ax, data, label, edge in zip(
        axes, (db_a, db_b), ("Noisy", "Enhanced"), ("#ff9a4d", "#b3306e")
    ):
        ax.set_facecolor("#0d0818")
        image = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            vmin=-90,
            vmax=-20,
            extent=[0, duration, 0, BAND_LIMIT_HZ / 1000.0],
            interpolation="nearest",
        )
        ax.set_ylim(0, BAND_LIMIT_HZ / 1000.0)
        ax.set_ylabel("Frequency (kHz)", color=FG, fontsize=11)
        ax.tick_params(colors=FG, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#4a3f5e")
        ax.text(
            0.012, 0.94, label, transform=ax.transAxes, color="white", fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1b1330", edgecolor=edge, linewidth=1.4),
        )
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Time (s)", color=FG, fontsize=11)
    axes[1].set_xlim(0, duration)
    fig.text(0.47, 0.955, title, color=FG, fontsize=15, ha="center", va="center")

    bar = fig.colorbar(image, cax=cax)
    bar.set_label("Power (dBFS)", color=FG, fontsize=11)
    bar.ax.yaxis.set_tick_params(color=FG, labelcolor=FG, labelsize=9)
    bar.outline.set_edgecolor("#4a3f5e")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=100)
    plt.close(fig)
    print(f"wrote {out} ({duration:.2f} s, {BAND_LIMIT_HZ / 1000:.0f} kHz)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(r"<workspace>\cleanfield-repo\assets\audio"))
    parser.add_argument("--denoise-pair", default="denoise-zh-babble")
    args = parser.parse_args()

    render(
        args.assets / f"{args.denoise_pair}-before.wav",
        args.assets / f"{args.denoise_pair}-after.wav",
        "CleanField noise suppression",
        args.assets / "spectrogram-denoise.png",
    )
    render(
        args.assets / "echo-zh-before.wav",
        args.assets / "echo-zh-after.wav",
        "CleanField echo cancellation",
        args.assets / "spectrogram-echo.png",
    )


if __name__ == "__main__":
    main()

