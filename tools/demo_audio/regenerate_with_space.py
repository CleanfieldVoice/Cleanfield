"""One command to regenerate every demo asset from the Space.

Requires ``HF_TOKEN`` in the environment (ZeroGPU quota); the token is read
from the environment and never written to disk.

    $env:HF_TOKEN = ...
    python tools/demo_audio/regenerate_with_space.py

Steps: synthesize the five speech renders, rebuild the ten denoise pairs and the
echo pair, re-measure, then re-render both spectrograms.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run(command: list[str], **env: str) -> None:
    print("$", " ".join(command))
    merged = dict(os.environ)
    merged.update({key: str(value) for key, value in env.items()})
    result = subprocess.run(command, cwd=str(REPO), env=merged)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech", type=Path, default=Path(r"<workspace>\tts\speech-space"))
    parser.add_argument("--work", type=Path, default=Path(r"<workspace>\tts\build"))
    parser.add_argument(
        "--echo-driver", type=Path, default=Path(r"<workspace>\tts\cargo-target\release\echo_driver.exe")
    )
    parser.add_argument("--skip-build-driver", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set; the Space needs it for ZeroGPU quota")

    target = r"<workspace>\tts\cargo-target"
    if not args.skip_build_driver:
        run(
            ["cargo", "build", "--release", "--manifest-path", "tools/demo_audio/echo_driver/Cargo.toml"],
            CARGO_TARGET_DIR=target,
        )

    run([sys.executable, "tools/demo_audio/synthesize_space.py", "--out", str(args.speech)])
    run(
        [sys.executable, "tools/demo_audio/build_demo_audio.py"],
        CF_DEMO_SPEECH=str(args.speech),
        CF_DEMO_WORK=str(args.work),
        CF_ECHO_DRIVER=str(args.echo_driver),
    )
    run([sys.executable, "tools/demo_audio/make_spectrograms.py"])


if __name__ == "__main__":
    main()

