#!/usr/bin/env python3
"""Reproduce the 18 bundled real measured RIRs from OpenSLR SLR28.

The archive is 1.3 GB, so this script reads only the 18 members it needs with
HTTP range requests (the `remotezip` package). Pass `--zip` to work from an
already-downloaded `rirs_noises.zip` instead.

    pip install remotezip
    python extract_slr28.py            # downloads the 18 members only
    python extract_slr28.py --zip ~/Downloads/rirs_noises.zip

The script writes `*.wav` next to itself and checks the SHA-256 of every output
against the values recorded in PROVENANCE.md. It is a provenance/reproduction
tool: the generated files are already in the tree, so tests never run it.

Licence: OpenSLR SLR28 declares Apache License 2.0 for the archive. See
PROVENANCE.md and LICENSE-APACHE-2.0.txt.
"""

import argparse
import hashlib
import io
import os
import sys
import wave
import zipfile

import numpy as np

FS = 16000
ARCHIVE = "https://www.openslr.org/resources/28/rirs_noises.zip"
BASE = "RIRS_NOISES/real_rirs_isotropic_noises/"

# id, archive member, short description used only in the console output
SELECTED = [
    ("air-booth", "air_type1_air_binaural_booth_1_2.wav", "AIR booth 2.0 m"),
    ("rvb-smallroom1-near", "RVB2014_type1_rir_smallroom1_near_angla.wav", "REVERB small room 1 near"),
    ("rvb-smallroom2-near", "RVB2014_type1_rir_smallroom2_near_angla.wav", "REVERB small room 2 near"),
    ("rvb-smallroom2-far", "RVB2014_type1_rir_smallroom2_far_angla.wav", "REVERB small room 2 far"),
    ("air-meeting", "air_type1_air_binaural_meeting_0_1.wav", "AIR meeting 1.0 m"),
    ("rwcp-e2a", "RWCP_type1_rir_cirline_e2a_imp090.wav", "RWCP e2a ch 0"),
    ("rwcp-e1b", "RWCP_type1_rir_circle_e1b_imp090.wav", "RWCP e1b ch 0"),
    ("rwcp-e1c", "RWCP_type1_rir_circle_e1c_imp090.wav", "RWCP e1c ch 0"),
    ("rvb-mediumroom1-near", "RVB2014_type1_rir_mediumroom1_near_angla.wav", "REVERB medium room 1 near"),
    ("rvb-mediumroom2-near", "RVB2014_type1_rir_mediumroom2_near_angla.wav", "REVERB medium room 2 near"),
    ("air-office", "air_type1_air_binaural_office_1_2.wav", "AIR office 2.0 m"),
    ("rvb-largeroom1-near", "RVB2014_type1_rir_largeroom1_near_angla.wav", "REVERB large room 1 near"),
    ("rvb-mediumroom2-far", "RVB2014_type1_rir_mediumroom2_far_angla.wav", "REVERB medium room 2 far"),
    ("air-stairway", "air_type1_air_binaural_stairway_1_1_0.wav", "AIR stairway 1.0 m"),
    ("rwcp-ofc", "RWCP_type1_rir_cirline_ofc_imp090.wav", "RWCP ofc ch 0"),
    ("air-lecture", "air_type1_air_binaural_lecture_0_2.wav", "AIR lecture 2.0 m"),
    ("rvb-largeroom2-near", "RVB2014_type1_rir_largeroom2_near_angla.wav", "REVERB large room 2 near"),
    ("rvb-largeroom2-far", "RVB2014_type1_rir_largeroom2_far_angla.wav", "REVERB large room 2 far"),
]

# SHA-256 of the files as committed, in the same order as SELECTED.
EXPECTED = {
    "air-booth": "98b8cf31b571d82c3e7b9b56daf11efefe8ff6ff90ebbe901aa81433837fa6a8",
    "rvb-smallroom1-near": "461dbc95e61851307c5ba3ccec3b1d452945da0381732add72dba7c8a4e77ac8",
    "rvb-smallroom2-near": "71bef70c5e667787aef37b450ee13e478d3b7d9a7e8f0f0e603fc197ece9d8a5",
    "rvb-smallroom2-far": "e67713d9387d71c89f4bc974f84926d28f822eb34339fc105ae52dd490d41506",
    "air-meeting": "546457ba40eac626b3d6117e31375b0dae8d272842e18ec70d40b44ca3985bfc",
    "rwcp-e2a": "33769ecf9d38677276b190b08a3eb08440ac2d84b4b702df9f27d5b02721c44b",
    "rwcp-e1b": "c41082ed63a6af3b2520bf3482078dea14bd4f1cf9bbea10d8cca763105c9aad",
    "rwcp-e1c": "a3f5f42e03bf5014aa06e806f8a147a4a4c40c8b6acbfaea4c4c2c51e56a8217",
    "rvb-mediumroom1-near": "cdf24fd9bf9507b6b5ca665e004d147206b8bd26d4f43b55deec2e3833cc58d0",
    "rvb-mediumroom2-near": "b1bc66d38dad7b0387311da34d892499d782266c33c3b1dc28c216210a8e07f7",
    "air-office": "482e71e480c892d65b6219f9dc13f4d716744a2dcb23445f3e9fc0b9d53a05f7",
    "rvb-largeroom1-near": "64120b11297efb57670676c19fc84b9d35d1adad2a464ac1e8052b5e98ad2e5a",
    "rvb-mediumroom2-far": "fc8693eba26f5c71cbcde668a0b571961a63426ca833919c6353d05d07749663",
    "air-stairway": "1af67f07434a53d15b5d413974bbb7ec22b39717bed1cefb429478746647f815",
    "rwcp-ofc": "66870e020430d175536556f1e37ff4de7daf84732c29610e847e3d9c8b9bb99b",
    "air-lecture": "c779eb4b83f4e56807c3297dd5ea7174e86caff93a29cbce91456fcb4015c451",
    "rvb-largeroom2-near": "5346f3900e64d746a8c6f2bae27291c04a37c97258762aa75ea155266acf9e49",
    "rvb-largeroom2-far": "4bb8916d416e9c142cbca0f009f5031671832ad4015f2476dc1e875d47607e7a",
}

MAX_SECONDS = 1.60


def read_wav_channel0(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if rate != FS or width != 2:
        raise ValueError(f"unexpected WAV header: {rate} Hz, {width * 8} bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, channels)[:, 0].copy()


def convert(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    peak = int(np.argmax(np.abs(x)))
    threshold = 0.01 * abs(x[peak])
    onset = peak
    while onset > 0 and abs(x[onset - 1]) > threshold:
        onset -= 1
    onset = max(0, onset - 8)
    x = x[onset:]
    x = x[: int(MAX_SECONDS * FS)]
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="use an already-downloaded rirs_noises.zip")
    parser.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    if args.zip:
        archive = zipfile.ZipFile(args.zip)
        opener = lambda name: archive.read(BASE + name)
    else:
        try:
            import remotezip
        except ImportError:
            print("install remotezip (`pip install remotezip`) or pass --zip", file=sys.stderr)
            return 2
        archive = remotezip.RemoteZip(ARCHIVE, support_suffix_range=False)
        opener = lambda name: archive.read(BASE + name)

    failures = 0
    for rid, member, description in SELECTED:
        pcm = convert(read_wav_channel0(opener(member)))
        path = os.path.join(args.out, rid + ".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes(pcm.tobytes())
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        ok = digest == EXPECTED[rid]
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} {rid:<24} {description:<28} {len(pcm):>6} taps  {digest[:16]}")
        if not ok:
            print(f"    expected {EXPECTED[rid]}")

    print(f"{len(SELECTED) - failures}/{len(SELECTED)} files reproduce their recorded hash")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
