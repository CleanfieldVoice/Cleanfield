#!/usr/bin/env python3
"""Convert a reference .npz dump into the flat CFRF binary format.

This is an optional escape hatch for environments where the `.npz` ZIP/DEFLATE
reader is not wanted.  The values are copied verbatim; no numeric conversion is
performed.  Only little-endian f32 arrays are exported.

Usage:
    python tools/export_ref.py --npz <dump.npz> --out <dump.cfrf>
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"CFRF"
VERSION = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    arrays = []
    for name in z.files:
        try:
            a = z[name]
        except ValueError:
            # Object/metadata array (not needed by the numeric checks).
            continue
        if a.dtype != np.float32 or a.ndim > 4:
            continue
        arrays.append((name.encode("ascii"), np.ascontiguousarray(a)))

    if len(arrays) > 65535:
        raise SystemExit("too many arrays")
    out = bytearray()
    out += MAGIC + struct.pack("<HH", VERSION, len(arrays))
    for name, a in arrays:
        if len(name) > 255:
            raise SystemExit(f"array name too long: {name!r}")
        dims = list(a.shape) + [0] * (4 - a.ndim)
        out += struct.pack("<B", len(name)) + name
        out += struct.pack("<BB", 0, a.ndim)  # dtype 0 = f32
        out += struct.pack("<4I", *dims)
        out += struct.pack("<I", a.nbytes)
        out += a.tobytes(order="C")
    args.out.write_bytes(out)
    print(f"wrote {args.out}: {len(arrays)} arrays, {len(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
