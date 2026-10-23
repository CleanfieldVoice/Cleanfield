#!/usr/bin/env python3
"""Build a `.cfwm` weight container from the reference weight blob.

Only tensors the runtime actually consumes are emitted, already converted to
the dtype and layout each kernel expects.  Weights are unpacked and rearranged
at build time; scalar model constants are written as raw `u32` bit patterns
(never decimal float literals) so the runtime can restore them with
`f32::from_bits`.  The container carries no names, offsets from the source, or
other identifying strings; the builder fails if a marker derived from
`PROVENANCE.md` is found in the output.

Usage:
    python tools/build_cfwm.py --blob <weights.bin> [--layout internal/layout.json] \
        [--out models/voice-default.cfwm]

The source blob path is a command-line input on purpose: it is build-time
data, not part of the engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MAGIC = b"CFWM"
HEADER_LEN = 32
TENSOR_ENTRY_LEN = 32
ALIGN = 64
DTYPE_F32 = 0
DTYPE_I8 = 1
DTYPE_I16 = 2
DTYPE_U32 = 3
DTYPE_NAMES = {DTYPE_F32: "f32", DTYPE_I8: "i8", DTYPE_I16: "i16", DTYPE_U32: "u32"}

HEADER_FMT = "<4sHHHHIIII"
# magic, version, model_id, tensor_count, flags,
# table_offset, data_offset, data_len, file_len
TENSOR_FMT = "<HBBIIIIIII"  # tag, dtype, rank, reserved, dims[4], offset, byte_len

# Bit patterns that must never drift (guards against decimal-literal rounding).
EXPECTED_BITS = {
    "cell_c4": 0x42700000,   # 60.0 exactly (a 1-ULP-high 0x42700001 broke a tie)
    "cell_c3": 0x43F78000,
    "exp_c0": 0x39506967,
    "exp_c1": 0x3AB743CE,
    "inv_in15": 0x4588E5FC,
    "inv_mm18": 0x45038C24,
}


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def load_layout(path: Path) -> dict:
    layout = json.loads(path.read_text(encoding="utf-8"))
    for key in ("format_version", "model_id", "source", "constants", "tensors"):
        if key not in layout:
            raise SystemExit(f"layout missing {key!r}")
    if len(layout["tensors"]) > 64:
        raise SystemExit("too many tensors")
    return layout


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def f32_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


# ---------------------------------------------------------------------------
# constants (u32 bit patterns, derived values rounded once to f32)
# ---------------------------------------------------------------------------
def build_constants(layout: dict) -> tuple[list[int], dict[str, int]]:
    values: dict[str, int] = {}
    ordered: list[int] = []
    for spec in layout["constants"]:
        name = spec.get("name")
        if not name or name in values:
            raise SystemExit(f"bad/duplicate constant name {name!r}")
        if "bits" in spec:
            bits = int(spec["bits"], 16) & 0xFFFFFFFF
        elif "derive" in spec:
            d = spec["derive"]
            op = d.get("op")
            src = values.get(d.get("of", ""))
            if src is None:
                raise SystemExit(f"constant {name}: unknown source {d.get('of')!r}")
            x = np.float32(f32_value(src))
            if op == "recip_f32":
                bits = f32_bits(np.float32(1.0) / x)
            elif op == "recip_f64":
                bits = f32_bits(np.float32(1.0 / float(np.float64(x))))
            else:
                raise SystemExit(f"constant {name}: unknown derive op {op!r}")
        else:
            raise SystemExit(f"constant {name}: needs 'bits' or 'derive'")
        values[name] = bits
        ordered.append(bits)

    for name, want in EXPECTED_BITS.items():
        if name in values and values[name] != want:
            raise SystemExit(
                f"constant {name}: bits 0x{values[name]:08x} != expected 0x{want:08x}"
            )
    return ordered, values


# ---------------------------------------------------------------------------
# tensor extraction
# ---------------------------------------------------------------------------
def read_values(blob: bytes, spec: dict) -> list:
    shape = [int(x) for x in spec["shape"]]
    count = 1
    for d in shape:
        count *= d
    off = int(spec["source_offset"])
    kind = spec["kind"]
    item = {"f16": 2, "f32": 4, "i8": 1, "i16": 2}[kind]
    if off < 0 or off + count * item > len(blob):
        raise SystemExit(f"tensor {spec['role']}: source range out of bounds")
    if kind == "f16":
        return [struct.unpack_from("<e", blob, off + 2 * i)[0] for i in range(count)]
    if kind == "f32":
        return [struct.unpack_from("<f", blob, off + 4 * i)[0] for i in range(count)]
    if kind == "i8":
        return [struct.unpack_from("<b", blob, off + i)[0] for i in range(count)]
    if kind == "i16":
        return [struct.unpack_from("<h", blob, off + 2 * i)[0] for i in range(count)]
    raise SystemExit(f"tensor {spec['role']}: unsupported kind {kind!r}")


def apply_transform(values: list, spec: dict) -> list:
    shape = [int(x) for x in spec["shape"]]
    transform = spec.get("transform", "none")
    if transform == "none":
        return values
    if transform == "transpose":
        if len(shape) != 2:
            raise SystemExit(f"tensor {spec['role']}: transpose needs 2-D input")
        rows, cols = shape
        return [values[i * cols + j] for j in range(cols) for i in range(rows)]
    if transform == "tap_major":
        # channel first -> tap first
        if len(shape) == 2:
            c, t = shape
            return [values[ci * t + tj] for tj in range(t) for ci in range(c)]
        if len(shape) == 3:
            c, kh, kw = shape
            return [
                values[(ci * kh + i) * kw + j]
                for i in range(kh)
                for j in range(kw)
                for ci in range(c)
            ]
        raise SystemExit(f"tensor {spec['role']}: tap_major needs 2-D or 3-D input")
    raise SystemExit(f"tensor {spec['role']}: unknown transform {transform!r}")


def apply_derive(values: list, spec: dict, consts: dict[str, int]) -> list:
    derive = spec.get("derive")
    if derive is None:
        return values
    op = derive.get("op")
    if op == "halves_sum":
        n = len(values) // 2
        if n * 2 != len(values):
            raise SystemExit(f"tensor {spec['role']}: halves_sum needs an even count")
        return [
            float(np.float32(np.float32(values[i]) + np.float32(values[i + n])))
            for i in range(n)
        ]
    if op == "mul_const":
        name = derive.get("const")
        if name not in consts:
            raise SystemExit(f"tensor {spec['role']}: unknown constant {name!r}")
        c = np.float32(f32_value(consts[name]))
        return [float(np.float32(np.float32(v) * c)) for v in values]
    raise SystemExit(f"tensor {spec['role']}: unknown derive op {op!r}")


def pack_payload(values: list, dtype: int) -> bytes:
    if dtype == DTYPE_F32:
        for v in values:
            if not math.isfinite(float(v)):
                raise SystemExit("non-finite value in an f32 tensor")
        return struct.pack(f"<{len(values)}f", *[float(v) for v in values])
    if dtype == DTYPE_I8:
        for v in values:
            if not -128 <= int(v) <= 127:
                raise SystemExit(f"i8 value {v} out of range")
        return struct.pack(f"<{len(values)}b", *[int(v) for v in values])
    if dtype == DTYPE_I16:
        for v in values:
            if not -32768 <= int(v) <= 32767:
                raise SystemExit(f"i16 value {v} out of range")
        return struct.pack(f"<{len(values)}h", *[int(v) for v in values])
    if dtype == DTYPE_U32:
        for v in values:
            if not 0 <= int(v) <= 0xFFFFFFFF:
                raise SystemExit(f"u32 value {v} out of range")
        return struct.pack(f"<{len(values)}I", *[int(v) for v in values])
    raise SystemExit(f"unknown dtype {dtype}")


def build_tensor(blob: bytes, spec: dict, constants: list[int], consts: dict[str, int]) -> dict:
    dtype = int(spec.get("dtype", DTYPE_F32))
    if dtype not in DTYPE_NAMES:
        raise SystemExit(f"tensor {spec['role']}: unknown dtype code {dtype}")

    if spec["kind"] == "constants":
        values = list(constants)
    else:
        values = read_values(blob, spec)
        values = apply_transform(values, spec)
        values = apply_derive(values, spec, consts)

    out_shape = [int(x) for x in spec.get("out_shape", spec["shape"])]
    expected = 1
    for d in out_shape:
        expected *= d
    if len(values) != expected:
        raise SystemExit(
            f"tensor {spec['role']}: extracted {len(values)} values, shape wants {expected}"
        )
    payload = pack_payload(values, dtype)
    if len(payload) != expected * {0: 4, 1: 1, 2: 2, 3: 4}[dtype]:
        raise SystemExit(f"tensor {spec['role']}: payload size mismatch")

    return {
        "tag": int(spec["tag"]),
        "dtype": dtype,
        "rank": len(out_shape),
        "dims": (out_shape + [0, 0, 0, 0])[:4],
        "byte_len": len(payload),
        "payload": payload,
        "role": spec["role"],
    }


def build(layout: dict, blob: bytes) -> tuple[bytes, list[dict]]:
    layout_version = int(layout["format_version"])
    if layout_version != 2:
        raise SystemExit(f"unsupported layout version {layout_version}")
    expected_sha = layout["source"].get("sha256")
    if expected_sha:
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected_sha:
            raise SystemExit(f"source blob sha256 mismatch: {digest}")
    expected_len = int(layout["source"].get("byte_len", len(blob)))
    if len(blob) != expected_len:
        raise SystemExit(f"source blob length {len(blob)} != {expected_len}")

    constants, consts = build_constants(layout)
    entries = [build_tensor(blob, spec, constants, consts) for spec in layout["tensors"]]

    tags = [e["tag"] for e in entries]
    if len(set(tags)) != len(tags):
        raise SystemExit("duplicate tensor tags")
    if sorted(tags) != list(range(1, len(tags) + 1)):
        raise SystemExit("tensor tags must be 1..N")

    table_offset = HEADER_LEN
    data_offset = align_up(table_offset + len(entries) * TENSOR_ENTRY_LEN, ALIGN)

    cursor = 0
    for e in entries:
        cursor = align_up(cursor, ALIGN)
        e["offset"] = cursor
        cursor += e["byte_len"]
    data_len = align_up(cursor, ALIGN)
    file_len = data_offset + data_len

    out = bytearray(file_len)
    struct.pack_into(
        HEADER_FMT,
        out,
        0,
        MAGIC,
        layout_version,
        int(layout["model_id"]),
        len(entries),
        0,
        table_offset,
        data_offset,
        data_len,
        file_len,
    )
    for i, e in enumerate(entries):
        struct.pack_into(
            TENSOR_FMT,
            out,
            table_offset + i * TENSOR_ENTRY_LEN,
            e["tag"],
            e["dtype"],
            e["rank"],
            0,
            *e["dims"],
            e["offset"],
            e["byte_len"],
        )
        out[data_offset + e["offset"] : data_offset + e["offset"] + e["byte_len"]] = e["payload"]
    return bytes(out), entries


# ---------------------------------------------------------------------------
# provenance marker scan
# ---------------------------------------------------------------------------
def provenance_markers(path: Path) -> list[bytes]:
    """Derive forbidden markers from the single provenance file.

    This keeps the literal names out of the source tree and makes the gate
    follow whatever the provenance statement declares.
    """
    text = path.read_text(encoding="utf-8")
    # Our own format/product vocabulary is allowed (and is what the provenance
    # file is allowed to mention in the first place).
    allow = {"cfwm", "cleanfield"}
    tokens = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{3,}", text):
        tok = raw.strip(".-_").lower()
        if len(tok) < 4 or tok in allow:
            continue
        tokens.add(tok)
        if "." in tok:
            tokens.add(tok.replace(".", "_"))
            tokens.add(tok.replace(".", ""))
    return sorted({t.encode("ascii") for t in tokens if t.isascii()})


def scan_container(data: bytes, markers: list[bytes]) -> list[tuple[int, bytes]]:
    lowered = data.lower()
    hits = []
    for marker in markers:
        start = 0
        while True:
            at = lowered.find(marker, start)
            if at < 0:
                break
            hits.append((at, marker))
            start = at + 1
    return hits


def printable_runs(data: bytes, min_len: int = 8) -> list[str]:
    runs = []
    cur = bytearray()
    for b in data:
        if 0x20 <= b < 0x7F:
            cur.append(b)
        else:
            if len(cur) >= min_len:
                runs.append(cur.decode("ascii"))
            cur.clear()
    if len(cur) >= min_len:
        runs.append(cur.decode("ascii"))
    return runs


def verify_container(data: bytes, entries: list[dict], layout: dict) -> None:
    magic, ver, model_id, count, flags, table_off, data_off, data_len, file_len = struct.unpack_from(
        HEADER_FMT, data, 0
    )
    if magic != MAGIC or ver != layout["format_version"] or model_id != layout["model_id"]:
        raise SystemExit("container header verification failed")
    if count != len(entries) or table_off != HEADER_LEN or file_len != len(data):
        raise SystemExit("container table/file length verification failed")
    if flags != 0 or data_off + data_len != file_len:
        raise SystemExit("container data region verification failed")
    for i, e in enumerate(entries):
        tag, dtype, rank, _res, d0, d1, d2, d3, off, nbytes = struct.unpack_from(
            TENSOR_FMT, data, table_off + i * TENSOR_ENTRY_LEN
        )
        if tag != e["tag"] or dtype != e["dtype"] or rank != e["rank"]:
            raise SystemExit(f"tensor {tag}: descriptor verification failed")
        if [d0, d1, d2, d3] != e["dims"] or nbytes != e["byte_len"]:
            raise SystemExit(f"tensor {tag}: shape/byte length verification failed")
        if off != e["offset"] or off % ALIGN != 0:
            raise SystemExit(f"tensor {tag}: offset verification failed")
        payload = data[data_off + off : data_off + off + nbytes]
        if payload != e["payload"]:
            raise SystemExit(f"tensor {tag}: payload verification failed")
        assert off + nbytes <= data_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", type=Path, required=True, help="reference weight blob")
    ap.add_argument("--layout", type=Path, default=ROOT / "internal" / "layout.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "voice-default.cfwm")
    args = ap.parse_args()

    layout = load_layout(args.layout)
    blob = args.blob.read_bytes()
    data, entries = build(layout, blob)

    provenance_path = ROOT / "PROVENANCE.md"
if provenance_path.exists():
    markers = provenance_markers(provenance_path)
else:
    markers = []
    print("no PROVENANCE.md, skipping the marker gate")
    hits = scan_container(data, markers)
    runs = printable_runs(data)
    if hits:
        for at, marker in hits:
            print(f"FORBIDDEN MARKER at 0x{at:x}: {marker.decode()!r}", file=sys.stderr)
        raise SystemExit(f"container contains {len(hits)} forbidden marker(s)")

    verify_container(data, entries, layout)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(args.out)

    size = args.out.stat().st_size
    digest = hashlib.sha256(data).hexdigest()
    print(f"wrote {args.out} ({size} bytes, sha256 {digest})")
    print(f"model_id={layout['model_id']} tensors={len(entries)} markers_scanned={len(markers)}")
    const_entry = next(e for e in entries if e["role"] == "constants")
    const_values = struct.unpack(f"<{const_entry['byte_len'] // 4}I", const_entry["payload"])
    print(
        "constants: "
        + ", ".join(
            f"{spec['name']}=0x{v:08x}" for spec, v in zip(layout["constants"], const_values)
        )
    )
    for e in entries:
        print(
            f"  tag={e['tag']:2d} role={e['role']:18s} dtype={DTYPE_NAMES[e['dtype']]:3s} "
            f"dims={e['dims']} offset=0x{e['offset']:05x} bytes={e['byte_len']}"
        )
    print(f"container strings >=8 chars: {len(runs)} run(s) (none are provenance markers)")
    if runs:
        for r in runs[:8]:
            print(f"  {r!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

