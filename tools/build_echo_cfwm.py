#!/usr/bin/env python3
"""Build the echo model's `.cfwm` weight container.

Same architecture as `build_cfwm.py`: only the tensors the runtime consumes
are emitted, already in the dtype and layout each kernel expects; scalar model
constants are raw `u32` bit patterns (never decimal float literals); the
provenance-marker gate and the constant-bit-pattern gate run before anything is
written.

Two echo-specific inputs are resolved here:

* the private layout (`internal/echo-layout.json`) names a source parameter
  record per tensor; the record is resolved against the parameter table that
  ships next to the weight blob, and the resolved offset is checked against any
  pinned `source_offset` in the layout (drift detection);
* with `--engine`, a private build checker compares every non-constant tensor
  byte for byte with the array the reference engine actually uses, so the
  container is validated by identity, not by trusting a directory table.

Usage (from `cleanfield/`):

    python tools/build_echo_cfwm.py \
        --blob <reference weight blob> --engine <reference engine>

`--blob` is build-time data on purpose: it is not part of the engine.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAGIC = b"CFWM"
HEADER_LEN = 32
TENSOR_ENTRY_LEN = 32
ALIGN = 64
DTYPE_F32 = 0
DTYPE_I8 = 1
DTYPE_I16 = 2
DTYPE_U32 = 3
DTYPE_F16 = 4
DTYPE_NAMES = {DTYPE_F32: "f32", DTYPE_I8: "i8", DTYPE_I16: "i16", DTYPE_U32: "u32", DTYPE_F16: "f16"}
DTYPE_SIZE = {DTYPE_F32: 4, DTYPE_I8: 1, DTYPE_I16: 2, DTYPE_U32: 4, DTYPE_F16: 2}

HEADER_FMT = "<4sHHHHIIII"
TENSOR_FMT = "<HBBIIIIIII"

# Bit patterns that must never drift (a one-ULP transcription changed a
# rounding tie in an earlier engine; `tanh_c4` is the same 60.0 trap).
EXPECTED_BITS = {
    "tanh_c4": 0x42700000,       # 60.0 exactly
    "sig88_clamp": 0x42B0C0A5,
    "exp_c0": 0x39506967,
    "exp_c1": 0x3AB743CE,
    "leaky_slope": 0x3C23D70A,
    "sig_c1": 0x40520000,
    "quant_scale": 0x46000000,
    "hidden_scale": 0x42FE0000,
    "hidden_scale_wide": 0x46FFFE00,
}


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def load_layout(path: Path) -> dict:
    layout = json.loads(path.read_text(encoding="utf-8"))
    for key in ("format_version", "model_id", "constants", "tensors"):
        if key not in layout:
            raise SystemExit(f"layout missing {key!r}")
    if layout["format_version"] != 2:
        raise SystemExit(f"unsupported layout version {layout['format_version']}")
    if len(layout["tensors"]) > 128:
        raise SystemExit("too many tensors")
    return layout


def build_constants(layout: dict) -> tuple[list[int], dict[str, int]]:
    values: dict[str, int] = {}
    ordered: list[int] = []
    for spec in layout["constants"]:
        name = spec.get("name")
        if not name or name in values:
            raise SystemExit(f"bad/duplicate constant name {name!r}")
        if "bits" not in spec:
            raise SystemExit(f"constant {name}: needs an explicit 'bits' word")
        bits = int(spec["bits"], 16) & 0xFFFFFFFF
        values[name] = bits
        ordered.append(bits)
    for name, want in EXPECTED_BITS.items():
        if name in values and values[name] != want:
            raise SystemExit(
                f"constant {name}: bits 0x{values[name]:08x} != expected 0x{want:08x}"
            )
    return ordered, values


def load_params(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for rec in data["records"]:
        name = rec["name"]
        if name in out:
            raise SystemExit(f"duplicate parameter record {name!r}")
        out[name] = rec
    return out


def read_values(blob: bytes, spec: dict, params: dict[str, dict]) -> tuple[list, int]:
    shape = [int(x) for x in spec["shape"]]
    count = 1
    for d in shape:
        count *= d
    item = {"f16": 2, "f32": 4, "i8": 1, "i16": 2}[spec["kind"]]

    if "source" in spec:
        name = spec["source"]
        if name not in params:
            raise SystemExit(f"tensor {spec['role']}: unknown parameter record {name!r}")
        rec = params[name]
        off = int(rec["blob_offset"])
        elements = int(rec["element_count"])
        if elements != count:
            raise SystemExit(
                f"tensor {spec['role']}: record {name!r} has {elements} elements, "
                f"shape {shape} wants {count}"
            )
        if "source_offset" in spec and int(spec["source_offset"]) != off:
            raise SystemExit(
                f"tensor {spec['role']}: source offset drifted "
                f"(layout 0x{int(spec['source_offset']):x}, parameters 0x{off:x})"
            )
    else:
        off = int(spec["source_offset"])
        if "source_offset" not in spec:
            raise SystemExit(f"tensor {spec['role']}: no source record or offset")

    if off < 0 or off + count * item > len(blob):
        raise SystemExit(f"tensor {spec['role']}: source range out of bounds")
    kind = spec["kind"]
    if kind == "f16":
        values = [struct.unpack_from("<e", blob, off + 2 * i)[0] for i in range(count)]
    elif kind == "f32":
        values = [struct.unpack_from("<f", blob, off + 4 * i)[0] for i in range(count)]
    elif kind == "i8":
        values = [struct.unpack_from("<b", blob, off + i)[0] for i in range(count)]
    elif kind == "i16":
        values = [struct.unpack_from("<h", blob, off + 2 * i)[0] for i in range(count)]
    else:
        raise SystemExit(f"tensor {spec['role']}: unsupported kind {kind!r}")
    return values, off


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
    raise SystemExit(f"tensor {spec['role']}: unknown transform {transform!r}")


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
    if dtype == DTYPE_F16:
        for v in values:
            if not math.isfinite(float(v)):
                raise SystemExit("non-finite value in an f16 tensor")
        return struct.pack(f"<{len(values)}e", *[float(v) for v in values])
    raise SystemExit(f"unknown dtype {dtype}")


def build_tensor(
    blob: bytes, spec: dict, constants: list[int], params: dict[str, dict]
) -> dict:
    dtype = int(spec.get("dtype", DTYPE_F32))
    if dtype not in DTYPE_NAMES:
        raise SystemExit(f"tensor {spec['role']}: unknown dtype code {dtype}")

    if spec["kind"] == "constants":
        values = list(constants)
        source_offset = None
    else:
        values, source_offset = read_values(blob, spec, params)
        values = apply_transform(values, spec)

    out_shape = [int(x) for x in spec.get("out_shape", spec["shape"])]
    expected = 1
    for d in out_shape:
        expected *= d
    if len(values) != expected:
        raise SystemExit(
            f"tensor {spec['role']}: extracted {len(values)} values, shape wants {expected}"
        )
    payload = pack_payload(values, dtype)
    if len(payload) != expected * DTYPE_SIZE[dtype]:
        raise SystemExit(f"tensor {spec['role']}: payload size mismatch")

    return {
        "tag": int(spec["tag"]),
        "dtype": dtype,
        "rank": len(out_shape),
        "dims": (out_shape + [0, 0, 0, 0])[:4],
        "byte_len": len(payload),
        "payload": payload,
        "values": values,
        "role": spec["role"],
        "source_offset": source_offset,
        "validate": spec.get("validate"),
    }


def build(layout: dict, blob: bytes, params: dict[str, dict]) -> tuple[bytes, list[dict]]:
    constants, _ = build_constants(layout)
    entries = [build_tensor(blob, spec, constants, params) for spec in layout["tensors"]]

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
        int(layout["format_version"]),
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
# private engine validator (loaded from --engine-check)
# ---------------------------------------------------------------------------
def load_checker(path: Path):
    spec = importlib.util.spec_from_file_location("cf_echo_engine_check", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# ---------------------------------------------------------------------------
# provenance marker scan and container verification
# ---------------------------------------------------------------------------
def provenance_markers(path: Path) -> list[bytes]:
    text = path.read_text(encoding="utf-8")
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
    ap.add_argument(
        "--params",
        type=Path,
        default=None,
        help="parameter table (default: tensors.json next to the blob)",
    )
    ap.add_argument("--layout", type=Path, default=ROOT / "internal" / "echo-layout.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "echo-default.cfwm")
    ap.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="reference engine used for the byte-identity check",
    )
    ap.add_argument(
        "--engine-check",
        type=Path,
        default=ROOT / "internal" / "validate_echo_engine.py",
        help="private checker invoked with --engine",
    )
    ap.add_argument("--no-engine-check", action="store_true")
    args = ap.parse_args()

    layout = load_layout(args.layout)
    blob = args.blob.read_bytes()
    params_path = args.params if args.params else args.blob.parent / "tensors.json"
    params = load_params(params_path)
    data, entries = build(layout, blob, params)

    markers = provenance_markers(ROOT / "PROVENANCE.md")
    hits = scan_container(data, markers)
    runs = printable_runs(data)
    if hits:
        for at, marker in hits:
            print(f"FORBIDDEN MARKER at 0x{at:x}: {marker.decode()!r}", file=sys.stderr)
        raise SystemExit(f"container contains {len(hits)} forbidden marker(s)")

    verify_container(data, entries, layout)

    engine_count = None
    if not args.no_engine_check:
        if args.engine is None:
            raise SystemExit("--engine is required for the byte-identity check (or --no-engine-check)")
        checker = load_checker(args.engine_check)
        engine_count = checker.validate(entries, args.engine, blob)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(args.out)

    size = args.out.stat().st_size
    digest = hashlib.sha256(data).hexdigest()
    print(f"wrote {args.out} ({size} bytes, sha256 {digest})")
    print(
        f"model_id={layout['model_id']} tensors={len(entries)} "
        f"markers_scanned={len(markers)} blob_sha256={hashlib.sha256(blob).hexdigest()}"
    )
    if engine_count is not None:
        print(
            f"engine byte-identity: {engine_count[0]}/{engine_count[1] - 1} non-constant "
            "tensors exact (constants are pinned by bit pattern)"
        )
    const_entry = next(e for e in entries if e["role"] == "constants")
    const_values = struct.unpack(f"<{const_entry['byte_len'] // 4}I", const_entry["payload"])
    print(
        "constants: "
        + ", ".join(
            f"{spec['name']}=0x{v:08x}" for spec, v in zip(layout["constants"], const_values)
        )
    )
    by_dtype: dict[str, int] = {}
    for e in entries:
        by_dtype[DTYPE_NAMES[e["dtype"]]] = by_dtype.get(DTYPE_NAMES[e["dtype"]], 0) + 1
    print("tensor dtypes: " + ", ".join(f"{k}={v}" for k, v in sorted(by_dtype.items())))
    print(f"container strings >=8 chars: {len(runs)} run(s) (none are provenance markers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
