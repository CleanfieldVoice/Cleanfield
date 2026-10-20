#!/usr/bin/env python3
"""CleanField conformance verifier (native C ABI).

Loads the conformance pack, loads `cleanfield.dll` / `libcleanfield.so` through
ctypes, runs every input frame through `cf_voice_process`, and compares the
result with the expected values bit for bit.

The check is exact: PASS means every one of the 123,120 expected float32 values
matched its expected bit pattern, on the first pass **and** after
`cf_voice_reset()`.  There is no tolerance and no partial credit.

Requires only the Python standard library.  The engine and model are not
bundled with this script; point `--lib` and `--model` at the built artefacts
(or use the defaults under `cleanfield/`).

Usage
-----
    python conformance/verify.py
    python conformance/verify.py --lib ../target/release/cleanfield.dll \
                                 --model ../models/voice-default.cfwm
    python conformance/verify.py --json result.json

Exit code 0 on PASS, 1 on FAIL, 2 on a setup error (missing files, bad pack).
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                    # cleanfield/

CF_OK = 0
CF_ABI_VERSION = 1
CF_VOICE_INPUT_LEN = 513
CF_VOICE_OUTPUT_LEN = 1026
CF_VOICE_WORKSPACE_BYTES = 440_416
CF_VOICE_MODEL_BYTES = 2_554_560

ERROR_NAMES = {
    0: "CF_OK",
    1: "CF_ERR_ARG (bad argument)",
    2: "CF_ERR_CONTAINER (not a valid .cfwm image)",
    3: "CF_ERR_MODEL (wrong model id, failed validation, wrong handle type)",
    4: "CF_ERR_STATE (null handle/frame buffer or failed frame)",
    5: "CF_ERR_BACKEND (requested backend unavailable on this CPU)",
}


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_f32(path: Path, count: int) -> tuple[float, ...]:
    data = path.read_bytes()
    if len(data) != count * 4:
        raise SystemExit("%s: expected %d bytes, found %d" % (path, count * 4, len(data)))
    return struct.unpack("<%df" % count, data)


def load_pack(pack: Path) -> dict:
    manifest_path = pack / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("missing %s (run generate_pack.py first)" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "cleanfield-conformance/1":
        raise SystemExit("unsupported pack format %r" % manifest.get("format"))
    for entry in manifest["sets"]:
        if entry["input_width"] != CF_VOICE_INPUT_LEN:
            raise SystemExit("%s: input_width %r != 513" % (entry["name"], entry["input_width"]))
        if entry["output_width"] != CF_VOICE_OUTPUT_LEN:
            raise SystemExit("%s: output_width %r != 1026" % (entry["name"], entry["output_width"]))
    return manifest


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------
def default_library() -> Path:
    candidates = [
        ROOT / "target" / "release" / "cleanfield.dll",
        ROOT / "target" / "release" / "libcleanfield.so",
        ROOT / "target" / "release" / "libcleanfield.dylib",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def open_library(path: Path):
    if not path.exists():
        raise SystemExit(
            "library not found: %s\nBuild it with: cargo build --release -p cf-ffi"
            % path
        )
    lib = C.CDLL(str(path))

    lib.cf_abi_version.restype = C.c_uint32
    lib.cf_abi_version.argtypes = []
    lib.cf_last_error.restype = C.c_uint32
    lib.cf_last_error.argtypes = []
    lib.cf_voice_new.restype = C.c_void_p
    lib.cf_voice_new.argtypes = [C.POINTER(C.c_uint8), C.c_size_t]
    lib.cf_voice_free.restype = None
    lib.cf_voice_free.argtypes = [C.c_void_p]
    lib.cf_voice_process.restype = C.c_int32
    lib.cf_voice_process.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_float),
        C.POINTER(C.c_float),
    ]
    lib.cf_voice_reset.restype = C.c_int32
    lib.cf_voice_reset.argtypes = [C.c_void_p]
    lib.cf_voice_input_len.restype = C.c_uint32
    lib.cf_voice_input_len.argtypes = []
    lib.cf_voice_output_len.restype = C.c_uint32
    lib.cf_voice_output_len.argtypes = []
    lib.cf_voice_workspace_bytes.restype = C.c_uint32
    lib.cf_voice_workspace_bytes.argtypes = []
    lib.cf_voice_model_bytes.restype = C.c_uint32
    lib.cf_voice_model_bytes.argtypes = [C.c_void_p]
    lib.cf_model_backend_name.restype = C.c_char_p
    lib.cf_model_backend_name.argtypes = [C.c_void_p]
    lib.cf_model_workspace_bytes.restype = C.c_uint32
    lib.cf_model_workspace_bytes.argtypes = [C.c_void_p]
    return lib


def open_model(lib, model_path: Path):
    if not model_path.exists():
        raise SystemExit("model container not found: %s" % model_path)
    blob = model_path.read_bytes()
    if len(blob) % 4:
        raise SystemExit("%s: length %d is not a multiple of 4" % (model_path, len(blob)))
    buffer = (C.c_uint8 * len(blob)).from_buffer_copy(blob)
    handle = lib.cf_voice_new(buffer, len(blob))
    if not handle:
        code = lib.cf_last_error()
        raise SystemExit(
            "cf_voice_new failed: code %d, %s" % (code, ERROR_NAMES.get(code, "unknown"))
        )
    return handle, len(blob)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def compare(actual: list[float], expected: tuple[float, ...]) -> tuple[int, float]:
    """Return (exact_matches, max_abs_diff) over the whole array."""
    exact = 0
    max_diff = 0.0
    for index, value in enumerate(actual):
        other = expected[index]
        if struct.pack("<f", value) == struct.pack("<f", other):
            exact += 1
        elif math.isfinite(value) and math.isfinite(other):
            diff = abs(value - other)
            if diff > max_diff:
                max_diff = diff
        else:
            max_diff = math.inf
    return exact, max_diff


def run_set(lib, handle, pack: Path, entry: dict, replay: bool) -> dict:
    frames = entry["frames"]
    in_width = entry["input_width"]
    out_width = entry["output_width"]
    # Each set is an independent stream: clear the cross-frame state so the
    # first frame of this set sees the same initial state the reference engine
    # used when the pack was generated.
    rc = lib.cf_voice_reset(handle)
    if rc != CF_OK:
        raise SystemExit("%s: cf_voice_reset returned %d" % (entry["name"], rc))
    inputs = unpack_f32(pack / entry["input_file"], frames * in_width)
    expected = unpack_f32(pack / entry["output_file"], frames * out_width)

    in_buf = (C.c_float * in_width)()
    out_buf = (C.c_float * out_width)()
    actual: list[float] = []
    per_frame = []
    for frame in range(frames):
        start = frame * in_width
        for index in range(in_width):
            in_buf[index] = inputs[start + index]
        rc = lib.cf_voice_process(handle, in_buf, out_buf)
        if rc != CF_OK:
            raise SystemExit(
                "%s frame %d: cf_voice_process returned %d (%s)"
                % (entry["name"], frame, rc, ERROR_NAMES.get(-rc, "unknown"))
            )
        values = list(out_buf)
        frame_exact, frame_diff = compare(values, expected[frame * out_width:(frame + 1) * out_width])
        per_frame.append(frame_exact)
        actual.extend(values)

    exact, max_diff = compare(actual, expected)
    result = {
        "name": entry["name"],
        "frames": frames,
        "values": len(expected),
        "exact": exact,
        "mismatches": len(expected) - exact,
        "max_abs_diff": max_diff,
        "per_frame_exact": per_frame,
        "reset_replay": None,
    }

    if replay:
        rc = lib.cf_voice_reset(handle)
        if rc != CF_OK:
            raise SystemExit("%s: cf_voice_reset returned %d" % (entry["name"], rc))
        replayed: list[float] = []
        for frame in range(frames):
            start = frame * in_width
            for index in range(in_width):
                in_buf[index] = inputs[start + index]
            rc = lib.cf_voice_process(handle, in_buf, out_buf)
            if rc != CF_OK:
                raise SystemExit("%s replay frame %d: returned %d" % (entry["name"], frame, rc))
            replayed.extend(out_buf)
        replay_exact, replay_diff = compare(replayed, expected)
        result["reset_replay"] = {
            "exact": replay_exact,
            "mismatches": len(expected) - replay_exact,
            "max_abs_diff": replay_diff,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a CleanField integration against the conformance pack.")
    parser.add_argument("--pack", type=Path, default=HERE, help="pack directory (default: script directory)")
    parser.add_argument("--lib", type=Path, default=None, help="native library path")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "voice-default.cfwm",
                        help="voice container (.cfwm) path")
    parser.add_argument("--json", type=Path, default=None, help="write a machine-readable result")
    parser.add_argument("--no-reset", action="store_true", help="skip the reset-and-replay check")
    args = parser.parse_args()

    pack = args.pack.resolve()
    manifest = load_pack(pack)
    lib_path = (args.lib or default_library()).resolve()
    lib = open_library(lib_path)

    setup_errors = []
    abi = lib.cf_abi_version()
    if abi != CF_ABI_VERSION:
        setup_errors.append("cf_abi_version() = %d, expected %d" % (abi, CF_ABI_VERSION))

    input_len = lib.cf_voice_input_len()
    output_len = lib.cf_voice_output_len()
    workspace = lib.cf_voice_workspace_bytes()
    if input_len != CF_VOICE_INPUT_LEN:
        setup_errors.append("cf_voice_input_len() = %d, expected %d" % (input_len, CF_VOICE_INPUT_LEN))
    if output_len != CF_VOICE_OUTPUT_LEN:
        setup_errors.append("cf_voice_output_len() = %d, expected %d" % (output_len, CF_VOICE_OUTPUT_LEN))
    if workspace != CF_VOICE_WORKSPACE_BYTES:
        setup_errors.append("cf_voice_workspace_bytes() = %d, expected %d"
                            % (workspace, CF_VOICE_WORKSPACE_BYTES))

    hash_errors = []
    for entry in manifest["sets"]:
        for key, field in (("input_file", "input_sha256"), ("output_file", "output_sha256")):
            path = pack / entry[key]
            if not path.exists():
                hash_errors.append("%s: missing %s" % (entry["name"], path))
            elif sha256(path) != entry[field]:
                hash_errors.append("%s: %s hash mismatch" % (entry["name"], key))

    handle, model_bytes = open_model(lib, args.model.resolve())
    backend = (lib.cf_model_backend_name(handle) or b"?").decode("ascii", "replace")
    handle_workspace = lib.cf_model_workspace_bytes(handle)

    print("pack:    %s" % pack)
    print("library: %s" % lib_path)
    print("model:   %s (%d bytes, model_bytes=%d)"
          % (args.model, model_bytes, lib.cf_voice_model_bytes(handle)))
    print("abi:     %d; backend: %s; workspace: %d bytes"
          % (abi, backend, handle_workspace))
    print()
    header = "%-18s %7s %8s %8s %7s %12s %s" % (
        "set", "frames", "values", "exact", "mism", "max|diff|", "reset replay")
    print(header)
    print("-" * len(header))

    results = []
    try:
        for entry in manifest["sets"]:
            result = run_set(lib, handle, pack, entry, replay=not args.no_reset)
            results.append(result)
            replay = result["reset_replay"]
            replay_text = ("off" if replay is None else
                           ("exact" if replay["mismatches"] == 0
                            else "%d MISMATCH" % replay["mismatches"]))
            print("%-18s %7d %8d %8d %7d %12.3e %s"
                  % (result["name"], result["frames"], result["values"], result["exact"],
                     result["mismatches"], result["max_abs_diff"], replay_text))
    finally:
        lib.cf_voice_free(handle)

    values = sum(r["values"] for r in results)
    exact = sum(r["exact"] for r in results)
    mismatches = values - exact
    max_diff = max((r["max_abs_diff"] for r in results), default=0.0)
    replay_ok = all(r["reset_replay"] is None or r["reset_replay"]["mismatches"] == 0
                    for r in results)

    print()
    print("checks: hashes=%s abi=%s queries=%s exact=%d/%d reset=%s"
          % ("ok" if not hash_errors else "FAIL",
             "ok" if abi == CF_ABI_VERSION else "FAIL",
             "ok" if not setup_errors else "FAIL",
             exact, values,
             "ok" if replay_ok else "FAIL"))

    failed = bool(setup_errors or hash_errors or mismatches or not replay_ok)
    for message in setup_errors + hash_errors:
        print("error: %s" % message)
    print()
    print("RESULT: %s" % ("FAIL" if failed else "PASS"))

    if args.json:
        args.json.write_text(json.dumps({
            "format": "cleanfield-conformance-result/1",
            "verifier": "conformance/verify.py",
            "pass": not failed,
            "library": str(lib_path),
            "model": str(args.model),
            "abi_version": abi,
            "backend": backend,
            "workspace_bytes": handle_workspace,
            "values": values,
            "exact": exact,
            "mismatches": mismatches,
            "max_abs_diff": max_diff,
            "reset_replay_ok": replay_ok,
            "sets": results,
            "errors": setup_errors + hash_errors,
        }, indent=2) + "\n", encoding="utf-8")
        print("wrote %s" % args.json)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
