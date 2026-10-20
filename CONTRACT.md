# CleanField integration contract

What a host application must know to drive the engine correctly. Internal model
details are intentionally out of scope.

## Audio format

| Property | Value |
| --- | --- |
| Native sample rate | **16 000 Hz** |
| Channels | **mono** (stereo must be downmixed by the host) |
| Sample type | `float32`, nominal range ±1.0 |
| Model input | 513 floats/frame (voice) or 3 × 513 (echo) |
| Model output | 1026 floats/frame (voice) or 513 mask values (echo) |
| Bandwidth | only **0–8 kHz** is processed; higher bands pass through |

Other sample rates are the host's responsibility. If the host runs at 48 kHz,
resample to 16 kHz before the engine and back afterwards, or apply the mask to
the 0–8 kHz sub-bands only. A linear resampler is adequate for latency tests but
audible artefacts appear at 48 kHz; use a windowed-sinc or polyphase resampler
in production.

## Framing

The host may submit frames of any size. Internally the engine reblocks to a
256-sample hop (16 ms at 16 kHz) and keeps a 768-sample analysis window. Choose
10, 16 or 20 ms frames for the least reblocking. Very small frames (≤ 32 samples)
increase overhead without improving latency.

## State

One engine instance per audio stream. The model carries 5 frames of feature
history plus recurrent state; it is **stateful across frames**.

- `reset()` clears that state and costs one frame; it produces an audible
  transient if called mid-stream, so only call it at stream start.
- Never share an instance between two streams.
- Toggling processing on/off must not destroy the instance if you want a
  click-free switch.

## Latency

Algorithmic latency is **80 ms**: a 768-sample window plus 2 × 256-sample
look-ahead (1 280 samples at 16 kHz). Add the host's own buffering and the
device buffer.

The window is a fixed part of the front end. Reducing look-ahead to 1 × 256
gives ~64 ms at a small quality cost; re-validate before shipping such a change.

## Echo path

The echo stage chains two components:

1. **Linear AEC** — a 512-point partitioned-block frequency-domain NLMS filter
   (8 × 256 taps), a bounded cross-correlation bulk-delay estimate, and a
   double-talk freeze. It produces the residual for the current 256-sample
   frame and exposes the far-end reference advanced by the applied delay.
2. **Neural echo model** — the 89-layer model. It consumes one `3 × 513` frame
   and returns 513 real mask values.

The three input channels, in this order:

| Channel | Signal |
| --- | --- |
| 0 | linear residual (`microphone − estimated echo`) |
| 1 | microphone |
| 2 | aligned reference (advanced by the AEC bulk delay) |

The front end is 16 kHz with a 1024-point FFT, a 768-sample periodic Hann
window and a 256-sample hop. Each of the 513 bins is transformed as
`0.5 * log10(power * preemphasis + 1e-16)`, with
`preemphasis[k] = 1 + 0.92² − 2 × 0.92 × cos(2πk/1024)`. Per bin, the
smaller-power of the residual and microphone spectra is selected (the full
complex value, phase preserved), and the model mask, clamped to `[0, 1]`, is
applied to that spectrum one hop later. The result is inverse transformed,
windowed with the same Hann window, overlap-added with an `8/9` synthesis scale
and emitted in 256-sample blocks.

Latency added by the echo path:

| Stage | Samples | Time at 16 kHz |
| --- | ---: | ---: |
| Linear AEC frame boundary (one frame) | 256 | 16 ms |
| Neural stage (768-sample window + 256-sample look-ahead) | 1024 | 64 ms |
| **Total** | **1280** | **80 ms** |

The AEC's reference-alignment delay advances the reference; it does not add to
the output latency. Host and device buffering are additional.

**Exactness.** The echo model is bit exact against its reference. The linear AEC
has no oracle and is **not** bit exact; it is validated functionally (ERLE,
near-end preservation, double-talk stability, delay recovery, determinism, zero
allocation). The end-to-end echo path is therefore not bit exact either. The
measured behaviour is in `crates/cf-aec/README.md` and
`crates/cf-echo-pipeline/README.md`.

## Resource envelope

| Item | Value |
| --- | --- |
| Steady-state allocation | **0** |
| Workspace | 440 416 bytes (voice), 354 848 bytes (echo) |
| Per-frame CPU, AVX2 | 0.085 ms |
| Per-frame CPU, scalar | 0.24 ms |
| Per-frame CPU, wasm (scalar) | ~2–3 ms |
| Model container | 2 554 560 bytes (voice) |

The engine is single-threaded per instance. It is safe to run one instance per
thread; handles must not be shared across threads.

## Failure semantics

| Condition | Behaviour |
| --- | --- |
| Container bytes invalid/corrupt | constructor returns `NULL`; `cf_last_error()` = `2` |
| Model id mismatch | constructor returns `NULL`; `cf_last_error()` = `3` |
| CPU lacks AVX2 | automatic scalar fallback; `cf_model_backend_name()` reports `scalar` |
| CPU lacks FMA | scalar path uses a portable soft FMA; still bit-identical |
| Out of memory | constructor returns `NULL`; the library never panics |
| `process` called with wrong lengths | returns a negative error code |

Error codes: `1` argument, `2` container, `3` model, `4` state, `5` backend.

## ABI and versioning

- `cf_abi_version()` returns the ABI revision (currently `1`).
- The revision increments **only** for breaking changes; new exports are
  additive and do not bump it.
- The model container format is versioned independently.
- Integrators should assert `cf_abi_version()` at startup and refuse to run on
  a mismatch.

## Bit-exactness

The **neural models** reproduce the reference implementation bit for bit on
every tested input. That covers the voice model and the 89-layer echo model.
The **linear AEC** stage has no reference binary and is not bit exact; it is
validated functionally. The complete echo path (AEC + model + STFT front end) is
consequently not bit exact. Hosts can verify the model-only paths with
`conformance/`:

```
python conformance/verify.py    # or: node conformance/verify.mjs
```

A PASS means every value of every frame matched exactly.
