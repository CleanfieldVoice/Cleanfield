# Room-condition robustness of the linear AEC

This is the evidence document for `crates/cf-aec` under realistic room
conditions. It is generated from the same deterministic, `std`-only scene code
that the `room_conditions` integration test uses; the report and the assertions
cannot drift apart.

## Verdict

159 seeded runs (53 scenes x 3 seeds) were measured on the synthetic matrix. A
further **216 runs** over 18 **measured** room responses are reported in
"Real rooms"; they confirm the stability and near-end results and show that the
synthetic matrix was **optimistic by 4.1 dB** on the matched aggregate
(5.3 dB on the clean baselines).

* **0** runs produced a non-finite sample or a non-finite filter state.
* **0** runs crossed the divergence definition (NaN, filter norm > 10, or
  echo-cancellation ERLE below -6 dB).
* Filter norm stayed at or below **0.9** on every run, against a bound of 10.
* **0 / 159** runs had a steady-state (post-event) echo ERLE below 3 dB (was
  **9 / 159**, all of them 200 ppm clock-drift scenes). The clock-rate
  correction described below lifts the drift aggregate from **3.6 dB** (min
  -1.0) to **16.8 dB** with the rate correction alone and **17.2 dB** after the
  double-talk fix (min 4.3, max 28.9), and the clean diffuse room that used to
  be the worst case posts **22.3 dB** with every seed above 21 dB. See
  `DRIFT-SPLIT.md` for the drift-specific measurement.
* **0 / 159** runs failed to reach the recovery criterion inside the measurement
  window (was 5).
* Clean static short and office rooms reached **29.0 dB** and **28.1 dB** mean
  echo ERLE after convergence, the hall **15.9 dB** and the diffuse room
  **22.3 dB**. At 30 dB SNR the raw means are **20.3-20.5 dB** (noise-removed
  **21.3-21.6 dB**); at 15 dB, **11.1-12.7 dB** (noise-removed
  **15.5-17.4 dB**). At 5 dB the raw ERLE is capped near 6.2 dB by the noise
  itself and measured **4.3-5.4 dB**, while noise-removed echo ERLE stayed
  **8.5-12.1 dB**.
* Double talk cost at most **0.2 dB** of near-end amplitude on any run
  (mean 0.0 dB, worst min -0.2 dB), and filter norms stayed at or below 0.9.
* The clock-rate correction costs about **1.7 us per 16 ms frame**: 1.1 us for
  the cubic resampler in the steady state and 0.6 us amortised for the extra
  correlation analyses during its acquisition phase (measured in release,
  `examples/rate_probe`). CPU per frame otherwise rounds to **0.0 ms** in these
  test/report builds (the printed cell has 0.1 ms resolution; the separate
  `aec_report` measures about 0.17 ms). This is a report number, not a contract.

The honest summary: the late-reflection lock that dominated an earlier run is
fixed. GCC-PHAT first-onset selection plus the 96-sample causal guard recovers
the diffuse A5 seed at **+22.1 dB** instead of -4.4 dB, and the two-tier re-lock
gate keeps a working alignment from being stolen by a later pitch/period peak.
The linear stage is stable and useful in clean and moderate-noise rooms, it
preserves the near end, and it does not diverge. The 200 ppm clock-drift
collapse is fixed too: the reference-to-microphone rate is estimated from the
correlation profiles the delay estimator already computes and removed with a
fractional resampler, which returns the drift scenes to within about 1.2 dB of
the static bound. The double-talk detector also stopped treating a stationary
noise floor as near-end: the office / pink 15 dB / double-talk scene rises from
**12.5 dB to 15.8 dB** against a **19.9 dB** matched static control. Drift +
noise + double talk is still the weakest combination (6-16 dB echo ERLE post),
but it no longer collapses.

The measured-room check in "Real rooms" does not change those verdicts: none of
the 216 real-room runs collapses, diverges or loses the near end, but the real
aggregate sits **4.1 dB below** the matched synthetic aggregate. The synthetic
grid was optimistic about the room population (it has no room longer than
256 ms and no far-mic condition), not wrong about the canceller's stability.

### Historical note: the re-lock gate was not the lever

An earlier investigation of the drift collapses treated the two-tier delay
re-lock gate as the suspect, because a stale alignment can leave a residual that
looks like double talk. `DRIFT-SPLIT.md` measured **zero** applied-delay changes
after the initial lock across all 27 drift runs, so gate or hangover tuning had
nothing to act on. The lever was estimating and removing the clock rate; the
re-lock gate remains for genuine delay steps and late reflections but is not
what fixed the drift results. Likewise the old double-talk cap (32 frozen frames
followed by 16 adaptation frames) was never reached in the failures that
remained: the target scene's false freezes were 4-15 frame bursts, not a latch.
See fix 6 below.

## Signal and normalisation policy

* 16 000 Hz mono, `f32`, one AEC frame = 256 samples = 16 ms. Scene length is
  640 frames = 10.24 s.
* Every synthetic room response is normalised to **unit L2 energy**,
  `sum h[n]^2 = 1`. Echo power therefore tracks far-end power for a
  white-enough reference; ERLE is not flattered or penalised by the absolute
  gain of a synthetic room.
* The far-end and near-end talkers are source-filter speech-like signals
  (jittered glottal pulses, two drifting formants, syllabic and phrase
  envelopes), seeded independently of the noise. The far-end reference RMS is
  0.08 before the scene peak guard.
* Background noise is generated independently of the talkers and scaled so
  that `echo RMS / noise RMS = 10^(SNR/20)`, with SNR in {30, 15, 5} dB.
* Near-end speech is faded in for 10 ms, held for the 2.88 s double-talk
  segment (frames 240-420), and faded out for 10 ms. Its RMS in that segment is
  `echo RMS * 10^(level/20)` for level in {-6, 0, +3} dB.
* If the mixed microphone would clip above 0.95, the microphone, echo, noise,
  near-end and reference are all scaled by the same factor. The ratios above
  are therefore unchanged.
* The reference is the raw far-end signal. Bulk delay, room response and clock
  drift are properties of the echo path, not of the reference; the delay
  estimator must recover them from the microphone.

### Room responses

| Room | Taps | Span (ms) | Peak index | Peak (ms) | Centroid (ms) | Energy < 40 ms | Peak energy share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short-direct-20ms | 384 | 24.0 | 0 | 0.0 | 1.1 | 100.0 % | 71.5 % |
| office-60ms | 1024 | 64.0 | 0 | 0.0 | 3.3 | 99.7 % | 52.3 % |
| hall-250ms | 4096 | 256.0 | 0 | 0.0 | 20.4 | 82.4 % | 30.8 % |
| diffuse-sparse-150ms | 2400 | 150.0 | 520 | 32.5 | 30.7 | 71.8 % | 11.7 % |
| late-direct-45ms | 3200 | 200.0 | 720 | 45.0 | 65.0 | 0.0 % | 25.0 % |
| weak-direct-strong-early | 1920 | 120.0 | 96 | 6.0 | 13.2 | 99.2 % | 26.2 % |

The `hall-250ms` response is direct-dominated but 256 ms long, with a
substantial diffuse tail; the `late-direct-45ms` response has no energy before
40 ms and its direct arrival is at 45 ms.

## Scene matrix

The matrix is a designed experiment, not the full Cartesian product (which
would be 720 scenes and unreadable). The 53 scenes cover:

* six room families at their clean baseline (static delay, no noise, no double
  talk);
* office and hall x {white, pink, babble} x {30, 15, 5} dB SNR;
* short-direct and diffuse rooms at the hardest noise condition (babble 5 dB);
* all six rooms with a +5 ms delay step at frame 300;
* all six rooms with 200 ppm clock drift over the file;
* office, hall and weak-direct-strong-early x {-6, 0, +3} dB near-end;
* six combined stress scenes (long/diffuse room + noise + drift/step +
  double talk).

The exact list is in `crates/cf-aec/tests/room/matrix.rs`.

## Metrics and measurement windows

| Metric | Definition |
| --- | --- |
| ERLE pre | `10 log10(mean mic^2 / mean residual^2)` over frames 150-240 (2.40-3.84 s), before double talk |
| ERLE post | same over frames 480-620 (7.68-9.92 s), after double talk |
| Echo ERLE | `10 log10(mean echo^2 / mean (residual - known noise)^2)`; removes the known noise component so cancellation can be judged at low SNR |
| Recovery | first frame after double-talk end where the 10-frame echo ERLE is within 3 dB of its pre value, or `None` if it never is |
| Near gain | least-squares amplitude projection of the residual onto the known near-end signal over the double-talk segment, in dB (0 dB = untouched) |
| DT flag rate | percentage of near-end-active frames in the double-talk segment where the detector froze adaptation; a frame is active when its near-end power is >= 10 % of the segment mean |
| Norm peak | maximum `filter_norm()` over the run; divergence bound 10 (unit-energy room needs about 1) |
| CPU | wall-clock time of the `Aec::process` loop divided by frames, in ms/frame |

"ERLE collapse" below means post-window echo ERLE < 3 dB. The pre-window is
also a convergence window, so it is not used for the collapse flag.

## Per-scene results

The cell format is **mean [min, max] over the three seeds** (`00C0FFEE`,
`5EED1234`, `A5A55A5A`). `-` means not applicable or no finite sample. The
generated report also prints applied/bulk delay and the delay-step application
time; those two columns are omitted here for width. The "bulk" number is the
scene's requested bulk delay only, before the room response; the late-direct
response adds another 45 ms, so `applied/bulk` is not an error for that room.
The delay behaviour is discussed in the aggregate and "what degrades" sections.

| # | condition | ERLE pre (dB) | ERLE post (dB) | Echo ERLE pre (dB) | Echo ERLE post (dB) | Recovery (ms) | Near gain (dB) | DT flag (%) | Norm peak | CPU (ms/frame) | Flags |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | short-direct-20ms / no noise / static / no DT | 16.6 [16.4, 16.9] | 29.0 [28.5, 29.9] | 16.6 [16.4, 16.9] | 29.0 [28.5, 29.9] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 2 | office-60ms / no noise / static / no DT | 15.4 [14.5, 16.2] | 28.1 [27.5, 29.2] | 15.4 [14.5, 16.2] | 28.1 [27.5, 29.2] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 3 | hall-250ms / no noise / static / no DT | 11.5 [10.8, 12.1] | 15.9 [15.1, 16.8] | 11.5 [10.8, 12.1] | 15.9 [15.1, 16.8] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 4 | diffuse-sparse-150ms / no noise / static / no DT | 12.6 [11.9, 13.2] | 22.3 [21.5, 23.4] | 12.6 [11.9, 13.2] | 22.3 [21.5, 23.4] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 5 | late-direct-45ms / no noise / static / no DT | 12.1 [11.3, 12.7] | 17.6 [16.7, 18.9] | 12.1 [11.3, 12.7] | 17.6 [16.7, 18.9] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 6 | weak-direct-strong-early / no noise / static / no DT | 15.0 [14.5, 16.0] | 24.5 [23.2, 25.3] | 15.0 [14.5, 16.0] | 24.5 [23.2, 25.3] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 7 | office-60ms / white 30 dB / static / no DT | 15.3 [14.2, 16.2] | 25.2 [24.3, 25.8] | 15.5 [14.4, 16.4] | 27.4 [26.5, 28.6] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 8 | office-60ms / white 15 dB / static / no DT | 11.5 [10.4, 12.7] | 14.0 [13.2, 15.0] | 15.1 [13.9, 16.0] | 23.4 [23.0, 24.1] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 9 | office-60ms / white 5 dB / static / no DT | 4.8 [3.9, 5.6] | 5.4 [4.8, 6.1] | 12.3 [11.0, 13.3] | 15.1 [14.7, 15.5] | - | - | - | 0.5 [0.5, 0.5] | 0.0 [0.0, 0.0] | ok |
| 10 | office-60ms / pink 30 dB / static / no DT | 15.0 [14.2, 15.6] | 25.3 [24.6, 25.8] | 15.2 [14.4, 15.8] | 27.2 [26.6, 28.2] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 11 | office-60ms / pink 15 dB / static / no DT | 10.9 [9.9, 11.8] | 14.1 [13.4, 14.9] | 14.0 [13.3, 14.5] | 21.7 [20.9, 22.2] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 12 | office-60ms / pink 5 dB / static / no DT | 4.4 [3.5, 5.2] | 5.4 [4.8, 6.0] | 10.7 [10.2, 11.7] | 13.1 [12.3, 13.7] | - | - | - | 0.5 [0.5, 0.6] | 0.0 [0.0, 0.0] | ok |
| 13 | office-60ms / babble 30 dB / static / no DT | 15.1 [14.2, 15.7] | 24.9 [23.9, 25.5] | 15.2 [14.4, 15.9] | 26.6 [25.5, 28.0] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 14 | office-60ms / babble 15 dB / static / no DT | 10.8 [10.0, 11.3] | 13.9 [12.7, 15.4] | 13.0 [12.7, 13.2] | 20.3 [18.4, 21.8] | - | - | - | 0.5 [0.5, 0.5] | 0.0 [0.0, 0.0] | ok |
| 15 | office-60ms / babble 5 dB / static / no DT | 3.7 [2.8, 4.9] | 4.6 [4.1, 5.0] | 6.6 [5.4, 8.4] | 10.3 [5.9, 14.0] | - | - | - | 0.6 [0.5, 0.7] | 0.0 [0.0, 0.0] | ok |
| 16 | hall-250ms / white 30 dB / static / no DT | 11.3 [10.6, 12.1] | 15.7 [14.7, 16.6] | 11.4 [10.7, 12.2] | 15.9 [14.8, 16.8] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 17 | hall-250ms / white 15 dB / static / no DT | 9.6 [8.9, 10.3] | 12.0 [12.0, 12.1] | 11.3 [10.8, 11.9] | 15.0 [14.4, 15.4] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 18 | hall-250ms / white 5 dB / static / no DT | 4.5 [3.8, 5.1] | 5.4 [5.2, 5.7] | 9.8 [9.2, 10.5] | 12.5 [11.6, 13.1] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 19 | hall-250ms / pink 30 dB / static / no DT | 11.4 [10.7, 12.2] | 15.8 [15.0, 16.6] | 11.4 [10.8, 12.2] | 16.0 [15.1, 16.8] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 20 | hall-250ms / pink 15 dB / static / no DT | 9.5 [8.7, 10.2] | 12.0 [12.0, 12.0] | 11.2 [10.8, 11.8] | 14.7 [14.1, 15.0] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 21 | hall-250ms / pink 5 dB / static / no DT | 4.2 [3.5, 4.7] | 5.3 [5.2, 5.6] | 9.3 [9.0, 9.6] | 11.2 [11.1, 11.3] | - | - | - | 0.5 [0.5, 0.5] | 0.0 [0.0, 0.0] | ok |
| 22 | hall-250ms / babble 30 dB / static / no DT | 11.4 [10.7, 12.0] | 15.7 [14.9, 16.5] | 11.5 [10.7, 12.1] | 15.9 [15.0, 16.8] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 23 | hall-250ms / babble 15 dB / static / no DT | 9.1 [8.6, 9.5] | 11.5 [11.2, 12.1] | 10.3 [10.0, 10.6] | 13.8 [13.7, 13.9] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 24 | hall-250ms / babble 5 dB / static / no DT | 3.5 [2.7, 4.2] | 4.6 [3.9, 5.0] | 5.9 [4.9, 6.9] | 9.2 [5.5, 11.6] | - | - | - | 0.6 [0.5, 0.7] | 0.0 [0.0, 0.0] | ok |
| 25 | short-direct-20ms / babble 5 dB / static / no DT | 3.7 [2.7, 4.4] | 4.7 [3.9, 5.2] | 6.6 [5.5, 7.8] | 10.5 [5.6, 14.1] | - | - | - | 0.6 [0.5, 0.7] | 0.0 [0.0, 0.0] | ok |
| 26 | diffuse-sparse-150ms / babble 5 dB / static / no DT | 3.0 [2.4, 3.6] | 4.6 [4.0, 5.0] | 4.9 [3.3, 5.8] | 9.1 [5.8, 13.1] | - | - | - | 0.6 [0.4, 0.7] | 0.0 [0.0, 0.0] | ok |
| 27 | short-direct-20ms / no noise / step+5ms / no DT | 16.6 [16.4, 16.9] | 13.9 [12.8, 15.0] | 16.6 [16.4, 16.9] | 13.9 [12.8, 15.0] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 28 | office-60ms / no noise / step+5ms / no DT | 15.4 [14.5, 16.2] | 13.8 [13.5, 14.4] | 15.4 [14.5, 16.2] | 13.8 [13.5, 14.4] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 29 | hall-250ms / no noise / step+5ms / no DT | 11.5 [10.8, 12.1] | 9.7 [6.3, 11.6] | 11.5 [10.8, 12.1] | 9.7 [6.3, 11.6] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 30 | diffuse-sparse-150ms / no noise / step+5ms / no DT | 12.6 [11.9, 13.2] | 8.3 [5.6, 10.8] | 12.6 [11.9, 13.2] | 8.3 [5.6, 10.8] | - | - | - | 0.4 [0.3, 0.4] | 0.0 [0.0, 0.0] | ok |
| 31 | late-direct-45ms / no noise / step+5ms / no DT | 12.1 [11.3, 12.7] | 11.3 [8.5, 13.6] | 12.1 [11.3, 12.7] | 11.3 [8.5, 13.6] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 32 | weak-direct-strong-early / no noise / step+5ms / no DT | 15.0 [14.5, 16.0] | 11.6 [9.6, 12.7] | 15.0 [14.5, 16.0] | 11.6 [9.6, 12.7] | - | - | - | 0.4 [0.3, 0.4] | 0.0 [0.0, 0.0] | ok |
| 33 | short-direct-20ms / no noise / drift200ppm / no DT | 8.2 [6.2, 10.6] | 26.3 [24.0, 28.9] | 8.2 [6.2, 10.6] | 26.3 [24.0, 28.9] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 34 | office-60ms / no noise / drift200ppm / no DT | 7.2 [4.8, 8.7] | 24.8 [22.1, 28.1] | 7.2 [4.8, 8.7] | 24.8 [22.1, 28.1] | - | - | - | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 35 | hall-250ms / no noise / drift200ppm / no DT | 6.3 [4.8, 8.6] | 15.4 [14.8, 16.0] | 6.3 [4.8, 8.6] | 15.4 [14.8, 16.0] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 36 | diffuse-sparse-150ms / no noise / drift200ppm / no DT | 5.8 [5.0, 6.6] | 19.9 [19.5, 20.5] | 5.8 [5.0, 6.6] | 19.9 [19.5, 20.5] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 37 | late-direct-45ms / no noise / drift200ppm / no DT | 6.7 [6.1, 7.0] | 15.6 [14.0, 16.4] | 6.7 [6.1, 7.0] | 15.6 [14.0, 16.4] | - | - | - | 0.4 [0.4, 0.4] | 0.0 [0.0, 0.0] | ok |
| 38 | weak-direct-strong-early / no noise / drift200ppm / no DT | 8.1 [6.2, 10.0] | 24.7 [23.7, 26.4] | 8.1 [6.2, 10.0] | 24.7 [23.7, 26.4] | - | - | - | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 39 | office-60ms / no noise / static / DT -6 dB | 15.4 [14.5, 16.2] | 24.6 [23.4, 26.2] | 15.4 [14.5, 16.2] | 24.6 [23.4, 26.2] | 16.0 [0.0, 48.0] | 0.0 [-0.0, 0.0] | 67.9 [64.0, 72.1] | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 40 | office-60ms / no noise / static / DT +0 dB | 15.4 [14.5, 16.2] | 23.5 [21.4, 24.6] | 15.4 [14.5, 16.2] | 23.5 [21.4, 24.6] | 21.3 [0.0, 48.0] | 0.0 [-0.0, 0.0] | 80.4 [70.8, 90.7] | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 41 | office-60ms / no noise / static / DT +3 dB | 15.4 [14.5, 16.2] | 22.5 [20.0, 24.1] | 15.4 [14.5, 16.2] | 22.5 [20.0, 24.1] | 26.7 [0.0, 64.0] | 0.0 [0.0, 0.0] | 84.2 [71.9, 91.9] | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 42 | hall-250ms / no noise / static / DT -6 dB | 11.5 [10.8, 12.1] | 15.5 [14.6, 16.5] | 11.5 [10.8, 12.1] | 15.5 [14.6, 16.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 63.7 [57.3, 67.4] | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 43 | hall-250ms / no noise / static / DT +0 dB | 11.5 [10.8, 12.1] | 15.2 [14.1, 16.3] | 11.5 [10.8, 12.1] | 15.2 [14.1, 16.3] | 21.3 [0.0, 64.0] | -0.0 [-0.0, 0.0] | 76.6 [64.0, 88.4] | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 44 | hall-250ms / no noise / static / DT +3 dB | 11.5 [10.8, 12.1] | 15.0 [13.7, 16.3] | 11.5 [10.8, 12.1] | 15.0 [13.7, 16.3] | 21.3 [0.0, 64.0] | 0.0 [0.0, 0.0] | 86.1 [75.3, 91.9] | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 45 | weak-direct-strong-early / no noise / static / DT -6 dB | 15.0 [14.5, 16.0] | 22.4 [21.8, 23.5] | 15.0 [14.5, 16.0] | 22.4 [21.8, 23.5] | 21.3 [0.0, 64.0] | -0.0 [-0.1, -0.0] | 68.3 [60.7, 77.9] | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 46 | weak-direct-strong-early / no noise / static / DT +0 dB | 15.0 [14.5, 16.0] | 21.2 [20.5, 21.6] | 15.0 [14.5, 16.0] | 21.2 [20.5, 21.6] | 26.7 [0.0, 64.0] | -0.0 [-0.0, -0.0] | 79.3 [67.4, 90.7] | 0.5 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 47 | weak-direct-strong-early / no noise / static / DT +3 dB | 15.0 [14.5, 16.0] | 20.4 [19.1, 21.2] | 15.0 [14.5, 16.0] | 20.4 [19.1, 21.2] | 112.0 [0.0, 320.0] | -0.0 [-0.0, 0.0] | 83.5 [70.8, 94.2] | 0.5 [0.4, 0.6] | 0.0 [0.0, 0.0] | ok |
| 48 | hall-250ms / babble 5 dB / drift200ppm / DT +0 dB | 2.0 [1.5, 2.4] | 3.7 [3.1, 4.4] | 3.1 [2.6, 3.5] | 6.2 [4.3, 8.3] | 133.3 [0.0, 320.0] | 0.0 [-0.0, 0.0] | 27.9 [0.0, 83.7] | 0.6 [0.5, 0.7] | 0.0 [0.0, 0.0] | ok |
| 49 | hall-250ms / white 15 dB / step+5ms / DT -6 dB | 9.6 [8.9, 10.3] | 7.2 [6.0, 9.2] | 11.3 [10.8, 11.9] | 8.0 [6.4, 10.5] | 629.3 [128.0, 1408.0] | 0.0 [0.0, 0.1] | 71.3 [59.6, 78.7] | 0.4 [0.3, 0.4] | 0.0 [0.0, 0.0] | ok |
| 50 | diffuse-sparse-150ms / babble 5 dB / drift200ppm / DT +3 dB | 1.7 [0.8, 2.4] | 3.6 [3.2, 4.5] | 2.5 [1.2, 3.6] | 5.8 [4.3, 7.7] | 149.3 [0.0, 336.0] | -0.1 [-0.1, -0.0] | 0.0 [0.0, 0.0] | 0.8 [0.8, 0.9] | 0.0 [0.0, 0.0] | ok |
| 51 | office-60ms / pink 15 dB / drift200ppm / DT +0 dB | 5.4 [4.2, 7.6] | 12.1 [11.1, 13.5] | 6.0 [4.7, 8.5] | 15.8 [13.7, 19.9] | 21.3 [0.0, 64.0] | -0.0 [-0.0, 0.0] | 57.6 [0.0, 91.9] | 0.5 [0.4, 0.6] | 0.0 [0.0, 0.0] | ok |
| 52 | late-direct-45ms / babble 15 dB / static / DT -6 dB | 9.7 [9.6, 9.8] | 11.9 [11.7, 12.2] | 10.9 [10.6, 11.5] | 14.6 [14.1, 14.9] | 0.0 [0.0, 0.0] | -0.1 [-0.2, 0.0] | 63.7 [37.1, 80.9] | 0.4 [0.4, 0.5] | 0.0 [0.0, 0.0] | ok |
| 53 | weak-direct-strong-early / white 5 dB / step+5ms / DT +3 dB | 4.7 [3.7, 5.2] | 4.1 [3.7, 4.7] | 11.7 [10.7, 12.4] | 7.5 [6.8, 7.9] | 618.7 [336.0, 912.0] | -0.0 [-0.1, 0.1] | 6.2 [0.0, 18.6] | 0.9 [0.8, 0.9] | 0.0 [0.0, 0.0] | ok |

## Aggregates

### By room family

| Room family | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short-direct-20ms | 12 | 11.3 [2.7, 16.9] | 18.5 [3.9, 29.9] | 12.0 [5.5, 16.9] | 19.9 [5.6, 29.9] | 0.5 [0.4, 0.7] |
| office-60ms | 48 | 11.3 [2.8, 16.2] | 17.6 [4.1, 29.2] | 13.0 [4.7, 16.4] | 21.1 [5.9, 29.2] | 0.5 [0.4, 0.7] |
| hall-250ms | 51 | 8.8 [1.5, 12.2] | 11.5 [3.1, 16.8] | 10.0 [2.6, 12.2] | 13.2 [4.3, 16.8] | 0.5 [0.3, 0.7] |
| diffuse-sparse-150ms | 15 | 7.1 [0.8, 13.2] | 11.8 [3.2, 23.4] | 7.7 [1.2, 13.2] | 13.1 [4.3, 23.4] | 0.5 [0.3, 0.9] |
| late-direct-45ms | 12 | 10.1 [6.1, 12.7] | 14.1 [8.5, 18.9] | 10.4 [6.1, 12.7] | 14.7 [8.5, 18.9] | 0.4 [0.4, 0.5] |
| weak-direct-strong-early | 21 | 12.5 [3.7, 16.0] | 18.4 [3.7, 26.4] | 13.5 [6.2, 16.0] | 18.9 [6.8, 26.4] | 0.5 [0.3, 0.9] |

### By background noise

| Noise | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 81 | 12.4 [4.8, 16.9] | 19.0 [5.6, 29.9] | 12.4 [4.8, 16.9] | 19.0 [5.6, 29.9] | 0.4 [0.3, 0.6] |
| white 30 dB | 6 | 13.3 [10.6, 16.2] | 20.5 [14.7, 25.8] | 13.4 [10.7, 16.4] | 21.6 [14.8, 28.6] | 0.5 [0.4, 0.5] |
| white 15 dB | 9 | 10.2 [8.9, 12.7] | 11.1 [6.0, 15.0] | 12.5 [10.8, 16.0] | 15.5 [6.4, 24.1] | 0.4 [0.3, 0.5] |
| white 5 dB | 9 | 4.6 [3.7, 5.6] | 4.9 [3.7, 6.1] | 11.3 [9.2, 13.3] | 11.7 [6.8, 15.5] | 0.6 [0.4, 0.9] |
| pink 30 dB | 6 | 13.2 [10.7, 15.6] | 20.5 [15.0, 25.8] | 13.3 [10.8, 15.8] | 21.6 [15.1, 28.2] | 0.5 [0.4, 0.5] |
| pink 15 dB | 9 | 8.6 [4.2, 11.8] | 12.7 [11.1, 14.9] | 10.4 [4.7, 14.5] | 17.4 [13.7, 22.2] | 0.5 [0.4, 0.6] |
| pink 5 dB | 6 | 4.3 [3.5, 5.2] | 5.4 [4.8, 6.0] | 10.0 [9.0, 11.7] | 12.1 [11.1, 13.7] | 0.5 [0.5, 0.6] |
| babble 30 dB | 6 | 13.2 [10.7, 15.7] | 20.3 [14.9, 25.5] | 13.3 [10.7, 15.9] | 21.3 [15.0, 28.0] | 0.5 [0.4, 0.5] |
| babble 15 dB | 9 | 9.9 [8.6, 11.3] | 12.4 [11.2, 15.4] | 11.4 [10.0, 13.2] | 16.2 [13.7, 21.8] | 0.5 [0.4, 0.5] |
| babble 5 dB | 18 | 2.9 [0.8, 4.9] | 4.3 [3.1, 5.2] | 4.9 [1.2, 8.4] | 8.5 [4.3, 14.1] | 0.6 [0.4, 0.9] |

### By delay behaviour

| Delay | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Norm peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 108 | 10.9 [2.4, 16.9] | 15.8 [3.9, 29.9] | 12.2 [3.3, 16.9] | 18.4 [5.5, 29.9] | 0.5 [0.4, 0.7] |
| step+5ms | 24 | 12.2 [3.7, 16.9] | 10.0 [3.7, 15.0] | 13.3 [10.7, 16.9] | 10.5 [5.6, 15.0] | 0.5 [0.3, 0.9] |
| drift200ppm | 27 | 5.7 [0.8, 10.6] | 16.2 [3.1, 28.9] | 6.0 [1.2, 10.6] | 17.2 [4.3, 28.9] | 0.5 [0.4, 0.9] |

### By double talk

| Near-end | Scenes | ERLE pre | ERLE post | Echo ERLE pre | Echo ERLE post | Recovery (ms) | Near gain (dB) | DT flag (%) | Norm peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 114 | 10.0 [2.4, 16.9] | 15.1 [3.9, 29.9] | 11.3 [3.3, 16.9] | 17.4 [5.5, 29.9] | - | - | - | 0.5 [0.3, 0.7] |
| -6 dB | 15 | 12.2 [8.9, 16.2] | 16.3 [6.0, 26.2] | 12.8 [10.6, 16.2] | 17.0 [6.4, 26.2] | 133.3 [0.0, 1408.0] | -0.0 [-0.2, 0.1] | 67.0 [37.1, 80.9] | 0.4 [0.3, 0.5] |
| +0 dB | 15 | 9.9 [1.5, 16.2] | 15.1 [3.1, 24.6] | 10.2 [2.6, 16.2] | 16.4 [4.3, 24.6] | 44.8 [0.0, 320.0] | -0.0 [-0.0, 0.0] | 64.4 [0.0, 91.9] | 0.5 [0.4, 0.7] |
| +3 dB | 15 | 9.6 [0.8, 16.2] | 13.1 [3.2, 24.1] | 11.2 [1.2, 16.2] | 14.2 [4.3, 24.1] | 185.6 [0.0, 912.0] | -0.0 [-0.1, 0.1] | 52.0 [0.0, 94.2] | 0.6 [0.4, 0.9] |

### Flags

| Flag | Count |
| --- | ---: |
| Runs | 159 |
| Non-finite output | 0 |
| Diverged (NaN, norm > 10, echo ERLE < -6 dB) | 0 |
| ERLE collapse (echo ERLE < 3 dB, post window) | 0 |
| Recovery not reached inside the window | 0 |

## Real rooms

The synthetic matrix was designed before any measured response was available.
This section is the check on that design: 18 **measured** room responses from
OpenSLR SLR28 are replayed through the same generator, the same 30 ms static
bulk delay and the same production AEC configuration. The synthetic matrix is
not replaced; it is still the only way to sweep a designed factor grid. The real
set is what says whether the designed grid was optimistic.

### Provenance and licence

* Dataset: OpenSLR SLR28, "Room Impulse Response and Noise Database"
  (`https://www.openslr.org/28/`), which aggregates the Aachen (AIR), 2014
  REVERB challenge and RWCP measurements.
* The distributor declares **Apache License 2.0**. That licence explicitly
  permits redistribution, including inside a repository whose own code has a
  different licence, as long as the Apache-2.0 text and the attribution are
  retained. Both are retained in
  `crates/cf-aec/tests/room/real/` (`LICENSE-APACHE-2.0.txt` and
  `PROVENANCE.md`); the distributor declares no NOTICE file, so there is no
  NOTICE content to propagate. The WAVs therefore stay in the repository and
  the suite stays offline.
* `crates/cf-aec/tests/room/real/PROVENANCE.md` records the source member,
  SHA-256 of every stored file and the exact extraction steps
  (`extract_slr28.py`). The stored set is 18 files, 535 KB.
* Each stored file is trimmed to the measured onset and keeps the measured tail
  up to 1.60 s. The loader accepts 16-bit PCM at 16 kHz or 48 kHz, any channel
  count, takes channel 0 and decimates 48 kHz to 16 kHz with a fixed
  windowed-sinc low-pass; a 44.1 kHz or non-PCM file is rejected rather than
  silently mis-timed. Every response is normalised to unit L2 energy, the same
  policy as the synthetic responses.

### Scenes and measurement

Each response gets four conditions at the standard static 30 ms bulk delay:

1. clean: speech-like reference through the measured response, no noise, no
   near end;
2. near-end speech at -6 dB relative to the echo;
3. near-end speech at +3 dB relative to the echo;
4. pink background noise at 15 dB echo-to-noise ratio.

That is 18 x 4 = 72 scenes and **216 runs** at the three standard seeds. The
production `AecConfig::default()` is used and clock-rate correction is on by
default; because every real scene has a static path, it never engaged
(0 / 216 runs), exactly as on the synthetic static scenes. The measurement
window is the same post-event window as the synthetic matrix: frames 480-620
(7.68-9.92 s), echo ERLE with the known noise component removed.

The RT20 column below is measured from the stored response by the loader
(Schroeder backward integration, -5 to -25 dB, scaled to 60 dB), not copied
from dataset metadata. It is a diagnostic: the 18 responses span 0.09-1.05 s,
and four of them are longer than anything the synthetic matrix models.

### Per measured response

Cells are the mean of three seeds at that condition, in dB. The bottom rows are
the mean, minimum and maximum across the 18 rooms (each room weighted equally).

| id | campaign | room / microphone | taps | RT20 (s) | C50 (dB) | clean | DT -6 dB | DT +3 dB | pink 15 dB | mean of 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| air-booth | AIR | booth, 2.0 m | 10892 | 0.158 | 23.6 | 26.0 | 23.6 | 21.5 | 17.2 | 22.1 |
| rvb-smallroom1-near | REVERB | small room 1, near | 13867 | 0.094 | 29.4 | 28.1 | 24.0 | 22.8 | 21.6 | 24.1 |
| rvb-smallroom2-near | REVERB | small room 2, near | 15275 | 0.134 | 25.9 | 26.8 | 23.7 | 22.0 | 21.5 | 23.5 |
| rvb-smallroom2-far | REVERB | small room 2, far | 15207 | 0.260 | 19.8 | 20.8 | 19.6 | 18.7 | 18.7 | 19.5 |
| air-meeting | AIR | meeting room, 1.0 m | 10882 | 0.293 | 16.2 | 22.8 | 21.5 | 20.4 | 19.6 | 21.1 |
| rwcp-e2a | RWCP | room e2a, array ch 0 | 9912 | 0.275 | 21.0 | 25.0 | 22.5 | 21.1 | 20.9 | 22.4 |
| rwcp-e1b | RWCP | room e1b, array ch 0 | 9940 | 0.372 | 12.2 | 16.3 | 15.7 | 15.0 | 14.8 | 15.5 |
| rwcp-e1c | RWCP | room e1c, array ch 0 | 11937 | 0.425 | 10.8 | 16.5 | 15.8 | 15.2 | 15.3 | 15.7 |
| rvb-mediumroom1-near | REVERB | medium room 1, near | 13810 | 0.337 | 17.3 | 19.8 | 19.1 | 18.0 | 18.1 | 18.7 |
| rvb-mediumroom2-near | REVERB | medium room 2, near | 13839 | 0.368 | 15.5 | 19.3 | 18.5 | 17.9 | 14.1 | 17.5 |
| air-office | AIR | office, 2.0 m | 10877 | 0.582 | 7.8 | 12.2 | 11.8 | 11.7 | 11.8 | 11.9 |
| rvb-largeroom1-near | REVERB | large room 1, near | 13950 | 0.473 | 15.1 | 14.2 | 14.0 | 13.6 | 13.3 | 13.8 |
| rvb-mediumroom2-far | REVERB | medium room 2, far | 13801 | 0.682 | 5.7 | 14.4 | 14.1 | 14.1 | 13.6 | 14.1 |
| air-stairway | AIR | stairway, 1.0 m | 25600 | 0.770 | 7.7 | 12.4 | 12.3 | 12.1 | 11.3 | 12.0 |
| rwcp-ofc | RWCP | room ofc, array ch 0 | 24903 | 0.804 | 6.2 | 9.2 | 9.1 | 8.9 | 9.2 | 9.1 |
| air-lecture | AIR | lecture room, 2.0 m | 21736 | 0.878 | 4.3 | 8.6 | 8.6 | 8.7 | 8.4 | 8.6 |
| rvb-largeroom2-near | REVERB | large room 2, near | 15274 | 0.622 | 15.3 | 16.3 | 16.0 | 15.4 | 14.7 | 15.6 |
| rvb-largeroom2-far | REVERB | large room 2, far | 15207 | 1.047 | 3.5 | 7.0 | 6.9 | 6.5 | 6.8 | 6.8 |
| **mean over 18 rooms** | | | | | | **17.6** | **16.5** | **15.8** | **15.0** | **16.2** |
| **min over 18 rooms** | | | | | | **7.0** | **6.9** | **6.5** | **6.8** | **6.5** |
| **max over 18 rooms** | | | | | | **28.1** | **24.0** | **22.8** | **21.6** | **28.1** |

### Aggregate

| Condition | Rooms | Runs | Echo ERLE post mean (dB) | min (dB) | max (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 18 | 54 | 17.6 | 7.0 | 28.1 |
| DT -6 dB | 18 | 54 | 16.5 | 6.9 | 24.0 |
| DT +3 dB | 18 | 54 | 15.8 | 6.5 | 22.8 |
| pink 15 dB | 18 | 54 | 15.0 | 6.8 | 21.6 |
| all four conditions | 18 | 216 | 16.2 | 6.8 | 24.1 |

The mean/min/max here are across the 18 room means (each room weighted
equally), which is why they differ slightly from the per-run extremes. By
campaign the means are 15.1 dB (AIR, 5 rooms), 17.1 dB (REVERB, 9) and 15.7 dB
(RWCP, 4); the campaigns are not balanced by room type, so this is descriptive,
not a ranking.

| RT20 band | Rooms | Clean mean (dB) | All-condition mean (dB) | min (dB) | max (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| < 0.20 s | 3 | 27.0 | 23.2 | 22.1 | 24.1 |
| 0.20-0.40 s | 6 | 20.7 | 19.1 | 15.5 | 22.4 |
| 0.40-0.70 s | 5 | 14.8 | 14.2 | 11.9 | 15.7 |
| >= 0.70 s | 4 | 9.3 | 9.1 | 6.8 | 12.0 |

Stability and near-end preservation hold on every one of the 216 runs:

| Flag | Count |
| --- | ---: |
| Non-finite output | 0 |
| Diverged (NaN, norm > 10, echo ERLE < -6 dB) | 0 |
| ERLE collapse (echo ERLE post < 3 dB) | 0 |
| Recovery not reached inside the window | 0 |
| Clock-rate correction engaged | 0 |
| Worst filter norm | 0.69 |

Near-end gain over the 108 double-talk runs is -0.0 dB mean, worst -0.2 dB;
recovery after double talk is 17.6 ms mean, worst 96 ms; the double-talk
detector flagged 74.4 % of near-end-active frames. The real rooms therefore do
not expose a stability or double-talk defect.

### Real vs synthetic

The synthetic numbers below are recomputed by the same report from the same
code, not copied from the table above. Each condition uses the synthetic scenes
with the same definition (clean baseline, static delay, no noise; or -6/+3 dB
near end; or pink 15 dB). The "all four" row weights the four conditions
equally, matching the real-room table.

| Condition | Real mean | Real min | Real max | Synthetic mean | Synthetic min | Synthetic max | Delta (real - synthetic) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 17.6 | 7.0 | 28.1 | 22.9 | 15.1 | 29.9 | **-5.3** |
| DT -6 dB | 16.5 | 6.9 | 24.0 | 20.8 | 14.6 | 26.2 | **-4.4** |
| DT +3 dB | 15.8 | 6.5 | 22.8 | 19.3 | 13.7 | 24.1 | **-3.6** |
| pink 15 dB | 15.0 | 6.8 | 21.6 | 18.2 | 14.1 | 22.2 | **-3.1** |
| all four conditions | 16.2 | 6.8 | 24.1 | 20.3 | 13.7 | 29.9 | **-4.1** |

Synthetic family reference (all runs of the existing matrix, plus the clean
baseline of each family):

| Room family | Runs | Echo ERLE post mean (dB) | min (dB) | max (dB) | Clean mean (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-direct-20ms | 12 | 19.9 | 5.6 | 29.9 | 29.0 |
| office-60ms | 48 | 21.1 | 5.9 | 29.2 | 28.1 |
| hall-250ms | 51 | 13.2 | 4.3 | 16.8 | 15.9 |
| diffuse-sparse-150ms | 15 | 13.1 | 4.3 | 23.4 | 22.3 |
| late-direct-45ms | 12 | 14.7 | 8.5 | 18.9 | 17.6 |
| weak-direct-strong-early | 21 | 18.9 | 6.8 | 26.4 | 24.5 |

**Verdict: the synthetic numbers were optimistic, not pessimistic.** The
synthetic aggregate is 4.1 dB above the measured aggregate, and the clean
baseline is 5.3 dB above. There is no condition in which the measured rooms do
better than the synthetic equivalent. The gap is not uniform:

* On short measured rooms (RT20 < 0.20 s) the clean mean is **27.0 dB**, within
  about 1-2 dB of the synthetic short-direct (29.0) and office (28.1) baselines.
  For the room lengths the synthetic matrix actually represents, it is close.
* On measured rooms at or above 0.70 s the clean mean is **9.3 dB** (6.8-12.0
  across conditions). The synthetic matrix has **no response that long** - its
  longest family is the 256 ms hall - and its clean minimum over all families is
  15.1 dB. So for the upper half of the measured set the synthetic suite is
  optimistic by roughly **6-8 dB**, and the real minimum (6.5 dB) is below every
  synthetic clean run.
* The synthetic "office-60ms" family is a 64 ms response, not a real office.
  The measured AIR office has RT20 0.582 s and posts 11.7-12.2 dB. Reading the
  synthetic office number as a real-office expectation would overstate
  cancellation by roughly 16 dB.
* The near/far microphone pairs are harder at the far position: small room 2
  near 26.8 dB clean vs far 20.8 dB, medium room 2 near 19.3 vs far 14.4, large
  room 2 near 16.3 vs far 7.0. The synthetic matrix has no microphone-distance
  factor, so it cannot see this loss.

The algorithm itself is not invalidated: every real run stays above the 3 dB
collapse bound, no filter diverges, near-end gain stays inside +-0.2 dB, and
double talk recovers within 96 ms. What was optimistic was the **room
population**: the synthetic grid under-samples long, diffuse measured rooms and
has no far-mic condition. No `cf-aec` or `cf-echo-pipeline` algorithm was
changed because of the real set; the evidence did not expose a defect, only a
coverage gap in the previous evidence.

## Fixes applied to `cf-aec`

The suite was written first; these are the changes the measurements forced.
No assertion threshold was relaxed to make the table look better.

### 1. Relative regularization floor (fixes 5 dB SNR divergence)

The old update used only the configured absolute regularization (`1e-2`).
That is negligible next to a speech bin, but larger than a spectral valley or a
pause, so in those bins the noise-driven update was effectively unregularized.
At 5 dB SNR several seeds grew the filter to norm 9-10 and produced
echo-cancellation ERLE between -3 and -8 dB: the canceller was adding energy.

`Aec::adapt` now adds a **relative floor** of
`0.1 * peak-hold(mean per-bin reference power)` to the configured value, where
the peak-hold decays by 0.999 per frame (16 s). The slow hold is deliberate: an
instantaneous mean collapses in a speech pause, which is exactly when the noise
floor would drive the update. Measured effect: filter norm stayed at or below
0.9 over all 159 runs, 5 dB runs report 4.3-5.4 dB raw ERLE (the physical
ceiling is about 6.2 dB) and 8.5-12.1 dB noise-removed echo ERLE. The cost is clean-room
steady-state ERLE: office fell from ~33 dB in the first room-matrix run to
~28 dB. That is a real trade-off, not a free win; it is still well above the
20 dB assertion.

### 2. Double-talk detector: threshold and bounded freeze

Two failure modes were measured:

* the old ratio threshold (0.3) fired on **echo-only drift** runs when the
  filter was slightly misaligned (ratio 0.3-0.4, coherence 0.65-0.70),
  freezing adaptation; clean hall/diffuse drift could then end below 0 dB;
* after a genuine double-talk burst with drift, the stale echo estimate kept
  the detector active for 184 frames after the near end stopped (381 active
  frames in a 10 s file), so the filter never recovered.

Changes: `DT_NEAR_END_RATIO` is now **0.5**; a freeze is capped at
**32 frames followed by 16 unconditional adaptation frames**
(`DT_MAX_FREEZE_FRAMES` / `DT_RELEASE_FRAMES`, private to the detector).
The cold-start guard and the 4-frame hangover are unchanged. At the time
(before the clock-rate correction existed) this took the office drift scene from
-2.9 dB post ERLE with the old rule to +5.1 dB; the remaining clean drift
collapse was tracking/tail-limited rather than a detector freeze, and the rate
correction below removed it. Near-end gain stayed at 0.0 dB mean (worst
-0.2 dB), and detection on near-end-active frames is 67.9 % at -6 dB, 80.4 % at
0 dB and 84.2 % at +3 dB on the clean office scenes (67.0 / 64.4 / 52.0 % when
every double-talk scene is pooled, in the same order).
The combined drift + noise + double-talk scenes remained the weak spot
(scenes 48, 50, 51); the cap shortened the deadlock but did not cure the
stationary-floor false detections, which are fix 6 below.

### 3. Larger causal guard for the bulk delay (fixes weak-direct alignment)

The correlation peak in a reflective room can sit on an early reflection tens
of samples after the direct arrival. With the old 32-sample guard, the applied
delay was later than the direct path and the leading taps were anti-causal.
The guard is now **96 samples**. Measured effect: the clean
weak-direct-strong-early scene posts 24.5 dB mean post ERLE (23.2 dB worst seed)
and no longer holds the filter at a late alignment; the 640-sample delay
assertion still passes (applied 540, error 100, within one 256-sample frame).
The trade-off is that the filter models 96 extra leading taps; the hall,
diffuse and late-direct tails are already longer than the 128 ms filter and
lose a little modelled tail.

The delay report changed from 609 to 540 samples because of this guard; that is
a bias, not an estimation error (the onset is at the true lag, the applied value
is deliberately 96 samples earlier).

### 4. GCC-PHAT first onset and a two-tier re-lock gate (fixes the late-reflection lock)

The old estimator applied the global plain-correlation peak. A diffuse or
late-reflection room can make a reflection stronger than the direct path, so the
estimator locked onto the reflection: the clean diffuse A5 seed applied 1684
samples for a true bulk delay of 480 and finished at **-4.4 dB** post ERLE.

The estimator now whitens the cross-spectrum (GCC-PHAT) with a relative
regularization floor, which removes most of the pitch-period structure of the
speech-like reference, and then takes the **first onset** whose smoothed PHAT
magnitude reaches 55 % of the maximum instead of the global peak. The plain
normalized correlation is kept only as a detection gate and as the quality
measure for a re-lock. The onset is scored within one decimated bin (four raw
samples): the PHAT lag is resolved on a four-sample grid, so scoring the exact
bin alone rejects a genuine step whenever the grid rounds one bin early.

A re-lock is gated **per measurement**, not only when the pair confirms:

* a move of at most one frame (256 samples), forward or backward, is
  **tracking** and needs the pre-existing 0.25 quality ratio;
* a move forwards of more than one frame must reach the pre-existing 0.5 ratio
  and needs **three** agreeing measurements;
* a move backwards of more than one frame is refused outright. The onset scan
  walks from lag zero upwards, so a periodic reference faults at a pitch period
  *before* the direct arrival; that artefact is always a large negative move.

Every measurement that could join the pair must pass; a rejected measurement
restarts the count. On confirmation the pair applies the candidate with the
better quality, not whichever arrived last. No threshold was relaxed: the 0.25
and 0.5 ratios are the old values. Direction asymmetry, the extra confirming
measurement and the first-onset rule are the new parts.

Measured effect: the clean diffuse scene posts **22.3 dB** mean post ERLE (every
seed above 21; A5 is **+22.1 dB**) instead of 13.2 dB with one seed at -4.4 dB.
Its applied delay is 380, i.e. onset 476 minus the 96-sample guard, for a true
480. The diffuse family noise-removed mean rises from 4.3 to **9.2 dB**.

### 5. Clock-rate estimation and a fractional reference resampler (fixes the drift collapse)

`DRIFT-SPLIT.md` localised the drift loss and showed that an exact rate
correction recovers the static bound, so the fix is to estimate the
reference-to-microphone clock ratio and remove it before adaptation.

* **Estimator** (`src/rate.rs`). Every bulk-delay analysis already produces a
  whitened correlation profile. A slow estimator takes the **sub-sample lag of
  the dominant peak** of that profile (three-point parabola over the smoothed
  profile, tracked within +-24 samples of the previous measurement so it stays
  on one arrival) and fits a straight line through the last 16 measurements,
  giving samples per sample, reported in ppm. The fit is robust: a first fit,
  MAD-based rejection of measurements off the line, a refit, a residual-RMS gate
  (which is what stops a delay *step* being read as a ramp), a required
  displacement of 4 samples over the window, 3 standard errors of slope
  significance, and agreement of both half-windows. The estimate is clamped to
  +-500 ppm, smoothed exponentially with a 60 ppm per-update slew limit, and set
  to zero inside a +-5 ppm dead band. A measurement that jumps by 32 samples
  restarts the window instead of being mixed with the previous coordinate.
* **Resampler** (`DelayEstimator::aligned_block`). The raw reference history is
  untouched, so the estimator always sees the full ratio; the **aligned** path -
  what the adaptive filter and the neural model see - is read out with a
  four-point cubic (Lagrange) interpolant at a position that advances by
  `1 - rate` samples per output sample. The whole and fractional parts of the
  accumulated warp are carried across frames, so enabling the correction,
  changing the estimate and dropping into the dead band never step the signal;
  the only alignment change is the interpolant's two-sample lookahead, once.
  The path stays causal: the warp is clamped so a tap is never a sample that has
  not been pushed yet.
* **Integration**. The correction is on by default, bypassed (bit-identical
  copy) until the estimate has converged, and it never resets the adaptive
  filter. While it is acquiring it runs the delay analysis every 16 frames
  instead of 64, bounded to 40 extra analyses (10.2 s) and four times the
  estimate's measurement rate; a static stream spends that budget and falls
  back to the old cadence without ever engaging. `Aec::set_clock_rate_correction`
  switches it, `Aec::clock_rate_ppm` and `Aec::clock_rate_corrected` report it.

Measured effect: with the rate correction alone, drift echo ERLE post rises from
**3.6 dB** (min -1.0) to **16.8 dB** (min 4.3, max 28.9), no drift run collapses,
and every static and step scene is unchanged to the last displayed digit (the
correction never engages on a path with no rate). It engages at frame **197**
mean [160, 368]; its applied rate settles within a few ppm of the true 200, with
a mean absolute residual of **3.9 ppm** over the post window (worst one-seed
sample 32.5 ppm while still acquiring). Its +-5 ppm dead band keeps a static
stream uncorrected, and the 60 ppm per-update slew limit bounds how fast the
applied rate can move. CPU: 1.1 us per 16 ms frame for the resampler, 0.6 us
amortised for the extra acquisition analyses.

### 6. Double-talk detector: stationary-floor gate (fixes the false freeze)

The detector's ratio and coherence tests fire during a far-end pause: the echo
estimate is near zero, so the ratio is one and the coherence is undefined
(zero), which is indistinguishable from near-end speech. With stationary
background noise that happens often, and in a drifted stream it keeps the filter
from re-converging after double talk.

Measurement on the office / pink 15 dB / double-talk scene (before the fix):

* adaptation was frozen on **48 %** of post-window frames, in 8 runs on average,
  mean length 8.5 frames, longest 15: not the 32-frame latch the old cap
  targeted, so the cap never engaged;
* **40 %** of those frozen frames were literal far-end pauses and **67 %** had a
  microphone power within 2x of the true background noise power, while only
  **20 %** of the freezes during real double talk were that close to the floor;
* disabling the freeze after the double-talk segment (ablation) gained **1.8 dB**
  of post-window echo ERLE; disabling it everywhere made the scene *worse*
  (12.15 dB), so the freeze during real double talk is protective and must stay.

Changes in `src/dtd.rs`:

* a slow microphone-power **noise floor** with fast-down / slow-up tracking
  (`DT_NOISE_FLOOR_RISE` = 0.001 per frame, a 16 s upward time constant), so it
  sits on the quietest (pause) frames and cannot climb through speech;
* a frame may only be near-end if the microphone power is at least
  `DT_NOISE_FLOOR_GATE` = 3x that floor. A far-end pause with noise only is no
  longer near-end; real speech, and a near end that is active while the far end
  is silent, stand far above it;
* the gate is armed only when the stream has a real stationary floor: the
  quietest valid frame must be between `DT_NOISE_PRESENCE_SHARE` = 0.25 % and
  `DT_NOISE_PRESENCE_MAX_SHARE` = 50 % of the slow microphone mean, after
  `DT_NOISE_PRESENCE_SETTLE_FRAMES` = 64 frames. A clean room's quiet frames are
  decaying room tail, measured at 0.038 % of the mean on the clean hall versus
  1.0-2.7 % on the noise scenes, so the clean static controls keep the old rule
  (they are unchanged to the last displayed digit).

Measured effect: post-window freeze on the target scene drops from **48 % to
9 %** (runs 3.0, mean length 3.8, longest 4), the scene posts **15.8 dB** echo
ERLE instead of 12.5, and its matched static control is **19.9 dB**. The
remaining freeze costs about **0.3 dB** by the same ablation. The 5 dB
babble + double-talk scenes are unchanged (6.2 / 5.8 dB), clean office / hall /
diffuse static are unchanged (28.1 / 15.9 / 22.3 dB), and no drift run collapses.
The cost is on noise-only low-SNR scenes: because the gate releases adaptation
on frames that are only background noise, the hall white / pink 5 dB scenes lose
1.2 dB of noise-removed echo ERLE (raw ERLE stays at the ~5.5 dB noise
ceiling), hall white / pink 15 dB lose 0.7 dB, and office white / pink 5 dB lose
0.5-0.8 dB. That is a real trade-off; those scenes are noise-bound, and the
alternative (never gating) is the 3.4 dB target loss above.

## What holds

* **No NaN and no divergence** on any of the 159 runs; filter norm <= 0.9
  against a bound of 10.
* **Clean short and office rooms** cancel 29.0 / 28.1 dB mean post ERLE, with
  every seed above 27 dB.
* **Noise at 30 and 15 dB** is handled gracefully: noise-removed echo ERLE
  means 21.3-21.6 dB and 15.5-17.4 dB. At 5 dB the raw ERLE is noise-limited by
  construction; the noise-removed echo ERLE is still 8.5-12.1 dB.
* **Near-end speech is preserved**: mean gain 0.0 dB, worst -0.2 dB over 45
  double-talk runs, and filter norm stays at or below 0.9 even at +3 dB near-end.
* **Static double talk recovers**: office/hall/weak-direct static scenes reach
  the recovery criterion in 0-1408 ms depending on seed, and every double-talk
  run in the matrix now recovers (previously five never did).
* **Delay step** is absorbed on short and office rooms (reported at 2.10 s on
  every seed; post ERLE 13.9 / 13.8 dB mean).
* **Clock drift is removed**: the drift scenes average 17.2 dB echo ERLE post
  and none of them collapses, where every one of them did before the
  rate correction. The rate correction never engages on a static path, so the
  static and step scenes are otherwise unchanged by it.
* **Measured rooms are stable**: 216 runs over 18 real responses, 0 non-finite,
  0 divergence and 0 post-window collapse; near-end gain worst -0.2 dB and all
  double-talk runs recover within 96 ms.
* **Determinism** holds bit-for-bit for a replayed scene.
* **CPU** is below the 0.1 ms printed resolution in the test build, ~0.17 ms in
  the standalone release report, about 1 % of the 16 ms budget. The rate
  correction adds about 1.7 us per frame.

## What degrades

* **Long rooms are tail-limited.** The filter models 128 ms after the bulk
  delay; the hall has 256 ms of response, diffuse 150 ms and late-direct
  200 ms. Even a perfect 128 ms fit leaves late energy: the clean hall posts
  15.9 dB, clean late-direct 17.6 dB and clean diffuse 22.3 dB, and the family
  means that also include noise, steps and drift are 11.5 dB (hall), 11.8 dB
  (diffuse) and 14.1 dB (late-direct) raw ERLE post. The measured set is worse:
  the four rooms at RT20 >= 0.70 s post 6.8-12.0 dB, because their measured
  tails (up to 1.6 s) are much longer than the synthetic hall, and the 128 ms
  filter cannot reach them.
* **The synthetic room population was optimistic.** The matched aggregate is
  4.1 dB below the synthetic aggregate (clean 5.3 dB below); for rooms at or
  above 0.70 s the gap is about 6-8 dB, and the real minimum (6.5 dB) is below
  every synthetic clean run. The synthetic grid has no room longer than 256 ms,
  so it could not see this. This is a coverage limit of the evidence, not a
  newly discovered instability: no real run collapses or diverges.
* **Far-mic positions are not represented in the synthetic matrix.** The
  far-mic variants of the measured small, medium and large rooms post 4-10 dB
  below their near-mic siblings (26.8 vs 20.8, 19.3 vs 14.4, 16.3 vs 7.0 dB
  clean), and the synthetic matrix has no microphone-distance factor at all.
* **Clock drift is corrected, but not perfectly.** At 200 ppm the static path
  moves ~33 samples over 10 s. With the rate correction on, drift echo ERLE
  post averages **17.2 dB** (min 4.3) against a 18.4 dB static mean for the same
  scenes, single clean room means of 26.3 / 24.8 / 15.4 / 19.9 / 15.6 / 24.7 dB
  for short / office / hall / diffuse / late / weak. The remaining 1.2 dB is
  convergence lag: the estimator needs about 2 s of delay analyses before it
  engages, and the filter needs a few hundred frames of settled path after
  that. The applied rate settles within a few ppm, so the residual rate is not
  the limit.
* **The late-reflection lock is fixed in this matrix, but the gate is not a
  proof.** The diffuse A5 seed recovers at +22.1 dB, and a late reflection that
  is stronger than the direct path no longer steals the initial lock or an
  already-locked alignment. A persistent *forward* reflection that dominates the
  plain correlation for three consecutive measurements could still move the
  alignment; no scene in the matrix produces one. Large *backward* bulk-delay
  changes (more than one frame) are deliberately not tracked — the gate reads
  them as pitch artefacts, so a genuine one needs `Aec::set_delay_override`.
* **Tracking a small step can cost a filter reset.** The +5 ms step is now
  tracked — it is exactly the tracking tier that must be accepted — and a
  tracked change resets the adaptive filter. In the diffuse + step scene the
  reset lands at frames 431-559 and the scene posts 8.3 dB mean post ERLE, where
  the old estimator left the 80-sample step to the filter and posted 13.4 dB.
  Short and office step scenes still post 13.9 / 13.8 dB, and the diffuse family
  mean is 8.2 dB raw / 9.2 dB noise-removed, but this is a real cost of tracking
  late in the file rather than a free win.
* **Drift + noise + double talk is still the weakest combination.** Scenes 48,
  50 and 51 are no longer collapse flags (6.2 / 5.8 / 15.8 dB echo ERLE post)
  and they meet the recovery criterion (133.3 / 149.3 / 21.3 ms means;
  previously five runs never recovered), but they stay below the 18.4 dB static
  mean. At 5 dB SNR the detector's cold-start guard (6 dB of measured
  cancellation) cannot latch, so scenes 48 and 50 report a low DT flag rate and
  adaptation runs during the near end; the near-end gain is still 0.0 dB, but
  the echo estimate is not protected. Scene 53 (weak direct, white 5 dB, step,
  +3 dB near end) posts 4.1 dB raw ERLE and is not flagged.
* **The floor gate costs some noise-only low-SNR echo ERLE.** Frames that are
  only background noise are no longer frozen, so adaptation fits a little of
  that noise: hall white / pink 5 dB lose 1.2 dB of noise-removed echo ERLE,
  hall white / pink 15 dB lose 0.7 dB, and office white / pink 5 dB lose
  0.5-0.8 dB. Their raw ERLE is at the ~5.5 dB
  noise ceiling either way. The 5 dB babble + double-talk scenes and the clean
  static scenes are unchanged, and the target scene gains 3.4 dB; this is the
  price of that gain, not a free win.
* **5 dB SNR raw ERLE is capped near 6.2 dB** by the noise itself. Tables that
  show 4.3-5.4 dB there are not a cancellation failure; the noise-removed
  column is the relevant one.
* **Weak-direct and late-direct rooms** are usable but not strong (clean post
  24.5 and 17.6 dB mean); the weak-direct path only works because the
  96-sample guard makes the direct tap causal.

## What to fix next

1. **Clock-rate acquisition latency.** The correction removes the drift, but it
   needs about 2 s of delay analyses before it engages, so the first ~2 s of a
   drifted stream are still uncorrected and the filter has to re-converge after
   it engages. A shorter acquisition (a faster measurement cadence, or seeding
   the estimate from the delay step measurements) is the remaining drift work;
   `DRIFT-SPLIT.md` measures the 1.8 dB this costs against the perfect-rate
   ceiling.
2. **Tail model longer than 128 ms.** A second, slower partition set or a
   two-stage filter would cover the 150-256 ms synthetic rooms and the measured
   rooms above 0.70 s. The current 8 partitions are the main ceiling on
   hall/diffuse/late-direct ERLE and on the measured long rooms, independent of
   noise.
3. **Post-double-talk re-convergence and the detector trade-off.** With the
   false freeze removed the target scene still spends its post window catching
   up with drift acquisition; disabling even the remaining freeze after double
   talk buys only 0.3 dB. A rule that recognises a *stale filter* (rather than a
   stationary floor) as the reason for a high near-end ratio, or a two-filter
   fast/slow comparison, could keep the low-SNR noise-only behaviour and still
   protect the echo estimate in the combined scenes.
4. **A re-acquisition path for genuine large backward changes.** The two-tier
   gate refuses a backward move larger than one frame by design. A long
   confidence window (or an explicit `set_delay_override` by the host) is needed
   when the path really moves earlier, e.g. after a device or buffer change.
5. **Report CPU under load.** The per-scene CPU here is a single process on an
   idle development machine; it is not a worst-case or embedded measurement.
6. **Keep the measured rooms in the acceptance matrix.** The real set is the
   only check on the room population, and this run shows the synthetic grid was
   optimistic. Add the far-mic factor (and, if the licence terms allow, a few
   more long rooms) to the standing matrix so a future algorithm change cannot
   regress the long-room floor while the synthetic aggregate stays green.

## Reproduce

```text
cd cleanfield
cargo test -p cf-aec --test room_conditions -- --nocapture    # synthetic matrix + assertions
cargo run -p cf-aec --release --example room_report            # synthetic table
cargo test -p cf-aec --test real_rir_conditions -- --nocapture # measured rooms + assertions
cargo run -p cf-aec --release --example real_rir_report        # measured-room table
cargo test --workspace                                         # full workspace gate
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

The scene generator, measurements and reports live in
`crates/cf-aec/tests/room/`; the synthetic assertions are in
`crates/cf-aec/tests/room_conditions.rs` and the measured-room assertions in
`crates/cf-aec/tests/real_rir_conditions.rs`; the report examples are
`crates/cf-aec/examples/room_report.rs` and
`crates/cf-aec/examples/real_rir_report.rs`. The synthetic material is seeded
and `std`-only. The measured-room suite reads only the WAVs bundled in
`crates/cf-aec/tests/room/real/`; it needs no network and no download step.
