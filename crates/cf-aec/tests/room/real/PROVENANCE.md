# Real measured room impulse responses

These 18 files are the real-room evidence for `ROBUSTNESS.md`. They are
**measured** impulse responses, not synthetic ones. Every bundled file is a
single channel, 16 000 Hz, 16-bit PCM WAV, trimmed to the measured onset, with
the whole measured tail kept (longest 1.60 s). The loader also accepts 48 kHz
and multi-channel 16-bit PCM (it takes channel 0 and resamples 48 kHz to
16 kHz), so a later re-extraction does not require loader changes; none of the
bundled files currently exercises either path.

## Source and licence

* Dataset: **OpenSLR SLR28, "Room Impulse Response and Noise Database"**
  (<https://www.openslr.org/28/>).
* Archive: <https://www.openslr.org/resources/28/rirs_noises.zip> (1.3 GB).
* Licence declared by the distributor: **Apache License 2.0**, stated on the
  dataset's OpenSLR page. The full text is in `LICENSE-APACHE-2.0.txt` next to
  this file.
* Redistribution: Apache-2.0 explicitly permits copying, modification and
  redistribution, including in a repository whose own code carries a different
  licence, provided the Apache-2.0 text is retained and the notices/attribution
  are preserved. Both conditions are met in this directory: the licence text
  ships with the files and the three measurement campaigns are credited here and
  in `ROBUSTNESS.md`. The distributor declares no `NOTICE` file, so there is no
  NOTICE content to propagate. The files therefore stay in the repository; no
  download-at-test-time step is needed and the suite remains offline.
* Files used: the `RIRS_NOISES/real_rirs_isotropic_noises/` entries listed in
  the table below. That directory aggregates three measurement campaigns:
  the Aachen impulse response database (AIR), the 2014 REVERB challenge
  recordings, and the RWCP sound-scene database. The distributor's Apache-2.0
  declaration is the redistribution grant this tree relies on; the original
  campaigns are credited here and in `ROBUSTNESS.md`.
* Only the files in this directory are redistributed. Nothing outside its
  scope is copied into the repository.

## Processing

`extract_slr28.py` in this directory reproduces the files from the archive.
It performs exactly these steps per selected entry:

1. read channel 0 of the measured response (AIR and RWCP entries are
   binaural / array measurements; REVERB entries are 8-channel arrays);
2. remove the DC mean (the RWCP recordings carry a DC offset; the synthetic
   far-end reference is DC-blocked by construction, so the DC term is not a
   meaningful part of the echo path);
3. find the first sample within 1 % of the response peak, step back 8 samples,
   and drop everything before that point (leading measurement silence);
4. keep everything up to 1.60 s after that point and discard the rest;
5. quantise to 16-bit PCM at the native scale (no loudness normalisation in the
   file; the test loader applies the suite's unit-L2-energy policy).

The stored files are 535 KB in total. Each file is hashed below so the exact
material can be identified. The loader (`crates/cf-aec/tests/room/real.rs`)
parses channel 0 of a 16-bit PCM WAV, accepts 16 kHz or 48 kHz and any channel
count, decimates 48 kHz to 16 kHz with a fixed windowed-sinc low-pass, and
normalises to unit L2 energy, the same policy the synthetic responses use. It
rejects any other rate or sample format rather than silently changing the time
base.

## Selected responses

The table is in `RealRir::ALL` order, the order the reports use; it is not
re-sorted by RT60. `RT60` is the broadband T20 measured on the stored response
by the current loader (Schroeder backward integration, fit from -5 to -25 dB,
scaled by 3; see `RealRir::rt20_s`). `C50` is the 0-50 ms to 50 ms-late energy
ratio in dB, also measured on the stored response. Both are diagnostics, not
dataset metadata.

| id | campaign | room / microphone | source WAV | stored taps | RT60 (s) | C50 (dB) | SHA-256 (stored WAV) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| air-booth | AIR | booth, 2.0 m | air_type1_air_binaural_booth_1_2.wav | 10892 | 0.158 | 23.59 | `98b8cf31b571d82c3e7b9b56daf11efefe8ff6ff90ebbe901aa81433837fa6a8` |
| rvb-smallroom1-near | REVERB 2014 | small room 1, near mic | RVB2014_type1_rir_smallroom1_near_angla.wav | 13867 | 0.094 | 29.43 | `461dbc95e61851307c5ba3ccec3b1d452945da0381732add72dba7c8a4e77ac8` |
| rvb-smallroom2-near | REVERB 2014 | small room 2, near mic | RVB2014_type1_rir_smallroom2_near_angla.wav | 15275 | 0.134 | 25.93 | `71bef70c5e667787aef37b450ee13e478d3b7d9a7e8f0f0e603fc197ece9d8a5` |
| rvb-smallroom2-far | REVERB 2014 | small room 2, far mic | RVB2014_type1_rir_smallroom2_far_angla.wav | 15207 | 0.260 | 19.80 | `e67713d9387d71c89f4bc974f84926d28f822eb34339fc105ae52dd490d41506` |
| air-meeting | AIR | meeting room, 1.0 m | air_type1_air_binaural_meeting_0_1.wav | 10882 | 0.293 | 16.19 | `546457ba40eac626b3d6117e31375b0dae8d272842e18ec70d40b44ca3985bfc` |
| rwcp-e2a | RWCP | room e2a, circular array ch 0 | RWCP_type1_rir_cirline_e2a_imp090.wav | 9912 | 0.275 | 21.00 | `33769ecf9d38677276b190b08a3eb08440ac2d84b4b702df9f27d5b02721c44b` |
| rwcp-e1b | RWCP | room e1b, circular array ch 0 | RWCP_type1_rir_circle_e1b_imp090.wav | 9940 | 0.372 | 12.23 | `c41082ed63a6af3b2520bf3482078dea14bd4f1cf9bbea10d8cca763105c9aad` |
| rwcp-e1c | RWCP | room e1c, circular array ch 0 | RWCP_type1_rir_circle_e1c_imp090.wav | 11937 | 0.425 | 10.84 | `a3f5f42e03bf5014aa06e806f8a147a4a4c40c8b6acbfaea4c4c2c51e56a8217` |
| rvb-mediumroom1-near | REVERB 2014 | medium room 1, near mic | RVB2014_type1_rir_mediumroom1_near_angla.wav | 13810 | 0.337 | 17.30 | `cdf24fd9bf9507b6b5ca665e004d147206b8bd26d4f43b55deec2e3833cc58d0` |
| rvb-mediumroom2-near | REVERB 2014 | medium room 2, near mic | RVB2014_type1_rir_mediumroom2_near_angla.wav | 13839 | 0.368 | 15.46 | `b1bc66d38dad7b0387311da34d892499d782266c33c3b1dc28c216210a8e07f7` |
| air-office | AIR | office, 2.0 m | air_type1_air_binaural_office_1_2.wav | 10877 | 0.582 | 7.80 | `482e71e480c892d65b6219f9dc13f4d716744a2dcb23445f3e9fc0b9d53a05f7` |
| rvb-largeroom1-near | REVERB 2014 | large room 1, near mic | RVB2014_type1_rir_largeroom1_near_angla.wav | 13950 | 0.473 | 15.12 | `64120b11297efb57670676c19fc84b9d35d1adad2a464ac1e8052b5e98ad2e5a` |
| rvb-mediumroom2-far | REVERB 2014 | medium room 2, far mic | RVB2014_type1_rir_mediumroom2_far_angla.wav | 13801 | 0.682 | 5.69 | `fc8693eba26f5c71cbcde668a0b571961a63426ca833919c6353d05d07749663` |
| air-stairway | AIR | stairway, 1.0 m | air_type1_air_binaural_stairway_1_1_0.wav | 25600 | 0.770 | 7.75 | `1af67f07434a53d15b5d413974bbb7ec22b39717bed1cefb429478746647f815` |
| rwcp-ofc | RWCP | room ofc, circular array ch 0 | RWCP_type1_rir_cirline_ofc_imp090.wav | 24903 | 0.804 | 6.16 | `66870e020430d175536556f1e37ff4de7daf84732c29610e847e3d9c8b9bb99b` |
| air-lecture | AIR | lecture room, 2.0 m | air_type1_air_binaural_lecture_0_2.wav | 21736 | 0.878 | 4.32 | `c779eb4b83f4e56807c3297dd5ea7174e86caff93a29cbce91456fcb4015c451` |
| rvb-largeroom2-near | REVERB 2014 | large room 2, near mic | RVB2014_type1_rir_largeroom2_near_angla.wav | 15274 | 0.622 | 15.30 | `5346f3900e64d746a8c6f2bae27291c04a37c97258762aa75ea155266acf9e49` |
| rvb-largeroom2-far | REVERB 2014 | large room 2, far mic | RVB2014_type1_rir_largeroom2_far_angla.wav | 15207 | 1.047 | 3.47 | `4bb8916d416e9c142cbca0f009f5031671832ad4015f2476dc1e875d47607e7a` |

The `near` / `far` suffixes are the dataset's microphone-distance labels, not
a property this project assigned. The AIR distance is in the row text.
`air-lecture`, `air-stairway`, `rwcp-ofc` and `rvb-largeroom2-far` are the
strong-early-reflection cases (C50 below 8 dB, and the direct path is not
dominant); they are the speech-relevant hard cases for a canceller.

## Why not SLR28's synthetic half

SLR28 also contains tens of thousands of *simulated* responses. Those are not
used here: the point of this set is to replace or challenge the synthetic
family, so only responses that were physically measured are bundled.
