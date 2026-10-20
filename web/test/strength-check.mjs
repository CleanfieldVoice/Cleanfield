#!/usr/bin/env node
/**
 * Maximum-attenuation (mask floor) check.
 *
 * The demo used to expose a dry/wet blend: `mask = 1 + s * (m - 1)`.  Lowering
 * it mixed the noisy input spectrum back in, so any setting below 100 % raised
 * the noise floor.  The control is now the mask floor itself:
 *
 *     floor = 10^(maxAttenuationDb / 20)
 *     mask  = clamp(max(m, floor), 0, 1)
 *
 * This check verifies the new mapping end to end on the bundled sample:
 *
 *   * the default and −80 dB outputs are bit-identical to the previous build,
 *   * 0 dB is an exact passthrough of the input,
 *   * on a noise-only stretch −80/−40/−20 dB keep the noise attenuated,
 *     instead of returning it toward the input as the old blend did,
 *   * a speech-dominated stretch keeps its level (within 1 dB at −40 dB),
 *   * a live attenuation change is ramped over the documented 100 ms.
 *
 * It also carries a small frame-loop reference implementation.  The new-mode
 * reference output is required to be bit-identical to `processPcm()`; the
 * legacy-mode reference reproduces the old dry/wet law so the old noise numbers
 * in the report are measured, not quoted.
 *
 * Usage:
 *   node test/strength-check.mjs [input.wav] [--wasm path]
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  BIN_COUNT,
  EPSILON,
  FEATURE_POWER_SCALE,
  FFT_1024,
  FFT_SIZE,
  HOP_SIZE,
  LATENCY_SAMPLES,
  MASK_FLOOR,
  MASK_GAIN,
  SAMPLE_RATE,
  SYNTHESIS_SCALE,
  WINDOW,
  WINDOW_SIZE,
  applyMaskFloor,
  loadVoiceEngine,
  maskFloorForDb,
  processPcm,
} from '../dsp.js';
import { ATTENUATION_RAMP_SAMPLES, LiveDenoiser } from '../live-dsp.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = path.resolve(HERE, '../../../local-denoiser/samples/noisy.wav');
const DEFAULT_WASM = path.resolve(HERE, '../cf_wasm.wasm');
const DRY_TOLERANCE = 1e-7;
const NOISE_WINDOW = Math.round(0.5 * SAMPLE_RATE);
const NOISE_STEP = Math.round(0.25 * SAMPLE_RATE);
const SPEECH_TOLERANCE_DB = 1.0;

/**
 * SHA-256 of the raw float32 output of the pre-change build (dry/wet strength)
 * for the bundled sample at its default strength, plus the SHA-256 of the
 * bundled input WAV.  They are only compared when the bundled sample is used;
 * pass another input and the historical-hash check is reported as skipped.
 */
const BUNDLED_INPUT_SHA256 = '7489638a9d16ea9471b5adb7bdd4d8f26b802b2280477d7abd64053a418e0010';
const PREVIOUS_BUILD_OUTPUT_SHA256 =
  '4dbffcb680b0fd02e92474b4b6b65bccada7659dbfeb61a8adbc244e774a6c21';

function usage() {
  console.log('usage: node test/strength-check.mjs [input.wav] [--wasm path]');
}

function parseArgs(argv) {
  const options = { input: null, wasm: DEFAULT_WASM };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else if (arg === '--wasm') {
      options.wasm = path.resolve(argv[++i]);
    } else if (arg.startsWith('--')) {
      throw new Error(`unknown option: ${arg}`);
    } else if (!options.input) {
      options.input = path.resolve(arg);
    } else {
      throw new Error(`unexpected argument: ${arg}`);
    }
  }
  if (!options.input) options.input = DEFAULT_INPUT;
  return options;
}

/** Minimal RIFF/WAVE reader (PCM 16/24/32-bit, IEEE float 32/64-bit). */
function readWav(file) {
  const buffer = readFileSync(file);
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`${file}: not a RIFF/WAVE file`);
  }
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let pcm = null;
  let at = 12;
  while (at + 8 <= buffer.length) {
    const id = buffer.toString('ascii', at, at + 4);
    const size = buffer.readUInt32LE(at + 4);
    const body = at + 8;
    if (body + size > buffer.length) throw new Error(`${file}: truncated ${id} chunk`);
    if (id === 'fmt ') {
      format = buffer.readUInt16LE(body);
      channels = buffer.readUInt16LE(body + 2);
      sampleRate = buffer.readUInt32LE(body + 4);
      bits = buffer.readUInt16LE(body + 14);
    } else if (id === 'data') {
      pcm = buffer.subarray(body, body + size);
    }
    at = body + size + (size & 1);
  }
  if (!pcm || !channels || !sampleRate) throw new Error(`${file}: missing fmt/data chunk`);
  if (![1, 3].includes(format)) throw new Error(`${file}: unsupported WAV format ${format}`);

  const bytesPerSample = bits / 8;
  const frameBytes = bytesPerSample * channels;
  const frames = Math.floor(pcm.length / frameBytes);
  const samples = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const offset = i * frameBytes + c * bytesPerSample;
      let value;
      if (format === 3) {
        value = bits === 32 ? pcm.readFloatLE(offset) : pcm.readDoubleLE(offset);
      } else if (bits === 16) {
        value = pcm.readInt16LE(offset) / 32768;
      } else if (bits === 24) {
        const raw = pcm[offset] | (pcm[offset + 1] << 8) | (pcm[offset + 2] << 16);
        value = ((raw << 8) >> 8) / 8388608;
      } else {
        value = pcm.readInt32LE(offset) / 2147483648;
      }
      sum += value;
    }
    samples[i] = sum / channels;
  }
  return { samples, sampleRate, channels, bits, frames };
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return samples.length ? Math.sqrt(sum / samples.length) : 0;
}

function peak(samples) {
  let p = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > p) p = v;
  }
  return p;
}

function db(value) {
  return value > 0 ? 20 * Math.log10(value) : -Infinity;
}

function segmentDb(samples, start, length) {
  return db(rms(samples.subarray(start, start + length)));
}

function sha256Floats(samples) {
  return createHash('sha256')
    .update(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength))
    .digest('hex');
}

function compare(a, b) {
  let maxAbsDiff = 0;
  let sum = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const diff = a[i] - b[i];
    const abs = Math.abs(diff);
    if (abs > maxAbsDiff) maxAbsDiff = abs;
    sum += diff * diff;
    sa += a[i];
    sb += b[i];
    saa += a[i] * a[i];
    sbb += b[i] * b[i];
    sab += a[i] * b[i];
  }
  const numerator = n * sab - sa * sb;
  const denominator = Math.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
  return {
    n,
    maxAbsDiff,
    rmsDiff: n ? Math.sqrt(sum / n) : 0,
    correlation: denominator > 0 ? numerator / denominator : 1,
  };
}

function finite(samples) {
  for (let i = 0; i < samples.length; i++) {
    if (!Number.isFinite(samples[i])) return false;
  }
  return true;
}

/**
 * Frame-loop reference implementation used only by this test.  It mirrors
 * `processPcm()` exactly (same FFT, window, feature rounding, synthesis and
 * latency slicing) but takes the mask law as a callback, so the test can run
 * both the new floor law and the previous dry/wet law over the same model
 * predictions.
 *
 * @param {Float32Array} source
 * @param {import('../dsp.js').VoiceEngine} engine
 * @param {(magnitude: number) => number} maskFor
 * @returns {Float32Array}
 */
function runReference(source, engine, maskFor) {
  const length = source.length;
  const totalLength = length + LATENCY_SAMPLES;
  const frames = Math.floor(totalLength / HOP_SIZE);
  const output = new Float32Array(totalLength);

  const analysis = new Float32Array(WINDOW_SIZE);
  const overlap = new Float32Array(WINDOW_SIZE);
  const fftRe = new Float64Array(FFT_SIZE);
  const fftIm = new Float64Array(FFT_SIZE);
  const specRe = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
  const specIm = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
  const curRe = new Float32Array(BIN_COUNT);
  const curIm = new Float32Array(BIN_COUNT);
  const features = new Float32Array(BIN_COUNT);
  const mask = new Float32Array(BIN_COUNT);

  engine.reset();

  for (let frame = 0; frame < frames; frame++) {
    const inputBase = frame * HOP_SIZE;
    analysis.copyWithin(0, HOP_SIZE);
    for (let j = 0; j < HOP_SIZE; j++) {
      const index = inputBase + j;
      analysis[WINDOW_SIZE - HOP_SIZE + j] = index < length ? source[index] : 0;
    }

    fftRe.fill(0);
    fftIm.fill(0);
    for (let i = 0; i < WINDOW_SIZE; i++) {
      fftRe[i] = Math.fround(analysis[i] * WINDOW[i]);
    }
    FFT_1024.forward(fftRe, fftIm);
    for (let k = 0; k < BIN_COUNT; k++) {
      curRe[k] = fftRe[k];
      curIm[k] = fftIm[k];
    }

    for (let k = 0; k < BIN_COUNT; k++) {
      const re = curRe[k];
      const im = curIm[k];
      const power = Math.fround(Math.fround(re * re) + Math.fround(im * im));
      const scaled = Math.fround(power * FEATURE_POWER_SCALE);
      features[k] = Math.fround(Math.log(Math.fround(scaled + EPSILON)));
    }

    const prediction = engine.process(features);
    for (let k = 0; k < BIN_COUNT; k++) {
      const re = prediction[k];
      const im = prediction[k + BIN_COUNT];
      const power = Math.fround(
        Math.fround(Math.fround(re * re) + Math.fround(im * im)) + EPSILON
      );
      const magnitude = Math.fround(Math.sqrt(power));
      const m = Math.fround(magnitude * MASK_GAIN);
      mask[k] = maskFor(m);
    }

    const slot = frame & 1;
    const targetRe = specRe[slot];
    const targetIm = specIm[slot];

    fftRe.fill(0);
    fftIm.fill(0);
    for (let k = 0; k < BIN_COUNT; k++) {
      const m = mask[k];
      const re = Math.fround(targetRe[k] * m);
      const im = Math.fround(targetIm[k] * m);
      fftRe[k] = re;
      fftIm[k] = im;
      if (k > 0 && k < BIN_COUNT - 1) {
        fftRe[FFT_SIZE - k] = re;
        fftIm[FFT_SIZE - k] = -im;
      }
    }
    FFT_1024.inverse(fftRe, fftIm);

    for (let i = 0; i < WINDOW_SIZE; i++) {
      const reconstructed = Math.fround(fftRe[i]);
      const windowed = Math.fround(Math.fround(reconstructed * WINDOW[i]) * SYNTHESIS_SCALE);
      overlap[i] = Math.fround(overlap[i] + windowed);
    }

    targetRe.set(curRe);
    targetIm.set(curIm);

    const outputBase = HOP_SIZE + frame * HOP_SIZE;
    const available = Math.min(HOP_SIZE, totalLength - outputBase);
    for (let i = 0; i < available; i++) output[outputBase + i] = overlap[i];

    overlap.copyWithin(0, HOP_SIZE);
    overlap.fill(0, WINDOW_SIZE - HOP_SIZE);
  }

  return output.slice(LATENCY_SAMPLES, LATENCY_SAMPLES + length);
}

/** New mask floor law for a maximum attenuation in dB. */
function newMaskLaw(maxAttenuationDb) {
  const floor = maskFloorForDb(maxAttenuationDb);
  return (magnitude) => applyMaskFloor(magnitude, floor);
}

/** Previous dry/wet law for a strength in [0, 1].  Kept only for comparison. */
function legacyDryWetLaw(strength) {
  if (strength >= 1) return (magnitude) => applyMaskFloor(magnitude, MASK_FLOOR);
  if (strength <= 0) return () => 1;
  const s = Math.fround(strength);
  return (magnitude) => {
    const floored = applyMaskFloor(magnitude, MASK_FLOOR);
    return Math.fround(1 + Math.fround(s * Math.fround(floored - 1)));
  };
}

/**
 * Scan non-silent 0.5 s windows and return their input/output levels.  Windows
 * where the input is below −60 dBFS are skipped so digital silence cannot win
 * the "most attenuated" comparison.
 */
function scanWindows(input, output) {
  const rows = [];
  for (let start = 0; start + NOISE_WINDOW <= input.length; start += NOISE_STEP) {
    const inputDb = segmentDb(input, start, NOISE_WINDOW);
    if (!(inputDb > -60)) continue;
    const outputDb = segmentDb(output, start, NOISE_WINDOW);
    rows.push({ start, inputDb, outputDb, reduction: inputDb - outputDb });
  }
  return rows;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const input = readWav(options.input);
  if (input.sampleRate !== SAMPLE_RATE) {
    throw new Error(`input must be ${SAMPLE_RATE} Hz for this check (got ${input.sampleRate})`);
  }
  const wasmBytes = readFileSync(options.wasm);
  const module = new WebAssembly.Module(wasmBytes);
  const engine = await loadVoiceEngine(module);

  const inputSha = createHash('sha256').update(readFileSync(options.input)).digest('hex');
  const bundledInput = inputSha === BUNDLED_INPUT_SHA256;

  console.log(`input: ${options.input}`);
  console.log(`  ${input.frames} samples, ${input.sampleRate} Hz, ${input.channels} channel(s)`);
  console.log(`engine: backend=${engine.backendName || '?'}, model=${engine.modelImageBytes} bytes`);
  console.log('');

  // Endpoints and the four settings reported below.
  const baseline = processPcm(input.samples, engine);
  const atMinus80 = processPcm(input.samples, engine, { maxAttenuationDb: -80 });
  const at0 = processPcm(input.samples, engine, { maxAttenuationDb: 0 });
  const atMinus40 = processPcm(input.samples, engine, { maxAttenuationDb: -40 });
  const atMinus20 = processPcm(input.samples, engine, { maxAttenuationDb: -20 });
  const byDb = new Map([
    [-80, atMinus80],
    [-40, atMinus40],
    [-20, atMinus20],
    [0, at0],
  ]);

  const defaultVsMinus80 = compare(atMinus80.samples, baseline.samples);
  const zeroVsInput = compare(at0.samples, input.samples);
  const outputSha = sha256Floats(atMinus80.samples);

  console.log('mapping');
  console.log(
    `  floorDb = max attenuation in dB; floor = 10^(dB/20); mask = clamp(max(m, floor), 0, 1)`
  );
  console.log(
    `  floor(-80)=${maskFloorForDb(-80).toExponential(3)} (MASK_FLOOR=` +
      `${MASK_FLOOR.toExponential(3)}), floor(-40)=${maskFloorForDb(-40).toExponential(3)}, ` +
      `floor(-20)=${maskFloorForDb(-20).toExponential(3)}, floor(0)=${maskFloorForDb(0)}`
  );
  console.log(
    `  −80 dB vs default: maxAbsDiff=${defaultVsMinus80.maxAbsDiff.toExponential(3)}, ` +
      `rmsDiff=${defaultVsMinus80.rmsDiff.toExponential(3)}, ` +
      `correlation=${defaultVsMinus80.correlation.toFixed(9)}`
  );
  console.log(
    `  previous build at −80 dB: sha256=${outputSha}` +
      (bundledInput
        ? ` (${outputSha === PREVIOUS_BUILD_OUTPUT_SHA256 ? 'match' : 'MISMATCH'})`
        : ' (historical hash skipped: not the bundled sample)')
  );
  console.log(
    `  0 dB vs input: maxAbsDiff=${zeroVsInput.maxAbsDiff.toExponential(3)}, ` +
      `rmsDiff=${zeroVsInput.rmsDiff.toExponential(3)}, ` +
      `correlation=${zeroVsInput.correlation.toFixed(9)}, peak=${peak(at0.samples).toFixed(6)}`
  );
  console.log('');

  // Locate one strongly attenuated noise-only stretch and one loud,
  // speech-dominated stretch in the default (−80 dB) output.
  const windows = scanWindows(input.samples, atMinus80.samples);
  if (windows.length === 0) throw new Error('input has no non-silent 0.5 s windows to measure');
  const noiseWindow = windows.reduce((best, row) =>
    row.reduction > best.reduction ? row : best
  );
  const speechWindow = windows.reduce((best, row) => (row.inputDb > best.inputDb ? row : best));

  // Legacy dry/wet outputs for the same nominal settings (s = -dB / 80).
  const legacyStrengths = [1, 0.5, 0.25, 0];
  const legacy = new Map();
  for (const s of legacyStrengths) {
    legacy.set(s, runReference(input.samples, engine, legacyDryWetLaw(s)));
  }
  // The reference harness must reproduce production exactly, or the legacy
  // numbers it produced would not be a valid comparison.
  const referenceNew = runReference(input.samples, engine, newMaskLaw(-40));
  const referenceVsProduction = compare(referenceNew, atMinus40.samples);
  const legacyFullVsDefault = compare(legacy.get(1), baseline.samples);
  const legacyDryVsInput = compare(legacy.get(0), input.samples);

  const settings = [
    { db: -80, legacyStrength: 1 },
    { db: -40, legacyStrength: 0.5 },
    { db: -20, legacyStrength: 0.25 },
    { db: 0, legacyStrength: 0 },
  ];

  console.log(
    `noise-only stretch ${(noiseWindow.start / SAMPLE_RATE).toFixed(2)}–` +
      `${((noiseWindow.start + NOISE_WINDOW) / SAMPLE_RATE).toFixed(2)} s ` +
      `(input ${noiseWindow.inputDb.toFixed(2)} dBFS)`
  );
  console.log(
    '  setting      output dBFS   reduction   old dry/wet eq.   old dBFS   old reduction'
  );
  const noiseRows = [];
  for (const setting of settings) {
    const outputDb = segmentDb(byDb.get(setting.db).samples, noiseWindow.start, NOISE_WINDOW);
    const reduction = noiseWindow.inputDb - outputDb;
    const oldSamples = legacy.get(setting.legacyStrength);
    const oldDb = segmentDb(oldSamples, noiseWindow.start, NOISE_WINDOW);
    const oldReduction = noiseWindow.inputDb - oldDb;
    noiseRows.push({
      settingDb: setting.db,
      outputDb,
      reduction,
      legacyStrength: setting.legacyStrength,
      legacyDb: oldDb,
      legacyReduction: oldReduction,
    });
    console.log(
      `  ${`${setting.db} dB`.padEnd(11)} ${outputDb.toFixed(2).padStart(10)} ` +
        `${reduction.toFixed(2).padStart(10)}   ` +
        `${`s=${setting.legacyStrength.toFixed(2)}`.padEnd(17)} ` +
        `${oldDb.toFixed(2).padStart(8)} ` +
        `${oldReduction.toFixed(2).padStart(13)}`
    );
  }
  console.log('');

  console.log(
    `speech-dominated stretch ${(speechWindow.start / SAMPLE_RATE).toFixed(2)}–` +
      `${((speechWindow.start + NOISE_WINDOW) / SAMPLE_RATE).toFixed(2)} s ` +
      `(input ${speechWindow.inputDb.toFixed(2)} dBFS)`
  );
  console.log('  setting      output dBFS   change');
  const speechRows = [];
  for (const setting of settings) {
    const outputDb = segmentDb(byDb.get(setting.db).samples, speechWindow.start, NOISE_WINDOW);
    const change = outputDb - speechWindow.inputDb;
    speechRows.push({ settingDb: setting.db, outputDb, change });
    console.log(
      `  ${`${setting.db} dB`.padEnd(11)} ${outputDb.toFixed(2).padStart(10)} ` +
        `${change.toFixed(2).padStart(8)}`
    );
  }
  console.log('');

  // A live change must ramp the mask floor in dB over the documented 100 ms.
  const live = new LiveDenoiser(engine, {
    inputRate: SAMPLE_RATE,
    outputRate: SAMPLE_RATE,
  });
  const block = new Float32Array(HOP_SIZE);
  const out = new Float32Array(HOP_SIZE);
  const feed = (index) => {
    block.fill(0);
    const offset = index * HOP_SIZE;
    block.set(input.samples.subarray(offset, Math.min(input.samples.length, offset + HOP_SIZE)));
    live.processBlock(block, out);
    return out;
  };
  for (let i = 0; i < 24; i++) feed(i);
  const rampBefore = live.streaming.maxAttenuationDb;
  live.setMaxAttenuationDb(0);
  const ramp = [];
  let liveFinite = true;
  for (let i = 24; i < 40; i++) {
    feed(i);
    ramp.push(live.streaming.maxAttenuationDb);
    liveFinite = liveFinite && finite(out);
  }
  const rampLast = ramp[ramp.length - 1];
  const rampMonotonic = ramp.every((v, i) => i === 0 || v >= ramp[i - 1] - 1e-12);
  const expectedStep = (80 * HOP_SIZE) / ATTENUATION_RAMP_SAMPLES;
  console.log('live attenuation ramp');
  console.log(
    `  before: ${rampBefore.toFixed(3)} dB, after 1 hop: ${ramp[0].toFixed(3)} dB, ` +
      `after ${ramp.length} hops: ${rampLast.toFixed(6)} dB`
  );
  console.log(
    `  ramp: ${ATTENUATION_RAMP_SAMPLES} samples (${((ATTENUATION_RAMP_SAMPLES / SAMPLE_RATE) * 1000).toFixed(0)} ms), ` +
      `step=${expectedStep.toFixed(2)} dB per hop, monotonic=${rampMonotonic}, ` +
      `finite=${liveFinite}, frames=${live.frames}`
  );
  console.log('');

  const noiseByDb = new Map(noiseRows.map((row) => [row.settingDb, row]));
  const speechByDb = new Map(speechRows.map((row) => [row.settingDb, row]));
  const checks = {
    minus80MatchesDefault:
      defaultVsMinus80.maxAbsDiff === 0 &&
      defaultVsMinus80.rmsDiff === 0 &&
      defaultVsMinus80.correlation === 1,
    previousBuildBitIdentical: bundledInput ? outputSha === PREVIOUS_BUILD_OUTPUT_SHA256 : null,
    zeroMatchesInput:
      zeroVsInput.maxAbsDiff < DRY_TOLERANCE &&
      zeroVsInput.correlation > 0.999999 &&
      at0.samples.length === input.samples.length,
    floorMapping:
      maskFloorForDb(0) === 1 &&
      maskFloorForDb(-80) === MASK_FLOOR &&
      maskFloorForDb(-40) === Math.fround(10 ** (-40 / 20)) &&
      maskFloorForDb(-20) === Math.fround(10 ** (-20 / 20)),
    noiseStaysAttenuated:
      noiseByDb.get(-40).reduction >= 35 &&
      noiseByDb.get(-20).reduction >= 15 &&
      noiseByDb.get(-20).reduction < noiseByDb.get(-40).reduction &&
      noiseByDb.get(-40).reduction < noiseByDb.get(-80).reduction,
    oldBlendReintroducedNoise:
      noiseByDb.get(-40).legacyReduction < 10 && noiseByDb.get(-20).legacyReduction < 5,
    speechPreservedAtMinus40: Math.abs(speechByDb.get(-40).change) <= SPEECH_TOLERANCE_DB,
    referenceMatchesProduction:
      referenceVsProduction.maxAbsDiff === 0 &&
      legacyFullVsDefault.maxAbsDiff === 0 &&
      legacyDryVsInput.maxAbsDiff < DRY_TOLERANCE,
    liveRamp:
      rampMonotonic &&
      liveFinite &&
      Math.abs(ramp[0] - (-80 + expectedStep)) < 1e-9 &&
      rampLast === 0 &&
      live.frames === 40,
  };
  const checkEntries = Object.entries(checks);
  const knownChecks = checkEntries.filter(([, value]) => value !== null);
  console.log(
    `checks: ${checkEntries
      .map(([name, ok]) => `${name}=${ok === null ? 'skipped' : ok ? 'ok' : 'FAIL'}`)
      .join(', ')}`
  );
  const verdict = knownChecks.every(([, ok]) => ok) ? 'PASS' : 'FAIL';
  console.log(
    'RESULT ' +
      JSON.stringify({
        inputSamples: input.samples.length,
        bundledInput,
        mapping: {
          minus80: maskFloorForDb(-80),
          minus40: maskFloorForDb(-40),
          minus20: maskFloorForDb(-20),
          zero: maskFloorForDb(0),
        },
        endpoints: {
          minus80VsDefault: defaultVsMinus80,
          previousBuildSha256: outputSha,
          previousBuildMatches: bundledInput ? outputSha === PREVIOUS_BUILD_OUTPUT_SHA256 : null,
          zeroVsInput: zeroVsInput,
        },
        noiseWindow: {
          start: noiseWindow.start,
          seconds: NOISE_WINDOW / SAMPLE_RATE,
          inputDb: noiseWindow.inputDb,
          rows: noiseRows,
        },
        speechWindow: {
          start: speechWindow.start,
          seconds: NOISE_WINDOW / SAMPLE_RATE,
          inputDb: speechWindow.inputDb,
          rows: speechRows,
        },
        ramp: {
          samples: ATTENUATION_RAMP_SAMPLES,
          stepDb: expectedStep,
          first: ramp[0],
          last: rampLast,
          monotonic: rampMonotonic,
        },
        checks,
        verdict,
      })
  );

  engine.close();
  if (verdict !== 'PASS') process.exitCode = 1;
}

main().catch((error) => {
  console.error(`error: ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
