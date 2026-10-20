#!/usr/bin/env node
/**
 * Spectrogram check for the static web demo.
 *
 * Runs the exact offline DSP (`processPcm` from dsp.js) over a WAV with
 * spectrogram capture enabled and verifies that the two panels show the real
 * signal path:
 *
 *   1. the input panel is the windowed analysis spectrum that enters the model;
 *   2. the output panel is a windowed transform of the 768 most recently
 *      emitted samples, after the synthesis window and overlap-add;
 *   3. both panels are aligned to the same audio span for every column;
 *   4. the noise-only stretch shows a large input-vs-output reduction;
 *   5. the output panel is not a copy of the input panel;
 *   6. the captured column data has no periodic vertical-line signature, and
 *      the painter's horizontal scaling does not alternate its averaging
 *      weight from column to column.
 *
 * The quiet stretch is selected automatically as the lowest mean input
 * magnitude over a 1 s window, so the check is not tied to sample timestamps.
 *
 * Usage:
 *   node test/spectrogram-check.mjs [input.wav] [--wasm path]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  FFT_1024,
  FFT_SIZE,
  HOP_SIZE,
  LATENCY_SAMPLES,
  LOOKAHEAD_HOPS,
  SAMPLE_RATE,
  SPECTRUM_ALIGN_HOPS,
  SPECTRUM_NORM,
  WINDOW,
  WINDOW_SIZE,
  loadVoiceEngine,
  processPcm,
} from '../dsp.js';
import {
  FILE_HISTORY,
  SPEC_BINS,
  SPEC_MIN_DB,
  SpectrogramBuffer,
} from '../spectrogram-core.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = path.resolve(HERE, '../../../local-denoiser/samples/noisy.wav');
const DEFAULT_WASM = path.resolve(HERE, '../cf_wasm.wasm');
const WINDOW_SECONDS = 1;
const MIN_NOISE_DB = -90; // ignore digital silence when choosing the stretch
const NOISE_TARGET_DB = -15;
const CAPTURE_TOLERANCE_DB = 1e-3;

function usage() {
  console.log('usage: node test/spectrogram-check.mjs [input.wav] [--wasm path]');
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

function dbToMagnitude(db) {
  return 10 ** (db / 20);
}

function mean(values) {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : String(value);
}

/** Pearson correlation of two equal-length series. */
function correlation(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (let i = 0; i < n; i++) {
    sa += a[i];
    sb += b[i];
    saa += a[i] * a[i];
    sbb += b[i] * b[i];
    sab += a[i] * b[i];
  }
  const numerator = n * sab - sa * sb;
  const denominator = Math.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
  return denominator > 0 ? numerator / denominator : 0;
}

/** Normalised autocorrelation at lags 1..maxLag. */
function autocorrelation(series, maxLag) {
  const n = series.length;
  const average = mean(series);
  const centred = new Float64Array(n);
  let denom = 0;
  for (let i = 0; i < n; i++) {
    centred[i] = series[i] - average;
    denom += centred[i] * centred[i];
  }
  const out = new Float64Array(maxLag);
  for (let lag = 1; lag <= maxLag; lag++) {
    let num = 0;
    for (let i = 0; i + lag < n; i++) num += centred[i] * centred[i + lag];
    out[lag - 1] = denom > 0 ? num / denom : 0;
  }
  return out;
}

/**
 * Windowed FFT of `samples[base .. base + WINDOW_SIZE)` with zero padding.
 * `roundSpectrum` reproduces the Float32 copy that the model input path takes.
 */
function fillSpectrumDb(out, re, im, samples, base, roundSpectrum) {
  re.fill(0);
  im.fill(0);
  for (let i = 0; i < WINDOW_SIZE; i++) {
    const index = base + i;
    const sample = index >= 0 && index < samples.length ? samples[index] : 0;
    re[i] = Math.fround(sample * WINDOW[i]);
  }
  FFT_1024.forward(re, im);
  for (let k = 0; k < SPEC_BINS; k++) {
    const r = roundSpectrum ? Math.fround(re[k]) : re[k];
    const imag = roundSpectrum ? Math.fround(im[k]) : im[k];
    const magnitude = Math.sqrt(r * r + imag * imag) * SPECTRUM_NORM;
    const db = magnitude > 0 ? 20 * Math.log10(magnitude) : SPEC_MIN_DB;
    out[k] = db < SPEC_MIN_DB ? SPEC_MIN_DB : db > 0 ? 0 : db;
  }
}

function maxAbs(values) {
  let max = 0;
  for (const value of values) {
    const abs = Math.abs(value);
    if (abs > max) max = abs;
  }
  return max;
}

function gcd(a, b) {
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/**
 * Horizontal scaling of the painter.  The former implementation averaged the
 * source columns that fell inside `[floor(c*g), floor((c+1)*g))`, so adjacent
 * destination columns averaged either N or N+1 source columns.  The current
 * implementation uses fractional edge weights with a constant total weight.
 *
 * Both `old` and `new` below are the noise gain per destination column
 * (sum of squared normalised weights) for a destination of `target` columns.
 * A jump in that gain is what produced the evenly spaced vertical bands.
 */
function painterScaling(columns, target) {
  const group = columns / target;
  const oldGain = new Float64Array(target);
  const newGain = new Float64Array(target);
  let oldCountMin = Infinity;
  let oldCountMax = 0;
  for (let c = 0; c < target; c++) {
    const first = Math.floor(c * group);
    const last = Math.max(first + 1, Math.min(columns, Math.floor((c + 1) * group)));
    const count = last - first;
    if (count < oldCountMin) oldCountMin = count;
    if (count > oldCountMax) oldCountMax = count;
    oldGain[c] = 1 / count;

    const start = c * group;
    const end = start + group;
    const begin = Math.floor(start);
    const finish = Math.ceil(end);
    let sum = 0;
    for (let s = begin; s < finish; s++) {
      const weight = Math.min(end, s + 1) - Math.max(start, s);
      const normalised = weight / group;
      sum += normalised * normalised;
    }
    newGain[c] = sum;
  }
  const stats = (values) => {
    let min = Infinity;
    let max = -Infinity;
    let total = 0;
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
      total += value;
    }
    const avg = total / values.length;
    let maxStep = 0;
    for (let i = 1; i < values.length; i++) {
      const step = Math.abs(values[i] - values[i - 1]);
      if (step > maxStep) maxStep = step;
    }
    return { min, max, mean: avg, swing: (max - min) / avg, maxStep: maxStep / avg };
  };
  return {
    target,
    group,
    period: target / gcd(target, columns),
    oldCountMin,
    oldCountMax,
    old: stats(oldGain),
    new: stats(newGain),
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const input = readWav(options.input);
  console.log(`input: ${options.input}`);
  console.log(
    `  ${input.frames} samples, ${input.channels} channel(s), ${input.sampleRate} Hz, ` +
      `${input.bits}-bit, ${(input.frames / input.sampleRate).toFixed(3)} s`
  );
  if (input.sampleRate !== SAMPLE_RATE) {
    throw new Error(`input must be ${SAMPLE_RATE} Hz for this check (got ${input.sampleRate})`);
  }

  const wasmBytes = readFileSync(options.wasm);
  const engine = await loadVoiceEngine(new WebAssembly.Module(wasmBytes));
  console.log(`engine: backend=${engine.backendName || '?'}, model=${engine.modelImageBytes} bytes`);

  // Baseline with capture disabled.  The capture run below must produce exactly
  // the same audio; capture only reads after the output hop is written.
  const baseline = processPcm(input.samples, engine);

  const buffer = new SpectrogramBuffer(FILE_HISTORY);
  const columnStats = [];
  const capturedInput = [];
  const capturedOutput = [];
  let nonFinite = 0;
  let maxDb = SPEC_MIN_DB;
  let minDb = 0;

  const started = performance.now();
  const result = processPcm(input.samples, engine, {
    maxSpectrumColumns: FILE_HISTORY,
    onSpectrum: (index, inputDb, outputDb) => {
      if (inputDb.length !== SPEC_BINS || outputDb.length !== SPEC_BINS) {
        throw new Error('spectrum callback received the wrong bin count');
      }
      capturedInput.push(Float32Array.from(inputDb));
      capturedOutput.push(Float32Array.from(outputDb));
      let inputDbSum = 0;
      let outputDbSum = 0;
      let inputMagSum = 0;
      let outputMagSum = 0;
      let inputEnergy = 0;
      let outputEnergy = 0;
      for (let k = 0; k < SPEC_BINS; k++) {
        const inValue = inputDb[k];
        const outValue = outputDb[k];
        if (!Number.isFinite(inValue) || !Number.isFinite(outValue)) {
          nonFinite += 1;
          continue;
        }
        if (inValue > maxDb) maxDb = inValue;
        if (inValue < minDb) minDb = inValue;
        if (outValue > maxDb) maxDb = outValue;
        if (outValue < minDb) minDb = outValue;
        inputDbSum += inValue;
        outputDbSum += outValue;
        const inputMagnitude = dbToMagnitude(inValue);
        const outputMagnitude = dbToMagnitude(outValue);
        inputMagSum += inputMagnitude;
        outputMagSum += outputMagnitude;
        inputEnergy += inputMagnitude * inputMagnitude;
        outputEnergy += outputMagnitude * outputMagnitude;
      }
      buffer.pushDb(inputDb, outputDb);
      columnStats.push({
        index,
        inputDb: inputDbSum / SPEC_BINS,
        outputDb: outputDbSum / SPEC_BINS,
        inputMag: inputMagSum / SPEC_BINS,
        outputMag: outputMagSum / SPEC_BINS,
        inputEnergy,
        outputEnergy,
      });
    },
  });
  const elapsed = performance.now() - started;

  let audioMaxAbsDiff = 0;
  const audioCompareLength = Math.min(result.samples.length, baseline.samples.length);
  for (let i = 0; i < audioCompareLength; i++) {
    const diff = Math.abs(result.samples[i] - baseline.samples[i]);
    if (diff > audioMaxAbsDiff) audioMaxAbsDiff = diff;
  }
  const audioIdentityOk =
    result.frames === baseline.frames &&
    result.samples.length === baseline.samples.length &&
    audioMaxAbsDiff === 0;

  const columns = buffer.length;
  const expectedColumns = Math.max(0, result.frames - SPECTRUM_ALIGN_HOPS);
  const planesOk =
    buffer.input.length === buffer.capacity * SPEC_BINS &&
    buffer.output.length === buffer.capacity * SPEC_BINS;
  let nonZeroInput = 0;
  let nonZeroOutput = 0;
  for (let c = 0; c < columns; c++) {
    const offset = buffer.columnOffset(c);
    for (let k = 0; k < SPEC_BINS; k++) {
      if (buffer.input[offset + k] !== 0) nonZeroInput += 1;
      if (buffer.output[offset + k] !== 0) nonZeroOutput += 1;
    }
  }
  const produced = result.frames > 0 && columns > 0 && columnStats.length === columns;
  const finite = nonFinite === 0;
  const populated = nonZeroInput > 0 && nonZeroOutput > 0 && planesOk;
  const aligned = columns === expectedColumns && columns === capturedOutput.length;

  // Recompute both panels from the audio they are supposed to show.  Column c
  // pairs input frame c with the 768 output samples heard at the same instant:
  // both windows start at (c - 2) * HOP_SIZE.  Input columns 0 and 1 contain
  // the zero-padded start-up; output columns 0 and 1 include audio that the
  // exported file trims by its fixed 1,280-sample latency, so the output
  // comparison starts at column 2.
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  const expected = new Float32Array(SPEC_BINS);
  let inputDiffMax = 0;
  let inputDiffSum = 0;
  let inputDiffCount = 0;
  let inputDiffColumns = 0;
  let outputDiffMax = 0;
  let outputDiffSum = 0;
  let outputDiffCount = 0;
  let outputDiffColumns = 0;
  let outputSkipped = 0;
  for (let c = 0; c < columns; c++) {
    const base = (c - 2) * HOP_SIZE;
    let columnMax = 0;
    fillSpectrumDb(expected, re, im, input.samples, base, true);
    for (let k = 0; k < SPEC_BINS; k++) {
      const diff = Math.abs(expected[k] - capturedInput[c][k]);
      inputDiffSum += diff;
      inputDiffCount += 1;
      if (diff > columnMax) columnMax = diff;
    }
    if (columnMax > inputDiffMax) inputDiffMax = columnMax;
    inputDiffColumns += 1;

    if (base < 0 || base + WINDOW_SIZE > result.samples.length) {
      outputSkipped += 1;
      continue;
    }
    columnMax = 0;
    fillSpectrumDb(expected, re, im, result.samples, base, false);
    for (let k = 0; k < SPEC_BINS; k++) {
      const diff = Math.abs(expected[k] - capturedOutput[c][k]);
      outputDiffSum += diff;
      outputDiffCount += 1;
      if (diff > columnMax) columnMax = diff;
    }
    if (columnMax > outputDiffMax) outputDiffMax = columnMax;
    outputDiffColumns += 1;
  }
  const inputDiffMean = inputDiffCount > 0 ? inputDiffSum / inputDiffCount : 0;
  const outputDiffMean = outputDiffCount > 0 ? outputDiffSum / outputDiffCount : 0;
  const captureInputOk = inputDiffMax <= CAPTURE_TOLERANCE_DB;
  const captureOutputOk =
    outputDiffColumns > 0 && outputDiffMax <= CAPTURE_TOLERANCE_DB;

  // Whole-file energy correlation and level difference.  A copy would show a
  // correlation of exactly 1 and a mean difference of 0 dB.
  const inputEnergy = columnStats.map((entry) => entry.inputEnergy);
  const outputEnergy = columnStats.map((entry) => entry.outputEnergy);
  const wholeCorrelation = correlation(inputEnergy, outputEnergy);
  const meanColumnDbDelta = mean(columnStats.map((entry) => entry.outputDb - entry.inputDb));
  const wholeInputMag = mean(columnStats.map((entry) => entry.inputMag));
  const wholeOutputMag = mean(columnStats.map((entry) => entry.outputMag));
  const wholeMagnitudeDelta = 20 * Math.log10(wholeOutputMag / wholeInputMag);
  const totalEnergyDelta =
    10 * Math.log10(
      outputEnergy.reduce((sum, value) => sum + value, 0) /
        inputEnergy.reduce((sum, value) => sum + value, 0)
    );

  // Quietest 1 s window = noise-only stretch for this sample.  Digital
  // silence is ignored so a padded tail cannot win the selection.  Columns 0
  // and 1 contain the alignment start-up, so the search starts at column 2.
  const windowColumns = Math.max(1, Math.round((WINDOW_SECONDS * SAMPLE_RATE) / HOP_SIZE));
  let best = null;
  for (let start = Math.min(LOOKAHEAD_HOPS, Math.max(0, columns - windowColumns));
    start + windowColumns <= columns;
    start++) {
    let inputDbSum = 0;
    let inputMagSum = 0;
    let outputDbSum = 0;
    let outputMagSum = 0;
    for (let c = start; c < start + windowColumns; c++) {
      const entry = columnStats[c];
      inputDbSum += entry.inputDb;
      outputDbSum += entry.outputDb;
      inputMagSum += entry.inputMag;
      outputMagSum += entry.outputMag;
    }
    const meanInputDb = inputDbSum / windowColumns;
    if (meanInputDb < MIN_NOISE_DB) continue;
    const meanInputMag = inputMagSum / windowColumns;
    if (!best || meanInputMag < best.inputMag) {
      best = {
        start,
        inputDb: meanInputDb,
        outputDb: outputDbSum / windowColumns,
        inputMag: meanInputMag,
        outputMag: outputMagSum / windowColumns,
        inputEnergy: (() => {
          let sum = 0;
          for (let c = start; c < start + windowColumns; c++) sum += columnStats[c].inputEnergy;
          return sum / windowColumns;
        })(),
        outputEnergy: (() => {
          let sum = 0;
          for (let c = start; c < start + windowColumns; c++) sum += columnStats[c].outputEnergy;
          return sum / windowColumns;
        })(),
      };
    }
  }

  let reductionDb = 0;
  let noiseEnergyDelta = 0;
  let noiseColumnDbDelta = 0;
  let noiseCorrelation = 0;
  let dataAutocorr = null;
  if (best) {
    reductionDb = 20 * Math.log10(best.outputMag / best.inputMag);
    noiseEnergyDelta = 10 * Math.log10(best.outputEnergy / best.inputEnergy);
    noiseColumnDbDelta = best.outputDb - best.inputDb;
    const firstColumn = Math.max(best.start, 2);
    const lastColumn = Math.min(best.start + windowColumns, columns);
    const inSlice = inputEnergy.slice(firstColumn, lastColumn);
    const outSlice = outputEnergy.slice(firstColumn, lastColumn);
    noiseCorrelation = correlation(inSlice, outSlice);
    dataAutocorr = {
      firstColumn,
      lastColumn,
      columns: lastColumn - firstColumn,
      input: Array.from(autocorrelation(inSlice, 8)),
      output: Array.from(autocorrelation(outSlice, 8)),
    };
  }
  const noiseReductionOk = Boolean(best) && reductionDb <= NOISE_TARGET_DB;
  // Both panels share the speech envelope, so the whole-file correlation stays
  // high.  A copy would also have a 0 dB level delta and a noise-only
  // correlation near 1; the real path has a large level delta and no
  // correlation in the noise-only stretch.
  const notCopyOk =
    Boolean(best) &&
    wholeCorrelation < 0.9999 &&
    meanColumnDbDelta <= -6 &&
    noiseCorrelation < 0.5;

  // Painter horizontal scaling: the old integer groups changed their averaging
  // weight in steps; the current fractional filter keeps a constant total
  // weight.  At the former 1,739-column geometry and the 518-px canvas used by
  // the reference screenshot, the old step pattern repeats every 14 px.
  const scalingWidths = [518, 600, 700, 840, 900];
  const scaling = scalingWidths.map((target) => painterScaling(columns, target));
  const formerScaling = painterScaling(result.frames - LOOKAHEAD_HOPS, 518);
  const painterOk = scaling.every(
    (entry) => entry.new.maxStep < entry.old.maxStep * 0.5
  );

  console.log('');
  console.log('spectrogram buffers');
  const stride = Math.max(1, Math.ceil(result.frames / FILE_HISTORY));
  console.log(
    `  columns: ${columns} (stride ${stride}) × ${buffer.bins} bins, capacity ${buffer.capacity}`
  );
  console.log(
    `  planes: input ${buffer.input.length} bytes, output ${buffer.output.length} bytes ` +
      `(${nonZeroInput}/${nonZeroOutput} non-zero bytes)`
  );
  console.log(`  capture: ${elapsed.toFixed(1)} ms wall over ${result.frames} DSP frames`);
  console.log(`  captured dBFS range: ${minDb.toFixed(1)}…${maxDb.toFixed(1)} dB, non-finite ${nonFinite}`);
  console.log(
    `  audio with capture vs without: frames ${result.frames}/${baseline.frames}, ` +
      `max |Δsample| ${audioMaxAbsDiff.toExponential(3)}`
  );
  console.log(
    `  alignment: column c uses the input and output windows that start at ` +
      `(c - 2) × ${HOP_SIZE} samples and span ${WINDOW_SIZE} samples`
  );

  console.log('');
  console.log('capture fidelity vs the audio path');
  console.log(
    `  input panel: ${inputDiffColumns} columns, max |ΔdB| ${inputDiffMax.toExponential(3)}, ` +
      `mean ${inputDiffMean.toExponential(3)}`
  );
  console.log(
    `  output panel: ${outputDiffColumns} columns (${outputSkipped} start-up/tail columns ` +
      `outside the trimmed file), max |ΔdB| ${outputDiffMax.toExponential(3)}, ` +
      `mean ${outputDiffMean.toExponential(3)}`
  );

  console.log('');
  console.log(
    `whole file: mean input ${mean(columnStats.map((entry) => entry.inputDb)).toFixed(2)} dBFS, ` +
      `output ${mean(columnStats.map((entry) => entry.outputDb)).toFixed(2)} dBFS`
  );
  console.log(
    `  column-energy correlation input vs output: ${wholeCorrelation.toFixed(4)}`
  );
  console.log(
    `  mean per-column dB delta (output - input): ${meanColumnDbDelta.toFixed(2)} dB`
  );
  console.log(
    `  mean magnitude delta: ${wholeMagnitudeDelta.toFixed(2)} dB; ` +
      `total energy delta: ${totalEnergyDelta.toFixed(2)} dB`
  );

  if (best) {
    const startSeconds = (best.start * HOP_SIZE) / SAMPLE_RATE;
    const endSeconds = ((best.start + windowColumns) * HOP_SIZE) / SAMPLE_RATE;
    console.log('');
    console.log(
      `noise-only stretch: ${startSeconds.toFixed(2)}–${endSeconds.toFixed(2)} s ` +
        `(${windowColumns} columns, mean input ${best.inputDb.toFixed(2)} dBFS)`
    );
    console.log(
      `  mean magnitude: input ${best.inputMag.toExponential(4)}, ` +
        `output ${best.outputMag.toExponential(4)} (${reductionDb.toFixed(2)} dB)`
    );
    console.log(
      `  mean level: input ${best.inputDb.toFixed(2)} dBFS, output ${best.outputDb.toFixed(2)} dBFS ` +
        `(${noiseColumnDbDelta.toFixed(2)} dB)`
    );
    console.log(
      `  energy: ${noiseEnergyDelta.toFixed(2)} dB; ` +
        `column-energy correlation ${noiseCorrelation.toFixed(4)}`
    );
  } else {
    console.log('noise-only stretch: none found');
  }

  if (dataAutocorr) {
    console.log('');
    console.log(
      `per-column energy autocorrelation, lags 1..8 ` +
        `(columns ${dataAutocorr.firstColumn}–${dataAutocorr.lastColumn - 1})`
    );
    console.log(`  input:  ${dataAutocorr.input.map((value) => value.toFixed(3)).join(', ')}`);
    console.log(`  output: ${dataAutocorr.output.map((value) => value.toFixed(3)).join(', ')}`);
  }

  console.log('');
  console.log('painter horizontal scaling (noise gain per destination column)');
  for (const entry of scaling) {
    console.log(
      `  width ${entry.target}: group ${entry.group.toFixed(3)}, period ${entry.period} px; ` +
        `old N=${entry.oldCountMin}..${entry.oldCountMax}, ` +
        `swing ${(entry.old.swing * 100).toFixed(1)}%, step ${(entry.old.maxStep * 100).toFixed(1)}%; ` +
        `new swing ${(entry.new.swing * 100).toFixed(1)}%, step ${(entry.new.maxStep * 100).toFixed(1)}%`
    );
  }
  console.log(
    `  former geometry (${result.frames - LOOKAHEAD_HOPS} columns, 518 px): period ` +
      `${formerScaling.period} px, old step ${(formerScaling.old.maxStep * 100).toFixed(1)}%, ` +
      `new step ${(formerScaling.new.maxStep * 100).toFixed(1)}%`
  );

  const checks = {
    produced: produced ? 'ok' : 'FAIL',
    aligned: aligned ? 'ok' : 'FAIL',
    finite: finite ? 'ok' : 'FAIL',
    populated: populated ? 'ok' : 'FAIL',
    'audio-identity': audioIdentityOk ? 'ok' : 'FAIL',
    'capture-input': captureInputOk ? 'ok' : 'FAIL',
    'capture-output': captureOutputOk ? 'ok' : 'FAIL',
    'noise-reduction': noiseReductionOk ? 'ok' : 'FAIL',
    'not-copy': notCopyOk ? 'ok' : 'FAIL',
    'painter-scaling': painterOk ? 'ok' : 'FAIL',
  };
  const verdict = Object.values(checks).every((value) => value === 'ok') ? 'PASS' : 'FAIL';
  console.log('');
  console.log(`checks: ${Object.entries(checks).map(([key, value]) => `${key}=${value}`).join(', ')}`);
  console.log(
    'RESULT ' +
      JSON.stringify({
        frames: result.frames,
        columns,
        expectedColumns,
        bins: buffer.bins,
        capacity: buffer.capacity,
        planeBytes: { input: buffer.input.length, output: buffer.output.length },
        nonFinite,
        capturedDb: { min: minDb, max: maxDb },
        audioIdentity: {
          captureOffFrames: baseline.frames,
          captureOnFrames: result.frames,
          maxAbsDiff: audioMaxAbsDiff,
        },
        captureFidelity: {
          input: {
            columns: inputDiffColumns,
            maxAbsDb: inputDiffMax,
            meanAbsDb: inputDiffMean,
          },
          output: {
            columns: outputDiffColumns,
            skipped: outputSkipped,
            maxAbsDb: outputDiffMax,
            meanAbsDb: outputDiffMean,
          },
        },
        wholeFile: {
          inputDb: mean(columnStats.map((entry) => entry.inputDb)),
          outputDb: mean(columnStats.map((entry) => entry.outputDb)),
          correlation: wholeCorrelation,
          meanColumnDbDelta,
          magnitudeDeltaDb: wholeMagnitudeDelta,
          energyDeltaDb: totalEnergyDelta,
        },
        noiseStretch: best
          ? {
              startSeconds: (best.start * HOP_SIZE) / SAMPLE_RATE,
              endSeconds: ((best.start + windowColumns) * HOP_SIZE) / SAMPLE_RATE,
              inputDb: best.inputDb,
              outputDb: best.outputDb,
              inputMag: best.inputMag,
              outputMag: best.outputMag,
              reductionDb,
              energyDeltaDb: noiseEnergyDelta,
              columnDbDelta: noiseColumnDbDelta,
              correlation: noiseCorrelation,
            }
          : null,
        dataAutocorr,
        painterScaling: {
          widths: scaling,
          formerGeometry: formerScaling,
        },
        captureMs: elapsed,
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
