#!/usr/bin/env node
/**
 * Node-level DSP check for the static web demo.
 *
 * This bypasses the browser decoders, reads a WAV file directly (16/24/32-bit
 * PCM or float), feeds float samples into the same dsp.js pipeline the worker
 * uses, and checks the result is audible and unclipped.
 *
 * Usage:
 *   node node-check.mjs [input.wav] [--wasm path] [--out out.wav]
 *                       [--reference reference.wav]
 *
 * The default input is the repository sample at
 * ../../../local-denoiser/samples/noisy.wav (relative to this file).
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  loadVoiceEngine,
  processPcm,
  encodeWav,
  SAMPLE_RATE,
  LATENCY_SAMPLES,
} from '../dsp.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = path.resolve(HERE, '../../../local-denoiser/samples/noisy.wav');
const DEFAULT_WASM = path.resolve(HERE, '../cf_wasm.wasm');

function usage() {
  console.log(
    'usage: node node-check.mjs [input.wav] [--wasm path] [--out path] [--reference path]'
  );
}

function parseArgs(argv) {
  const options = { input: null, wasm: DEFAULT_WASM, out: null, reference: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else if (arg === '--wasm') {
      options.wasm = path.resolve(argv[++i]);
    } else if (arg === '--out') {
      options.out = path.resolve(argv[++i]);
    } else if (arg === '--reference') {
      options.reference = path.resolve(argv[++i]);
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

/**
 * Minimal RIFF/WAVE reader.  Returns mono float32 samples in [-1, 1].
 * Supports PCM 16/24/32-bit and IEEE float 32/64-bit, any channel count.
 */
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
  if (format === 1 && ![16, 24, 32].includes(bits)) {
    throw new Error(`${file}: unsupported PCM depth ${bits}`);
  }
  if (format === 3 && ![32, 64].includes(bits)) {
    throw new Error(`${file}: unsupported float depth ${bits}`);
  }

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

function dbFs(value) {
  return value > 0 ? 20 * Math.log10(value) : -Infinity;
}

function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return samples.length ? Math.sqrt(sum / samples.length) : 0;
}

function peak(samples) {
  let p = 0;
  for (let i = 0; i < samples.length; i++) p = Math.max(p, Math.abs(samples[i]));
  return p;
}

function correlation(a, b) {
  const n = Math.min(a.length, b.length);
  let sa = 0;
  let sb = 0;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sa += a[i];
    sb += b[i];
    sab += a[i] * b[i];
    saa += a[i] * a[i];
    sbb += b[i] * b[i];
  }
  const numerator = n * sab - sa * sb;
  const denominator = Math.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb));
  return denominator > 0 ? numerator / denominator : 0;
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

  const readStart = performance.now();
  const wasmBytes = readFileSync(options.wasm);
  const readMs = performance.now() - readStart;
  console.log(`wasm: ${options.wasm} (${wasmBytes.length} bytes, read ${readMs.toFixed(2)} ms)`);

  const compileStart = performance.now();
  const module = new WebAssembly.Module(wasmBytes);
  const compileMs = performance.now() - compileStart;
  const instantiateStart = performance.now();
  const engine = await loadVoiceEngine(module);
  const instantiateMs = performance.now() - instantiateStart;
  console.log(
    `wasm compile: ${compileMs.toFixed(2)} ms; instantiate + model: ${instantiateMs.toFixed(2)} ms`
  );

  console.log(
    `engine: backend=${engine.backendName || '?'}, model=${engine.modelImageBytes} bytes, ` +
      `workspace=${engine.workspaceBytes} bytes, features=${513}, outputs=${1026}`
  );

  // Cold first engine frame (one 513-value feature vector).
  engine.reset();
  const firstFrameStart = performance.now();
  engine.process(new Float32Array(513));
  const firstFrameMs = performance.now() - firstFrameStart;
  console.log(`first engine frame: ${firstFrameMs.toFixed(3)} ms`);

  const started = performance.now();
  const result = processPcm(input.samples, engine);
  const wallMs = performance.now() - started;

  console.log(`frames processed: ${result.frames}`);
  console.log(
    `processing time: ${result.processingMs.toFixed(1)} ms ` +
      `(wall ${wallMs.toFixed(1)} ms)`
  );
  console.log(`audio duration: ${result.durationSeconds.toFixed(3)} s`);
  console.log(`RTF: ${result.rtf.toFixed(4)}`);
  console.log(
    `RMS before: ${result.rmsBefore.toFixed(6)} (${dbFs(result.rmsBefore).toFixed(2)} dBFS)`
  );
  console.log(
    `RMS after:  ${result.rmsAfter.toFixed(6)} (${dbFs(result.rmsAfter).toFixed(2)} dBFS)`
  );
  console.log(`peak before: ${result.peakBefore.toFixed(6)}`);
  console.log(`peak after:  ${result.peakAfter.toFixed(6)}`);
  console.log(`clipped samples (>1.0): ${result.clippedSamples}`);
  console.log(`output samples: ${result.samples.length} (latency ${LATENCY_SAMPLES} removed)`);

  if (options.out) {
    writeFileSync(options.out, Buffer.from(encodeWav(result.samples)));
    console.log(`wrote: ${options.out}`);
  }

  let referenceStats = null;
  if (options.reference) {
    const reference = readWav(options.reference);
    if (reference.frames !== result.samples.length) {
      throw new Error(
        `reference length ${reference.frames} != output length ${result.samples.length}`
      );
    }
    let maxDiff = 0;
    let diffSum = 0;
    for (let i = 0; i < reference.samples.length; i++) {
      const diff = Math.abs(reference.samples[i] - result.samples[i]);
      if (diff > maxDiff) maxDiff = diff;
      diffSum += diff * diff;
    }
    referenceStats = {
      maxAbsDiff: maxDiff,
      rmsDiff: Math.sqrt(diffSum / reference.samples.length),
      correlation: correlation(reference.samples, result.samples),
    };
    console.log(
      `vs reference: maxAbsDiff=${maxDiff.toFixed(6)}, ` +
        `rmsDiff=${referenceStats.rmsDiff.toFixed(6)}, ` +
        `correlation=${referenceStats.correlation.toFixed(6)}`
    );
  }

  const notSilent = result.rmsAfter > 1e-4 && result.peakAfter > 0.01;
  const aligned = result.samples.length === input.samples.length;
  const verdict = result.clippedSamples === 0 && notSilent && aligned ? 'PASS' : 'FAIL';
  console.log(
    `checks: clipping=${result.clippedSamples === 0 ? 'ok' : 'FAIL'}, ` +
      `audible=${notSilent ? 'ok' : 'FAIL'}, aligned=${aligned ? 'ok' : 'FAIL'}`
  );
  console.log(
    'RESULT ' +
      JSON.stringify({
        frames: result.frames,
        processingMs: result.processingMs,
        rtf: result.rtf,
        durationSeconds: result.durationSeconds,
        rmsBefore: result.rmsBefore,
        rmsAfter: result.rmsAfter,
        peakBefore: result.peakBefore,
        peakAfter: result.peakAfter,
        clippedSamples: result.clippedSamples,
        wasmBytes: wasmBytes.length,
        modelImageBytes: engine.modelImageBytes,
        compileMs,
        firstFrameMs,
        verdict,
        reference: referenceStats,
      })
  );

  engine.close();
  if (verdict !== 'PASS') process.exitCode = 1;
}

main().catch((error) => {
  console.error(`error: ${error.stack || error.message || error}`);
  process.exitCode = 1;
});
