#!/usr/bin/env node
/**
 * Node-level streaming check for the live (microphone) DSP.
 *
 * Feeds the same 16 kHz WAV through the exact `LiveDenoiser` /
 * `StreamingDenoiser` code the AudioWorklet imports, one 256-sample hop at a
 * time, with the same zero padding and hop-flush the live path needs.  It then
 * compares the streaming output with the validated offline `processPcm()`
 * result and checks for discontinuities at hop boundaries.
 *
 * It also exercises the sample-rate-conversion path with a 48 kHz round trip
 * (16 kHz -> 48 kHz -> LiveDenoiser at 48 kHz -> 16 kHz) and reports how
 * closely it matches the offline result.
 *
 * Usage:
 *   node test/node-live.mjs [input.wav] [--wasm path]
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  HOP_SIZE,
  LATENCY_SAMPLES,
  SAMPLE_RATE,
  loadVoiceEngine,
  processPcm,
} from '../dsp.js';
import { LinearResampler, LiveDenoiser } from '../live-dsp.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_INPUT = path.resolve(HERE, '../../../local-denoiser/samples/noisy.wav');
const DEFAULT_WASM = path.resolve(HERE, '../cf_wasm.wasm');
const HOP_MS = (HOP_SIZE / SAMPLE_RATE) * 1000;
const OUTPUT_RATE = 48000;

function usage() {
  console.log('usage: node test/node-live.mjs [input.wav] [--wasm path]');
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

function maxAbs(samples) {
  let peak = 0;
  let bad = 0;
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i];
    if (!Number.isFinite(value)) {
      bad += 1;
      continue;
    }
    const abs = Math.abs(value);
    if (abs > peak) peak = abs;
  }
  return { peak, bad };
}

function maxDelta(samples, start, step, predicate) {
  let max = 0;
  for (let i = Math.max(1, start); i < samples.length; i++) {
    if (predicate && !predicate(i)) continue;
    const delta = Math.abs(samples[i] - samples[i - 1]);
    if (delta > max) max = delta;
  }
  return max;
}

function compare(a, b, count) {
  let maxDiff = 0;
  let sum = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  const n = Math.min(count, a.length, b.length);
  for (let i = 0; i < n; i++) {
    const diff = a[i] - b[i];
    const abs = Math.abs(diff);
    if (abs > maxDiff) maxDiff = abs;
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
    maxAbsDiff: maxDiff,
    rmsDiff: Math.sqrt(sum / n),
    correlation: denominator > 0 ? numerator / denominator : 0,
  };
}

function collectResampled(resampler, input, chunkSize) {
  const parts = [];
  let total = 0;
  for (let offset = 0; offset < input.length; offset += chunkSize) {
    const chunk = input.subarray(offset, Math.min(input.length, offset + chunkSize));
    const out = resampler.process(chunk);
    parts.push(out.slice());
    total += out.length;
  }
  const joined = new Float32Array(total);
  let at = 0;
  for (const part of parts) {
    joined.set(part, at);
    at += part.length;
  }
  return joined;
}

function streaming16k(engine, input, offlineFrames) {
  const live = new LiveDenoiser(engine, { inputRate: SAMPLE_RATE, outputRate: SAMPLE_RATE });
  const blocks = [];
  let outputLength = 0;
  let hops = 0;
  let hopMsTotal = 0;
  const hopTimes = [];
  const block = new Float32Array(HOP_SIZE);
  const out = new Float32Array(HOP_SIZE);
  const push = (samples) => {
    blocks.push(samples.slice());
    outputLength += samples.length;
  };
  const feed = (samples) => {
    const beforeFrames = live.frames;
    const beforeMs = live.streaming.hopMsTotal;
    block.fill(0);
    block.set(samples.subarray(0, Math.min(HOP_SIZE, samples.length)));
    live.processBlock(block, out);
    // Every hop block produces a full 256-sample output block.
    push(out);
    if (live.frames > beforeFrames) {
      hops += live.frames - beforeFrames;
      const deltaMs = live.streaming.hopMsTotal - beforeMs;
      hopMsTotal += deltaMs;
      hopTimes.push(deltaMs / (live.frames - beforeFrames));
    }
  };

  const started = performance.now();
  for (let offset = 0; offset < input.length; offset += HOP_SIZE) {
    feed(input.subarray(offset, Math.min(input.length, offset + HOP_SIZE)));
  }
  const zero = new Float32Array(HOP_SIZE);
  let guard = 0;
  while (live.frames < offlineFrames && guard < 64) {
    feed(zero);
    guard += 1;
  }
  const wallMs = performance.now() - started;

  const liveOutput = new Float32Array(outputLength);
  let at = 0;
  for (const b of blocks) {
    liveOutput.set(b, at);
    at += b.length;
  }
  return {
    live,
    output: liveOutput,
    frames: live.frames,
    hops,
    wallMs,
    hopMsTotal,
    hopTimes,
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
  const module = new WebAssembly.Module(wasmBytes);
  const engine = await loadVoiceEngine(module);
  console.log(`engine: backend=${engine.backendName || '?'}, model=${engine.modelImageBytes} bytes`);

  // Validated offline baseline (this resets the engine).
  const offline = processPcm(input.samples, engine);
  console.log(
    `offline: ${offline.frames} frames, ${offline.processingMs.toFixed(1)} ms, ` +
      `RTF ${offline.rtf.toFixed(4)}`
  );

  // Live path: same engine instance, reset once by the LiveDenoiser constructor.
  const stream = streaming16k(engine, input.samples, offline.frames);
  const latency = LATENCY_SAMPLES;
  const outputOffset = latency - HOP_SIZE; // live block 0 is raw output sample 256.
  if (stream.output.length < outputOffset + input.samples.length) {
    throw new Error(
      `stream produced ${stream.output.length} samples, need ${outputOffset + input.samples.length}`
    );
  }
  const aligned = stream.output.subarray(outputOffset, outputOffset + input.samples.length);

  const stats = compare(aligned, offline.samples, input.samples.length);
  const livePeak = maxAbs(aligned);
  const nanOffline = maxAbs(offline.samples).bad;
  const maxJumpLive = maxDelta(aligned, 1, HOP_SIZE);
  const maxJumpOffline = maxDelta(offline.samples, 1, HOP_SIZE);
  const boundaryLive = maxDelta(aligned, 1, HOP_SIZE, (i) => i % HOP_SIZE === 0);
  const boundaryOffline = maxDelta(offline.samples, 1, HOP_SIZE, (i) => i % HOP_SIZE === 0);
  const wallRtf = stream.wallMs / (stream.frames * HOP_MS);
  const timingAvg = stream.hopMsTotal / stream.frames;
  const timingMin = stream.hopTimes.length ? Math.min(...stream.hopTimes) : 0;
  const timingMax = stream.hopTimes.length ? Math.max(...stream.hopTimes) : 0;

  console.log('');
  console.log('live (256-sample hops, same DSP as the worklet)');
  console.log(
    `  latency: ${latency} samples = ${((latency / SAMPLE_RATE) * 1000).toFixed(1)} ms ` +
      `(768 window + 2 * 256 look-ahead)`
  );
  console.log(`  frames processed: ${stream.frames} (offline ${offline.frames})`);
  console.log(`  hop count: ${stream.hops}`);
  console.log(`  output samples: ${stream.output.length} (aligned offset ${outputOffset})`);
  console.log(`  DSP time: ${stream.hopMsTotal.toFixed(1)} ms`);
  console.log(
    `  frame time: ${timingAvg.toFixed(3)} ms / frame ` +
      `(min ${timingMin.toFixed(3)}, max ${timingMax.toFixed(3)}; ${HOP_MS.toFixed(1)} ms audio per frame)`
  );
  console.log(`  stream wall time: ${stream.wallMs.toFixed(1)} ms`);
  console.log(`  RTF (wall / audio): ${wallRtf.toFixed(4)}`);
  console.log(
    `  vs offline: maxAbsDiff=${stats.maxAbsDiff.toExponential(3)}, ` +
      `rmsDiff=${stats.rmsDiff.toExponential(3)}, correlation=${stats.correlation.toFixed(9)}`
  );
  console.log(`  non-finite samples (live/offline): ${livePeak.bad}/${nanOffline}`);
  console.log(`  peak: ${livePeak.peak.toFixed(6)} (offline ${offline.peakAfter.toFixed(6)})`);
  console.log(
    `  max sample-to-sample delta: live=${maxJumpLive.toFixed(6)}, offline=${maxJumpOffline.toFixed(6)}`
  );
  console.log(
    `  max delta at hop boundaries: live=${boundaryLive.toFixed(6)}, ` +
      `offline=${boundaryOffline.toFixed(6)}`
  );

  let resampleStats = null;
  let resampleOk = true;
  try {
    // 16 kHz -> 48 kHz by linear interpolation, with a small tail pad.
    const padded = new Float32Array(input.samples.length + 4);
    padded.set(input.samples);
    const up = new LinearResampler(SAMPLE_RATE, OUTPUT_RATE);
    const input48 = collectResampled(up, padded, 4096);

    const live48 = new LiveDenoiser(engine, {
      inputRate: OUTPUT_RATE,
      outputRate: OUTPUT_RATE,
    });
    const block48 = new Float32Array(128);
    const out48 = new Float32Array(128);
    const parts = [];
    let total48 = 0;
    const feed48 = (src, count) => {
      block48.fill(0);
      block48.set(src.subarray(0, count));
      live48.processBlock(block48, out48);
      parts.push(out48.slice());
      total48 += out48.length;
    };
    const start48 = performance.now();
    for (let offset = 0; offset < input48.length; offset += 128) {
      feed48(
        input48.subarray(offset, Math.min(input48.length, offset + 128)),
        Math.min(128, input48.length - offset)
      );
    }
    const zero48 = new Float32Array(128);
    let guard = 0;
    while (live48.frames < offline.frames && guard < 256) {
      feed48(zero48, 128);
      guard += 1;
    }
    // Drain the output queue.
    for (let i = 0; i < 8; i++) feed48(zero48, 128);
    const wall48 = performance.now() - start48;
    const output48 = new Float32Array(total48);
    let at = 0;
    for (const part of parts) {
      output48.set(part, at);
      at += part.length;
    }
    const down = new LinearResampler(OUTPUT_RATE, SAMPLE_RATE);
    const down16 = collectResampled(down, output48, 4096);

    let best = { lag: 0, correlation: -1, rmsDiff: Infinity };
    for (let lag = 800; lag <= 1400; lag += 1) {
      if (lag + offline.samples.length > down16.length) break;
      const slice = down16.subarray(lag, lag + offline.samples.length);
      const c = compare(slice, offline.samples, offline.samples.length);
      if (c.correlation > best.correlation) best = { lag, ...c };
    }
    resampleStats = {
      frames: live48.frames,
      wallMs: wall48,
      rtf: wall48 / (live48.frames * HOP_MS),
      lag: best.lag,
      correlation: best.correlation,
      maxAbsDiff: best.maxAbsDiff,
      rmsDiff: best.rmsDiff,
    };
    resampleOk = best.correlation > 0.98;
  } catch (error) {
    resampleOk = false;
    resampleStats = { error: error && error.message ? error.message : String(error) };
  }

  console.log('');
  if (resampleStats && !resampleStats.error) {
    console.log('48 kHz context round trip (linear resampling path)');
    console.log(`  frames processed: ${resampleStats.frames}`);
    console.log(
      `  wall time: ${resampleStats.wallMs.toFixed(1)} ms, ` +
        `RTF ${resampleStats.rtf.toFixed(4)}`
    );
    console.log(
      `  best alignment lag: ${resampleStats.lag} samples; ` +
        `correlation=${resampleStats.correlation.toFixed(6)}, ` +
        `maxAbsDiff=${resampleStats.maxAbsDiff.toFixed(6)}, ` +
        `rmsDiff=${resampleStats.rmsDiff.toFixed(6)}`
    );
  } else {
    console.log(`48 kHz context round trip: ${resampleStats.error}`);
  }

  const continuous =
    stats.maxAbsDiff < 1e-6 &&
    stats.correlation > 0.999999 &&
    livePeak.bad === 0 &&
    nanOffline === 0 &&
    stream.frames === offline.frames &&
    livePeak.peak <= 1.0000001;
  const noBoundaryClicks =
    boundaryLive <= Math.max(maxJumpLive, 1e-9) &&
    Math.abs(boundaryLive - boundaryOffline) < 1e-6;
  const verdict = continuous && noBoundaryClicks ? 'PASS' : 'FAIL';
  console.log('');
  console.log(
    `checks: continuous=${continuous ? 'ok' : 'FAIL'}, ` +
      `no-nan=${livePeak.bad === 0 ? 'ok' : 'FAIL'}, ` +
      `frames=${stream.frames === offline.frames ? 'ok' : 'FAIL'}, ` +
      `boundary=${noBoundaryClicks ? 'ok' : 'FAIL'}, ` +
      `resample=${resampleStats && !resampleStats.error ? (resampleOk ? 'ok (informational)' : 'weak (informational)') : 'n/a'}`
  );
  console.log(
    'RESULT ' +
      JSON.stringify({
        inputSamples: input.samples.length,
        offlineFrames: offline.frames,
        liveFrames: stream.frames,
        frameMs: timingAvg,
        rtf: wallRtf,
        wallMs: stream.wallMs,
        dspMs: stream.hopMsTotal,
        maxAbsDiff: stats.maxAbsDiff,
        rmsDiff: stats.rmsDiff,
        correlation: stats.correlation,
        nonFinite: livePeak.bad + nanOffline,
        boundaryDelta: { live: boundaryLive, offline: boundaryOffline },
        maxDelta: { live: maxJumpLive, offline: maxJumpOffline },
        peak: livePeak.peak,
        resample: resampleStats,
        resampleOk,
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
