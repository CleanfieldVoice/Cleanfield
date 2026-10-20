/**
 * CleanField Voice 1.0 — shared live (microphone) DSP.
 *
 * This module is plain ES module JavaScript with no dependencies beyond the
 * numerical primitives in `dsp.js`.  It is imported unchanged by the
 * AudioWorklet processor and by the Node streaming test, so the exact code
 * that runs on the audio render thread is covered without a browser.
 *
 * Contract: 16 kHz mono, 1024-point FFT, 768-sample periodic Hann window,
 * 256-sample hop, feature ln(|X[k]|^2 + 1e-20), mask from the engine's 1026
 * outputs, inverse FFT, overlap-add with the 8/9 synthesis normalisation.
 * One `StreamingDenoiser` keeps the engine's cross-frame state (5-frame history
 * and LSTM state); it is reset once in the constructor and never mid-stream.
 */

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
  VoiceEngine,
  WINDOW,
  WINDOW_SIZE,
} from './dsp.js';

const RING_SIZE = 1024;
const RING_MASK = RING_SIZE - 1;

const clock =
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? () => performance.now()
    : () => Date.now();

function fail(message) {
  const error = new Error(message);
  error.userFacing = true;
  return error;
}

/**
 * Frame-by-frame 16 kHz denoiser for live input.
 *
 * The arithmetic mirrors `processPcm()` frame for frame: the same analysis
 * window, feature scaling, mask, synthesis window, overlap-add and 8/9
 * normalisation.  Output is delayed by `LATENCY_SAMPLES` (1280) relative to
 * the input.  `processHop()` returns a reused 256-sample buffer.
 */
export class StreamingDenoiser {
  constructor(engine) {
    if (!(engine instanceof VoiceEngine)) throw fail('The denoising engine is not valid.');
    this.engine = engine;
    this.ring = new Float32Array(RING_SIZE);
    this.writeIndex = 0;
    this.analysis = new Float32Array(WINDOW_SIZE);
    this.overlap = new Float32Array(WINDOW_SIZE);
    this.fftRe = new Float64Array(FFT_SIZE);
    this.fftIm = new Float64Array(FFT_SIZE);
    this.specRe = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
    this.specIm = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
    this.curRe = new Float32Array(BIN_COUNT);
    this.curIm = new Float32Array(BIN_COUNT);
    this.features = new Float32Array(BIN_COUNT);
    this.mask = new Float32Array(BIN_COUNT);
    this.hopOutput = new Float32Array(HOP_SIZE);
    this.frames = 0;
    this.hopMsTotal = 0;
    // A live session must not reset the engine after the stream starts.
    engine.reset();
  }

  get averageHopMs() {
    return this.frames > 0 ? this.hopMsTotal / this.frames : 0;
  }

  /**
   * Push one 256-sample hop and return 256 output samples (reused array).
   * @param {Float32Array} hop
   * @returns {Float32Array}
   */
  processHop(hop) {
    if (!(hop instanceof Float32Array) || hop.length !== HOP_SIZE) {
      throw fail(`A live hop must contain exactly ${HOP_SIZE} samples.`);
    }
    const started = clock();

    // Ring-buffer the new hop, then read the most recent 768 samples in order.
    let write = this.writeIndex;
    const ring = this.ring;
    for (let i = 0; i < HOP_SIZE; i++) ring[(write + i) & RING_MASK] = hop[i];
    write = (write + HOP_SIZE) & RING_MASK;
    this.writeIndex = write;

    const analysis = this.analysis;
    let read = (write - WINDOW_SIZE) & RING_MASK;
    for (let i = 0; i < WINDOW_SIZE; i++) {
      analysis[i] = ring[read];
      read = (read + 1) & RING_MASK;
    }

    // Forward transform of the windowed 768 samples, zero-padded to 1024.
    const fftRe = this.fftRe;
    const fftIm = this.fftIm;
    fftRe.fill(0);
    fftIm.fill(0);
    for (let i = 0; i < WINDOW_SIZE; i++) {
      fftRe[i] = Math.fround(analysis[i] * WINDOW[i]);
    }
    FFT_1024.forward(fftRe, fftIm);

    const curRe = this.curRe;
    const curIm = this.curIm;
    for (let k = 0; k < BIN_COUNT; k++) {
      curRe[k] = fftRe[k];
      curIm[k] = fftIm[k];
    }

    const features = this.features;
    for (let k = 0; k < BIN_COUNT; k++) {
      const re = curRe[k];
      const im = curIm[k];
      const power = Math.fround(Math.fround(re * re) + Math.fround(im * im));
      const scaled = Math.fround(power * FEATURE_POWER_SCALE);
      features[k] = Math.fround(Math.log(Math.fround(scaled + EPSILON)));
    }

    const prediction = this.engine.process(features);

    const mask = this.mask;
    for (let k = 0; k < BIN_COUNT; k++) {
      const re = prediction[k];
      const im = prediction[k + BIN_COUNT];
      const power = Math.fround(
        Math.fround(Math.fround(re * re) + Math.fround(im * im)) + EPSILON
      );
      const magnitude = Math.fround(Math.sqrt(power));
      const m = Math.fround(magnitude * MASK_GAIN);
      mask[k] = m > MASK_FLOOR ? (m < 1 ? m : 1) : MASK_FLOOR;
    }

    // The mask from this frame is applied to the spectrum from two hops ago.
    const frame = this.frames;
    const slot = frame & 1;
    const targetRe = this.specRe[slot];
    const targetIm = this.specIm[slot];
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

    const overlap = this.overlap;
    for (let i = 0; i < WINDOW_SIZE; i++) {
      const reconstructed = Math.fround(fftRe[i]);
      const windowed = Math.fround(Math.fround(reconstructed * WINDOW[i]) * SYNTHESIS_SCALE);
      overlap[i] = Math.fround(overlap[i] + windowed);
    }

    // The current spectrum becomes the look-ahead history for later frames.
    targetRe.set(curRe);
    targetIm.set(curIm);

    const output = this.hopOutput;
    for (let i = 0; i < HOP_SIZE; i++) output[i] = overlap[i];
    overlap.copyWithin(0, HOP_SIZE);
    overlap.fill(0, WINDOW_SIZE - HOP_SIZE);

    this.frames = frame + 1;
    this.hopMsTotal += clock() - started;
    return output;
  }
}

/**
 * Linear sample-rate converter with a fractional read position.
 *
 * Used only when the AudioContext refuses the requested 16 kHz rate; at
 * 16 kHz the live pipeline runs without any conversion.
 */
export class LinearResampler {
  constructor(inputRate, outputRate) {
    if (!(inputRate > 0) || !(outputRate > 0)) {
      throw fail('Sample rates must be positive numbers.');
    }
    this.step = inputRate / outputRate;
    this.buffer = new Float32Array(4096);
    this.length = 0;
    this.position = 0;
    this.output = new Float32Array(1024);
  }

  /** @param {Float32Array} input @returns {Float32Array} resampled samples */
  process(input) {
    const chunk = input instanceof Float32Array ? input : Float32Array.from(input);
    this._append(chunk);

    const needed = Math.ceil(chunk.length / this.step) + 8;
    if (this.output.length < needed) this.output = new Float32Array(needed);

    const buffer = this.buffer;
    const output = this.output;
    const limit = this.length - 1;
    let position = this.position;
    let count = 0;
    while (position < limit) {
      const index = Math.floor(position);
      const fraction = position - index;
      const a = buffer[index];
      output[count++] = a + (buffer[index + 1] - a) * fraction;
      position += this.step;
    }
    this.position = position;

    if (this.position > 2048) this._compact();
    return output.subarray(0, count);
  }

  _append(chunk) {
    if (this.length + chunk.length > this.buffer.length) {
      this._compact();
      if (this.length + chunk.length > this.buffer.length) {
        const next = new Float32Array(
          Math.max(this.buffer.length * 2, this.length + chunk.length + 16)
        );
        next.set(this.buffer.subarray(0, this.length));
        this.buffer = next;
      }
    }
    this.buffer.set(chunk, this.length);
    this.length += chunk.length;
  }

  _compact() {
    const drop = Math.floor(this.position) - 1;
    if (drop > 0) {
      this.buffer.copyWithin(0, drop, this.length);
      this.length -= drop;
      this.position -= drop;
    }
  }
}

function normalizeGain(value) {
  const gain = Number(value);
  if (!Number.isFinite(gain) || gain < 0) return 1;
  return Math.min(8, Math.max(0, gain));
}

/**
 * Block-based live denoiser shared by the AudioWorklet and the Node test.
 *
 * Input and output blocks are at the same rate (the AudioContext rate):
 * capture is converted to 16 kHz when needed, processed one 256-sample hop at
 * a time, then converted back for playback.  The engine keeps advancing in Raw
 * mode so an A/B switch never resets the model.
 */
export class LiveDenoiser {
  constructor(engine, options = {}) {
    if (!(engine instanceof VoiceEngine)) throw fail('The denoising engine is not valid.');
    const inputRate = Number(options.inputRate) > 0 ? Number(options.inputRate) : SAMPLE_RATE;
    const outputRate = Number(options.outputRate) > 0 ? Number(options.outputRate) : SAMPLE_RATE;
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.streaming = new StreamingDenoiser(engine);
    this.gain = normalizeGain(options.gain);
    this.bypass = Boolean(options.bypass);
    this.down = inputRate === SAMPLE_RATE ? null : new LinearResampler(inputRate, SAMPLE_RATE);
    this.up = outputRate === SAMPLE_RATE ? null : new LinearResampler(SAMPLE_RATE, outputRate);
    this.hop = new Float32Array(HOP_SIZE);
    this.hopFill = 0;
    this.queue = new Float32Array(8192);
    this.queueHead = 0;
    this.queueTail = 0;
  }

  get frames() {
    return this.streaming.frames;
  }

  get averageHopMs() {
    return this.streaming.averageHopMs;
  }

  get latencySamples() {
    return LATENCY_SAMPLES;
  }

  setGain(value) {
    this.gain = normalizeGain(value);
  }

  setBypass(value) {
    this.bypass = Boolean(value);
  }

  /**
   * Process one block at the context rate.
   * @param {Float32Array} input
   * @param {Float32Array} output same length as input
   */
  processBlock(input, output) {
    if (!(input instanceof Float32Array) || !(output instanceof Float32Array)) {
      throw fail('Live audio blocks must be Float32Array instances.');
    }
    const length = Math.min(input.length, output.length);
    if (length === 0) return;

    // 1. Downsample the capture to 16 kHz when the context rate differs.
    let pcm = input;
    let count = length;
    if (this.down) {
      pcm = this.down.process(length === input.length ? input : input.subarray(0, length));
      count = pcm.length;
    }

    // 2. Accumulate hops; the engine is advanced in Raw mode too.
    for (let i = 0; i < count; i++) {
      this.hop[this.hopFill++] = pcm[i];
      if (this.hopFill === HOP_SIZE) {
        const hopOutput = this.streaming.processHop(this.hop);
        const gain = this.gain;
        if (gain === 1) {
          this._enqueue(hopOutput, HOP_SIZE);
        } else {
          for (let j = 0; j < HOP_SIZE; j++) hopOutput[j] *= gain;
          this._enqueue(hopOutput, HOP_SIZE);
        }
        this.hopFill = 0;
      }
    }

    // 3. Playback block.
    if (this.bypass) {
      output.set(input.subarray(0, length));
      // Keep the processed queue in step without playing it, so switching back
      // to Denoised does not replay stale audio.
      this._discard(length);
      return;
    }

    let written = 0;
    const available = this.queueTail - this.queueHead;
    if (available > 0) {
      written = Math.min(available, length);
      output.set(this.queue.subarray(this.queueHead, this.queueHead + written), 0);
      this.queueHead += written;
      if (this.queueHead === this.queueTail) {
        this.queueHead = 0;
        this.queueTail = 0;
      }
    }
    if (written < length) output.fill(0, written, length);
  }

  _enqueue(samples, count) {
    let source = samples;
    let sourceCount = count;
    if (this.up) {
      const resampled = this.up.process(samples.subarray(0, count));
      source = resampled;
      sourceCount = resampled.length;
    }
    if (sourceCount === 0) return;

    if (this.queueTail + sourceCount > this.queue.length) {
      const used = this.queueTail - this.queueHead;
      if (used + sourceCount > this.queue.length) {
        const next = new Float32Array(
          Math.max(this.queue.length * 2, used + sourceCount + 64)
        );
        next.set(this.queue.subarray(this.queueHead, this.queueTail));
        this.queue = next;
        this.queueHead = 0;
        this.queueTail = used;
      } else {
        this.queue.copyWithin(0, this.queueHead, this.queueTail);
        this.queueHead = 0;
        this.queueTail = used;
      }
    }
    this.queue.set(source.subarray(0, sourceCount), this.queueTail);
    this.queueTail += sourceCount;
  }

  _discard(count) {
    const available = this.queueTail - this.queueHead;
    const take = Math.min(available, count);
    this.queueHead += take;
    if (this.queueHead === this.queueTail) {
      this.queueHead = 0;
      this.queueTail = 0;
    }
  }
}
