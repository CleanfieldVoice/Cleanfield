/**
 * CleanField Voice 1.0 — live AudioWorklet processor.
 *
 * Runs on the audio render thread: one microphone input, the voice engine, and
 * the denoised output.  The wasm engine is instantiated *inside* the worklet.
 * AudioWorkletGlobalScope has no `fetch` in current browsers, so the worklet
 * first resolves the wasm URL from its own module URL and asks the page for the
 * bytes; in a browser that exposes `fetch` on the worklet global, it fetches
 * directly instead.
 *
 * All parameters (bypass, output gain, maximum attenuation) arrive as messages
 * and are applied without resetting the engine or reloading the model.
 */

import { HOP_SIZE, LATENCY_SAMPLES, SAMPLE_RATE, VoiceEngine } from './dsp.js';
import { LiveDenoiser } from './live-dsp.js';
import {
  SPEC_BATCH_COLUMNS,
  SPEC_BINS,
  writeQuantizedColumn,
} from './spectrogram-core.js';

const WASM_FILE = 'cf_wasm.wasm';
const RENDER_QUANTUM = 128;

function toDb(value) {
  return value > 0 ? Math.max(-100, 20 * Math.log10(value)) : -100;
}

function resolveSiblingUrl(file, fallback) {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.url) {
      return import.meta.url.slice(0, import.meta.url.lastIndexOf('/') + 1) + file;
    }
  } catch {
    // Fall through to the URL supplied by the page.
  }
  return fallback || null;
}

function clampGain(value) {
  const gain = Number(value);
  if (!Number.isFinite(gain) || gain < 0) return 1;
  return Math.min(8, gain);
}

function clampMaxAttenuationDb(value) {
  const db = Number(value);
  if (!Number.isFinite(db)) return -80;
  return db < -80 ? -80 : db > 0 ? 0 : db;
}

class LiveDenoiseProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = (options && options.processorOptions) || {};
    this.wasmUrl = resolveSiblingUrl(WASM_FILE, processorOptions.wasmUrl);
    this.engine = null;
    this.live = null;
    this.failed = false;
    this.wasmWaiter = null;
    this.gain = clampGain(processorOptions.gain);
    this.maxAttenuationDb = clampMaxAttenuationDb(processorOptions.maxAttenuationDb);
    this.bypass = Boolean(processorOptions.bypass);
    this.latencySamples = LATENCY_SAMPLES;
    this.frames = 0;
    this.lastFrames = 0;
    this.lastHopMs = 0;
    this.frameMs = 0;
    this.rtf = 0;
    this.levelBlocks = 0;
    this.reportEvery = Math.max(1, Math.round((0.1 * sampleRate) / RENDER_QUANTUM));
    this.inSquares = 0;
    this.outSquares = 0;
    this.peakIn = 0;
    this.peakOut = 0;
    this.levelSamples = 0;
    this.levelIn = -100;
    this.levelOut = -100;
    // Batched spectrogram columns: [input plane | output plane] per batch.
    this.specBatch = new Uint8Array(SPEC_BATCH_COLUMNS * SPEC_BINS * 2);
    this.specCount = 0;

    this.port.onmessage = (event) => this.handleMessage(event.data);
    this.port.postMessage({
      type: 'hello',
      contextSampleRate: sampleRate,
      wasmUrl: this.wasmUrl,
    });
    void this.start();
  }

  handleMessage(message) {
    if (!message || typeof message !== 'object') return;
    if (message.type === 'wasm' && message.bytes) {
      if (this.wasmWaiter) {
        const waiter = this.wasmWaiter;
        this.wasmWaiter = null;
        waiter.resolve(message.bytes);
      }
      return;
    }
    if (message.type === 'wasm-error') {
      if (this.wasmWaiter) {
        const waiter = this.wasmWaiter;
        this.wasmWaiter = null;
        waiter.reject(new Error(message.message || 'Could not load the engine.'));
      }
      return;
    }
    if (message.type === 'bypass') {
      this.bypass = Boolean(message.value);
      if (this.live) this.live.setBypass(this.bypass);
      return;
    }
    if (message.type === 'gain') {
      if (Number.isFinite(message.value)) {
        this.gain = clampGain(message.value);
        if (this.live) this.live.setGain(this.gain);
      }
      return;
    }
    if (message.type === 'attenuation') {
      if (Number.isFinite(message.value)) {
        this.maxAttenuationDb = clampMaxAttenuationDb(message.value);
        if (this.live) this.live.setMaxAttenuationDb(this.maxAttenuationDb);
      }
    }
  }

  async start() {
    try {
      let instance = null;
      let source = 'page';
      if (typeof fetch === 'function' && typeof WebAssembly.instantiateStreaming === 'function') {
        const response = await fetch(this.wasmUrl);
        if (!response.ok) {
          throw new Error(`Could not load the engine (HTTP ${response.status}).`);
        }
        const result = await WebAssembly.instantiateStreaming(response, {});
        instance = result.instance || result;
        source = 'worklet-fetch';
      } else {
        if (!this.wasmUrl) throw new Error('Could not resolve the engine URL.');
        const bytes = await this.requestWasm();
        const result = await WebAssembly.instantiate(bytes, {});
        instance = result.instance || result;
      }

      this.engine = new VoiceEngine(instance);
      this.live = new LiveDenoiser(this.engine, {
        inputRate: sampleRate,
        outputRate: sampleRate,
        gain: this.gain,
        maxAttenuationDb: this.maxAttenuationDb,
        bypass: this.bypass,
        captureSpectrum: true,
        onSpectrum: (streaming) => this.captureSpectrum(streaming),
      });
      this.latencySamples = this.live.latencySamples;

      this.port.postMessage({
        type: 'ready',
        contextSampleRate: sampleRate,
        wasmSource: source,
        backend: this.engine.backendName || 'scalar',
        modelBytes: this.engine.modelImageBytes,
        latencySamples: this.latencySamples,
      });
    } catch (error) {
      this.failed = true;
      this.port.postMessage({
        type: 'error',
        stage: 'engine',
        message: error && error.message ? error.message : 'Could not start the engine.',
      });
    }
  }

  requestWasm() {
    return new Promise((resolve, reject) => {
      this.wasmWaiter = { resolve, reject };
      this.port.postMessage({ type: 'need-wasm', url: this.wasmUrl });
    });
  }

  /**
   * Quantise one aligned spectrum pair into the batch.  Runs on the audio
   * thread, but only copies 2 × 513 bytes and does no rendering.
   */
  captureSpectrum(streaming) {
    const capture = streaming.capture || streaming;
    const column = this.specCount;
    const base = column * SPEC_BINS;
    writeQuantizedColumn(capture.spectrumInput, this.specBatch, base);
    writeQuantizedColumn(
      capture.spectrumOutput,
      this.specBatch,
      SPEC_BATCH_COLUMNS * SPEC_BINS + base
    );
    this.specCount += 1;
    if (this.specCount >= SPEC_BATCH_COLUMNS) this.flushSpectrum();
  }

  /** Send the batched columns to the page; the main thread only paints. */
  flushSpectrum() {
    if (this.specCount === 0) return;
    const count = this.specCount;
    this.specCount = 0;
    const bins = SPEC_BINS;
    // The batch stores the output plane at a fixed maximum-batch offset so
    // capture is a plain copy.  A report can flush a partial batch, so move the
    // output plane down to the count-based offset the packed format expects.
    if (count < SPEC_BATCH_COLUMNS) {
      this.specBatch.copyWithin(
        count * bins,
        SPEC_BATCH_COLUMNS * bins,
        SPEC_BATCH_COLUMNS * bins + count * bins
      );
    }
    const packed = this.specBatch.slice(0, count * bins * 2);
    this.port.postMessage(
      {
        type: 'spectrum',
        count,
        bins,
        frames: this.live ? this.live.frames : 0,
        columns: packed,
      },
      [packed.buffer]
    );
  }

  process(inputs, outputs) {
    const output = outputs[0] && outputs[0][0];
    if (!output) return true;
    const input = inputs[0] && inputs[0][0];

    if (!input) {
      output.fill(0);
      return true;
    }
    if (!this.live || this.failed) {
      // Stay silent while the engine loads rather than passing the microphone
      // through unprocessed; a failed worklet is reported to the page.
      output.fill(0);
      return true;
    }

    try {
      this.live.processBlock(input, output);
    } catch (error) {
      this.failed = true;
      output.fill(0);
      this.port.postMessage({
        type: 'error',
        stage: 'process',
        message: error && error.message ? error.message : 'Live processing failed.',
      });
      return true;
    }

    let inSquares = 0;
    let outSquares = 0;
    const length = Math.min(input.length, output.length);
    for (let i = 0; i < length; i++) {
      const inValue = Math.abs(input[i]);
      const outValue = Math.abs(output[i]);
      if (inValue > this.peakIn) this.peakIn = inValue;
      if (outValue > this.peakOut) this.peakOut = outValue;
      inSquares += input[i] * input[i];
      outSquares += output[i] * output[i];
    }
    this.inSquares += inSquares;
    this.outSquares += outSquares;
    this.levelSamples += length;
    this.levelBlocks += 1;

    if (this.levelBlocks >= this.reportEvery) this.report();
    return true;
  }

  report() {
    const frames = this.live.frames;
    const hopMs = this.live.streaming.hopMsTotal;
    const frameDelta = frames - this.lastFrames;
    const msDelta = hopMs - this.lastHopMs;
    if (frameDelta > 0) {
      this.frameMs = msDelta / frameDelta;
      this.rtf = this.frameMs / ((HOP_SIZE / SAMPLE_RATE) * 1000);
      this.lastFrames = frames;
      this.lastHopMs = hopMs;
    }
    if (this.levelSamples > 0) {
      const inRms = Math.sqrt(this.inSquares / this.levelSamples);
      const outRms = Math.sqrt(this.outSquares / this.levelSamples);
      this.levelIn = toDb(inRms);
      this.levelOut = toDb(outRms);
    }
    this.inSquares = 0;
    this.outSquares = 0;
    this.levelSamples = 0;
    this.levelBlocks = 0;

    this.flushSpectrum();

    this.port.postMessage({
      type: 'stats',
      frames,
      frameMs: this.frameMs,
      rtf: this.rtf,
      levelIn: this.levelIn,
      levelOut: this.levelOut,
      peakIn: this.peakIn,
      peakOut: this.peakOut,
      contextSampleRate: sampleRate,
      bypass: this.bypass,
      gain: this.gain,
      maxAttenuationDb: this.maxAttenuationDb,
      latencySamples: this.latencySamples,
    });
    this.peakIn = 0;
    this.peakOut = 0;
  }
}

registerProcessor('cleanfield-live', LiveDenoiseProcessor);
