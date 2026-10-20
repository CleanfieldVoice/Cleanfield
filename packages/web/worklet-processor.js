/**
 * CleanField Voice 1.0 — live AudioWorklet processor.
 *
 * Runs on the audio render thread.  It receives microphone audio at the
 * AudioContext rate, denoises it hop by hop through the shared `LiveDenoiser`,
 * and writes the result to the output (so the user hears it live).
 *
 * The wasm engine is instantiated *inside* the worklet from `cf_wasm.wasm`.
 * AudioWorkletGlobalScope has no `fetch` in current browsers, so the worklet
 * first resolves the wasm URL from its own module URL and asks the page for the
 * bytes (the page fetches that exact URL and transfers the buffer).  In a
 * future browser that exposes `fetch` on the worklet global, the worklet will
 * use `WebAssembly.instantiateStreaming(fetch(url))` directly.
 */

import { HOP_SIZE, LATENCY_SAMPLES, SAMPLE_RATE, VoiceEngine } from './dsp.js';
import { LiveDenoiser } from './live-dsp.js';

const WASM_FILE = 'cf_wasm.wasm';

function toDb(value) {
  return value > 0 ? Math.max(-100, 20 * Math.log10(value)) : -100;
}

function workletWasmUrl(fallback) {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.url) {
      return import.meta.url.slice(0, import.meta.url.lastIndexOf('/') + 1) + WASM_FILE;
    }
  } catch {
    // Fall through to the URL supplied by the page.
  }
  return fallback || null;
}

class LiveDenoiseProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = (options && options.processorOptions) || {};
    this.wasmUrl = workletWasmUrl(processorOptions.wasmUrl);
    this.engine = null;
    this.live = null;
    this.failed = false;
    this.wasmWaiter = null;
    this.gain = Number.isFinite(processorOptions.gain) ? processorOptions.gain : 1;
    this.bypass = Boolean(processorOptions.bypass);
    this.latencySamples = LATENCY_SAMPLES;
    this.frames = 0;
    this.lastFrames = 0;
    this.lastHopMs = 0;
    this.frameMs = 0;
    this.rtf = 0;
    this.levelBlocks = 0;
    this.reportEvery = Math.max(1, Math.round((0.1 * sampleRate) / 128));
    this.inSquares = 0;
    this.outSquares = 0;
    this.peakIn = 0;
    this.peakOut = 0;
    this.levelSamples = 0;
    this.levelIn = -100;
    this.levelOut = -100;

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
        waiter.reject(new Error(message.message || 'Could not load the denoising engine.'));
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
        this.gain = message.value;
        if (this.live) this.live.setGain(this.gain);
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
          throw new Error(`Could not load the denoising engine (HTTP ${response.status}).`);
        }
        const result = await WebAssembly.instantiateStreaming(response, {});
        instance = result.instance || result;
        source = 'worklet-fetch';
      } else {
        if (!this.wasmUrl) throw new Error('Could not resolve the denoising engine URL.');
        const bytes = await this.requestWasm();
        const result = await WebAssembly.instantiate(bytes, {});
        instance = result.instance || result;
      }

      this.engine = new VoiceEngine(instance);
      this.live = new LiveDenoiser(this.engine, {
        inputRate: sampleRate,
        outputRate: sampleRate,
        gain: this.gain,
        bypass: this.bypass,
      });
      this.latencySamples = this.live.latencySamples;
      this.port.postMessage({
        type: 'ready',
        contextSampleRate: sampleRate,
        wasmSource: source,
        backend: this.engine.backendName,
        modelBytes: this.engine.modelImageBytes,
        latencySamples: this.latencySamples,
      });
    } catch (error) {
      this.failed = true;
      this.port.postMessage({
        type: 'error',
        stage: 'engine',
        message: error && error.message ? error.message : 'Could not start the denoising engine.',
      });
    }
  }

  requestWasm() {
    return new Promise((resolve, reject) => {
      this.wasmWaiter = { resolve, reject };
      this.port.postMessage({ type: 'need-wasm', url: this.wasmUrl });
    });
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
        message: error && error.message ? error.message : 'Live denoising failed.',
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
      latencySamples: this.latencySamples,
    });
    this.peakIn = 0;
    this.peakOut = 0;
  }
}

registerProcessor('cleanfield-live', LiveDenoiseProcessor);
