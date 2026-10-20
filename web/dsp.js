/**
 * CleanField Voice 1.0 — client-side DSP pipeline.
 *
 * This module is plain ES module JavaScript with no dependencies.  It runs
 * unchanged in a browser worker, on the browser main thread (the fallback
 * path) and in Node, so the same STFT/OLA code can be exercised without a
 * browser.
 *
 * Contract:
 *   16 kHz mono, 1024-point FFT, 768-sample periodic Hann window,
 *   256-sample hop, feature ln(|X[k]|^2 + 1e-20) for k = 0..512,
 *   mask[k] = clamp(sqrt(re[k]^2 + im[k]^2) * gain, floor, 1) from the
 *   engine's two 513-value output planes, inverse FFT, overlap-add.  The
 *   floor is the "maximum attenuation" control: 0 dB is a passthrough and
 *   −80 dB is the full model mask.
 *
 * The fixed analysis latency (768 + 2 * 256 = 1280 samples) is removed from
 * the result, so the returned array is time-aligned with the input.
 *
 * The spectrogram panels are fed by transforms of the real audio path.  The
 * input panel is the windowed analysis spectrum that enters the model; the
 * output panel is a windowed transform of the 768 most recently emitted output
 * samples, taken after the synthesis window and overlap-add.  Capture only
 * computes extra values when it is enabled and never changes an output sample.
 */

import { SPEC_BINS, fillDbColumn } from './spectrogram-core.js';

export const SAMPLE_RATE = 16000;
export const FFT_SIZE = 1024;
export const WINDOW_SIZE = 768;
export const HOP_SIZE = 256;
export const BIN_COUNT = 513;
export const LOOKAHEAD_HOPS = 2;
export const LATENCY_SAMPLES = WINDOW_SIZE + LOOKAHEAD_HOPS * HOP_SIZE;
/**
 * How many emitted hops after an input frame its output window is complete.
 * The input frame's 768-sample analysis window is the same span of audio as the
 * 768 most recently emitted output samples at that point, so a column can show
 * both panels for the same instant.  The output hop trails its analysis frame
 * by `LOOKAHEAD_HOPS`, and the rolling output window trails the emitted hop by
 * `WINDOW_SIZE / HOP_SIZE - 1` hops.
 */
export const SPECTRUM_ALIGN_HOPS =
  WINDOW_SIZE / HOP_SIZE - 1 + LOOKAHEAD_HOPS; // 4
/** Ring slots needed to hold the input spectrum until its output window is ready. */
export const SPECTRUM_HISTORY = SPECTRUM_ALIGN_HOPS + 1; // 5
export const MODEL_INPUT_LEN = 513;
export const MODEL_OUTPUT_LEN = 1026;
export const MAX_DURATION_SECONDS = 30 * 60;

/**
 * The desktop engine was trained with power computed in int16 units and folds
 * 32768^2 into this float32 multiplier.  It is 0.99999726, i.e. 2.7e-6 below
 * unity in log-power, but we keep it so the wasm call receives the same
 * features as the reference pipeline.
 */
export const FEATURE_POWER_SCALE = Math.fround(
  Math.fround(9.313200211025219e-10) * 1073741824
);
export const EPSILON = Math.fround(1e-20);
export const MASK_GAIN = 1;
/**
 * Deepest mask floor the model path supports: −80 dB, matching the reference
 * file pipeline.  This is the fixed floor that earlier builds hard-coded; it is
 * still exact at the default maximum attenuation of −80 dB.
 */
export const MASK_FLOOR = Math.fround(10 ** (-80 / 20));
/** Deepest maximum attenuation the demo exposes, in dB of noise reduction. */
export const MAX_ATTENUATION_DB = 80;
/** Default maximum attenuation: the full model mask (the historical output). */
export const DEFAULT_MAX_ATTENUATION_DB = -MAX_ATTENUATION_DB;
/** Sum of squared periodic Hann at 75% overlap: 768 * 3/8 / 256 = 9/8. */
export const SYNTHESIS_SCALE = Math.fround(8 / 9);

/**
 * Clamp a maximum-attenuation setting to `[-80, 0]` dB, defaulting to −80 dB
 * for non-numbers.  The value is the mask floor expressed as a maximum
 * reduction: 0 dB leaves the mask at unity (passthrough), −80 dB applies the
 * full learned mask down to −80 dB.
 *
 * @param {unknown} value
 * @returns {number}
 */
export function normalizeMaxAttenuationDb(value) {
  const db = Number(value);
  if (!Number.isFinite(db)) return DEFAULT_MAX_ATTENUATION_DB;
  return db < -MAX_ATTENUATION_DB ? -MAX_ATTENUATION_DB : db > 0 ? 0 : db;
}

/**
 * Convert a maximum-attenuation setting in dB to a mask floor amplitude.
 *
 * `floorDb = -80 * s` in the old 0…1 strength notation, so a percentage-style
 * slider can use `s = -db / 80`.  Both endpoints are exact: 0 dB returns
 * exactly 1 (passthrough) and −80 dB returns the historical `MASK_FLOOR`
 * constant, which keeps the default output bit-identical to the previous build.
 *
 * @param {number} maxAttenuationDb in `[-80, 0]`
 * @returns {number} mask floor in `[1e-4, 1]`
 */
export function maskFloorForDb(maxAttenuationDb) {
  const db = normalizeMaxAttenuationDb(maxAttenuationDb);
  if (db >= 0) return 1;
  if (db <= -MAX_ATTENUATION_DB) return MASK_FLOOR;
  return Math.fround(10 ** (db / 20));
}

/**
 * Apply the mask floor to one learned mask magnitude.
 *
 * The applied mask is `min(max(m, floor), 1)`, evaluated with the same
 * comparison sequence the fixed −80 dB floor used before this control existed.
 * A floor of 1 is exactly unity (passthrough); a floor of `MASK_FLOOR` is the
 * historical full-strength mask.
 *
 * @param {number} magnitude `|Y[k]| * MASK_GAIN`, not yet clamped
 * @param {number} floor from `maskFloorForDb()`
 * @returns {number}
 */
export function applyMaskFloor(magnitude, floor) {
  return magnitude > floor ? (magnitude < 1 ? magnitude : 1) : floor;
}

export const WINDOW = (() => {
  const w = new Float32Array(WINDOW_SIZE);
  for (let i = 0; i < WINDOW_SIZE; i++) {
    w[i] = Math.fround(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / WINDOW_SIZE));
  }
  return w;
})();

/**
 * Scale that turns a windowed-FFT magnitude into full-scale amplitude:
 * a bin-centred full-scale sine has |X| = sum(w) / 2, so multiplying by
 * 2 / sum(w) makes it 0 dBFS.  Both spectrogram panels use this scale, so a
 * full-scale sine reads 0 dBFS in either panel.
 */
export const SPECTRUM_NORM = (() => {
  let sum = 0;
  for (let i = 0; i < WINDOW_SIZE; i++) sum += WINDOW[i];
  return 2 / sum;
})();

const clock =
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? () => performance.now()
    : () => Date.now();

class DemoError extends Error {
  constructor(message) {
    super(message);
    this.name = 'DemoError';
    this.userFacing = true;
  }
}

function fail(message) {
  return new DemoError(message);
}

// ---------------------------------------------------------------------------
// FFT (radix-2, decimation in time, double precision)
// ---------------------------------------------------------------------------

export class FFT {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw fail('FFT length must be a power of two.');
    this.n = n;
    const bits = Math.log2(n);
    this.rev = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>>= 1;
      }
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n >> 1);
    this.sin = new Float64Array(n >> 1);
    for (let i = 0; i < n >> 1; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
  }

  _run(re, im, inverse) {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }

    const cos = this.cos;
    const sin = this.sin;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let j = 0; j < half; j++) {
          const k = j * step;
          const wr = cos[k];
          const wi = inverse ? sin[k] : -sin[k];
          const a = start + j;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }

  forward(re, im) {
    this._run(re, im, false);
  }

  inverse(re, im) {
    this._run(re, im, true);
    const n = this.n;
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

export const FFT_1024 = new FFT(FFT_SIZE);

// ---------------------------------------------------------------------------
// wasm engine
// ---------------------------------------------------------------------------

function wasmFailure(exports, message) {
  const code = typeof exports.cf_last_error === 'function' ? exports.cf_last_error() : 0;
  return fail(code ? `${message} (error code ${code})` : message);
}

/**
 * Instantiate `cf_wasm.wasm` and create the voice model.
 *
 * @param {BufferSource|WebAssembly.Module} wasmSource
 * @param {{modelBytes?: Uint8Array}} [options] use a host-supplied .cfwm
 *        container instead of the embedded one.
 * @returns {Promise<VoiceEngine>}
 */
export async function loadVoiceEngine(wasmSource, options = {}) {
  try {
    let instance;
    if (wasmSource instanceof WebAssembly.Module) {
      instance = new WebAssembly.Instance(wasmSource, {});
    } else {
      const result = await WebAssembly.instantiate(wasmSource, {});
      instance = result.instance ?? result;
    }
    return new VoiceEngine(instance, options);
  } catch (error) {
    // Keep our own validation messages; wrap runtime/compile errors.
    if (error && error.userFacing) throw error;
    throw fail(
      `Could not initialise the denoising engine. Check that this browser supports ` +
        `WebAssembly and SIMD (${error?.message || error}).`
    );
  }
}

export class VoiceEngine {
  constructor(instance, { modelBytes = null } = {}) {
    this.instance = instance;
    this.exports = instance.exports;
    const e = this.exports;
    if (!e.memory || !(e.memory instanceof WebAssembly.Memory)) {
      throw fail('The denoising engine did not export its memory.');
    }
    for (const name of [
      'cf_voice_new',
      'cf_voice_reset',
      'cf_voice_process',
      'cf_voice_free',
      'cf_voice_input_len',
      'cf_voice_output_len',
      'cf_alloc',
      'cf_free',
    ]) {
      if (typeof e[name] !== 'function') throw fail(`The denoising engine did not export ${name}.`);
    }
    if (e.cf_voice_input_len() !== MODEL_INPUT_LEN || e.cf_voice_output_len() !== MODEL_OUTPUT_LEN) {
      throw fail('The denoising engine has an unexpected feature size.');
    }
    if (typeof e.cf_voice_backend_is_scalar === 'function' && e.cf_voice_backend_is_scalar() !== 1) {
      throw fail('The denoising engine is not using the scalar backend.');
    }
    if (typeof e.cf_fma_is_soft === 'function' && e.cf_fma_is_soft() !== 1) {
      throw fail('The denoising engine failed its floating-point self-check.');
    }

    this.modelImageBytes = typeof e.cf_voice_model_image_len === 'function'
      ? e.cf_voice_model_image_len() >>> 0
      : 0;
    this.workspaceBytes = typeof e.cf_voice_workspace_bytes === 'function'
      ? e.cf_voice_workspace_bytes() >>> 0
      : 0;

    let handle;
    if (modelBytes) {
      const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
      if (bytes.byteLength === 0 || bytes.byteLength % 4 !== 0) {
        throw fail('The model file length is invalid.');
      }
      const count = bytes.byteLength >>> 2;
      if (typeof e.cf_alloc_u32 !== 'function') throw fail('The denoising engine does not support an external model.');
      const ptr = e.cf_alloc_u32(count);
      if (!ptr) throw fail('Could not allocate memory for the model.');
      new Uint8Array(e.memory.buffer, ptr, bytes.byteLength).set(bytes);
      handle = e.cf_voice_new(ptr, bytes.byteLength);
      if (!handle) {
        // cf_voice_new takes ownership of the buffer on all paths.
        throw wasmFailure(e, 'Could not load the model file');
      }
      this.modelImageBytes = bytes.byteLength;
    } else {
      handle = e.cf_voice_new(0, 0);
      if (!handle) throw wasmFailure(e, 'Could not create the denoising model');
    }
    this.handle = handle;

    const inBytes = MODEL_INPUT_LEN * 4;
    const outBytes = MODEL_OUTPUT_LEN * 4;
    this.inputPtr = e.cf_alloc(inBytes, 8);
    this.outputPtr = e.cf_alloc(outBytes, 8);
    if (!this.inputPtr || !this.outputPtr) {
      this.close();
      throw fail('Could not allocate memory for the audio frames.');
    }
    this._out = new Float32Array(MODEL_OUTPUT_LEN);
    this._input = new Float32Array(MODEL_INPUT_LEN);

    this.backendName = this._readBackendName();
    this.modelBytes = typeof e.cf_voice_model_bytes === 'function'
      ? e.cf_voice_model_bytes(handle) >>> 0
      : 0;
  }

  _readBackendName() {
    const e = this.exports;
    if (typeof e.cf_voice_backend_name_ptr !== 'function') return '';
    const ptr = e.cf_voice_backend_name_ptr(this.handle);
    const len = e.cf_voice_backend_name_len(this.handle);
    if (!ptr || !len) return '';
    try {
      return new TextDecoder().decode(new Uint8Array(e.memory.buffer, ptr, len));
    } catch {
      return '';
    }
  }

  /** Run one 513-value feature frame; returns a reused 1026-value buffer. */
  process(features) {
    if (!this.handle) throw fail('The denoising engine is closed.');
    if (features.length !== MODEL_INPUT_LEN) throw fail('The feature frame has the wrong length.');
    const e = this.exports;
    new Float32Array(e.memory.buffer, this.inputPtr, MODEL_INPUT_LEN).set(features);
    const rc = e.cf_voice_process(this.handle, this.inputPtr, this.outputPtr);
    if (rc !== 0) throw wasmFailure(e, 'Denoising failed');
    this._out.set(new Float32Array(e.memory.buffer, this.outputPtr, MODEL_OUTPUT_LEN));
    return this._out;
  }

  reset() {
    if (!this.handle) throw fail('The denoising engine is closed.');
    if (this.exports.cf_voice_reset(this.handle) !== 0) {
      throw wasmFailure(this.exports, 'Could not reset the denoising state');
    }
  }

  close() {
    const e = this.exports;
    if (this.handle) {
      e.cf_voice_free(this.handle);
      this.handle = 0;
    }
    if (this.inputPtr) {
      e.cf_free(this.inputPtr, MODEL_INPUT_LEN * 4, 8);
      this.inputPtr = 0;
    }
    if (this.outputPtr) {
      e.cf_free(this.outputPtr, MODEL_OUTPUT_LEN * 4, 8);
      this.outputPtr = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// STFT / mask / overlap-add
// ---------------------------------------------------------------------------

function rms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    sum += v * v;
  }
  return Math.sqrt(sum / samples.length);
}

function peak(samples) {
  let p = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = Math.abs(samples[i]);
    if (v > p) p = v;
  }
  return p;
}

function clippedCount(samples, limit = 1) {
  let n = 0;
  for (let i = 0; i < samples.length; i++) {
    if (Math.abs(samples[i]) > limit) n++;
  }
  return n;
}

/**
 * Denoise a mono float32 array at 16 kHz.
 *
 * @param {Float32Array|number[]} input
 * @param {VoiceEngine} engine
 * @param {{onProgress?: (done: number, total: number) => void,
 *          onSpectrum?: (index: number, inputDb: Float32Array, outputDb: Float32Array) => void,
 *          maxSpectrumColumns?: number, maxAttenuationDb?: number}} [options]
 *        `maxAttenuationDb` is the mask floor in dB in `[-80, 0]` (default
 *        −80 = the full learned mask, 0 = exact passthrough).  The applied
 *        mask is `min(max(|Y[k]| * MASK_GAIN, 10^(dB/20)), 1)`.
 * @returns {{
 *   samples: Float32Array, frames: number, durationSeconds: number,
 *   processingMs: number, rtf: number, rmsBefore: number, rmsAfter: number,
 *   peakBefore: number, peakAfter: number, clippedSamples: number,
 *   latencySamples: number
 * }}
 */
export function processPcm(input, engine, options = {}) {
  if (!(engine instanceof VoiceEngine)) throw fail('The denoising engine is not valid.');
  const source = input instanceof Float32Array ? input : Float32Array.from(input);
  const maxAttenuationDb = normalizeMaxAttenuationDb(options.maxAttenuationDb);
  const maskFloor = maskFloorForDb(maxAttenuationDb);
  const length = source.length;
  if (length === 0) {
    return {
      samples: new Float32Array(0),
      frames: 0,
      durationSeconds: 0,
      processingMs: 0,
      rtf: 0,
      rmsBefore: 0,
      rmsAfter: 0,
      peakBefore: 0,
      peakAfter: 0,
      clippedSamples: 0,
      latencySamples: LATENCY_SAMPLES,
    };
  }
  const durationSeconds = length / SAMPLE_RATE;
  if (durationSeconds > MAX_DURATION_SECONDS) {
    throw fail(`Audio is too long (over ${MAX_DURATION_SECONDS / 60} minutes); trim it first.`);
  }
  for (let i = 0; i < length; i++) {
    if (!Number.isFinite(source[i])) throw fail('The audio contains an invalid sample value.');
  }

  const totalLength = length + LATENCY_SAMPLES;
  const frames = Math.floor(totalLength / HOP_SIZE);
  const output = new Float32Array(totalLength);

  const analysis = new Float32Array(WINDOW_SIZE);
  const overlap = new Float32Array(WINDOW_SIZE);
  const fftRe = new Float64Array(FFT_SIZE);
  const fftIm = new Float64Array(FFT_SIZE);
  // The ring holds the two previous spectra; the current one waits in cur*.
  const specRe = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
  const specIm = [new Float32Array(BIN_COUNT), new Float32Array(BIN_COUNT)];
  const curRe = new Float32Array(BIN_COUNT);
  const curIm = new Float32Array(BIN_COUNT);
  const features = new Float32Array(BIN_COUNT);
  const mask = new Float32Array(BIN_COUNT);

  const progressEvery = Math.max(1, Math.floor(frames / 50));
  const onProgress = typeof options.onProgress === 'function' ? options.onProgress : null;

  // Optional spectrogram capture.  It reuses values the audio path already
  // computes and adds exactly one transform (the output panel FFT), which runs
  // only while capture is enabled.  All buffers are allocated here and reused
  // for every frame, so enabling capture allocates nothing per frame.
  const onSpectrum = typeof options.onSpectrum === 'function' ? options.onSpectrum : null;
  const requestedColumns = Number(options.maxSpectrumColumns);
  const maxColumns =
    Number.isFinite(requestedColumns) && requestedColumns >= 1 ? Math.floor(requestedColumns) : 0;
  const spectrumStride = maxColumns > 0 ? Math.max(1, Math.ceil(frames / maxColumns)) : 1;
  const spectrumInput = onSpectrum ? new Float32Array(SPEC_BINS) : null;
  const spectrumOutput = onSpectrum ? new Float32Array(SPEC_BINS) : null;
  // Columns are averaged over `spectrumStride` frames instead of dropping frames,
  // otherwise the painted time axis leaves empty columns (visible as dark bands).
  const spectrumAccIn = onSpectrum ? new Float64Array(SPEC_BINS) : null;
  const spectrumAccOut = onSpectrum ? new Float64Array(SPEC_BINS) : null;
  let spectrumPending = 0;
  // Input spectra wait here until the output window covering the same audio
  // span has been emitted.  One slot per hop of look-back, plus the slot being
  // written this frame.
  const inputHistory = onSpectrum
    ? Array.from({ length: SPECTRUM_HISTORY }, () => new Float32Array(SPEC_BINS))
    : null;
  // The 768 most recently emitted output samples, oldest at outputRingWrite.
  const outputRing = onSpectrum ? new Float32Array(WINDOW_SIZE) : null;
  let outputRingWrite = 0;

  engine.reset();
  const started = clock();

  for (let frame = 0; frame < frames; frame++) {
    const inputBase = frame * HOP_SIZE;

    // Shift the 768-sample analysis window and append one hop.
    analysis.copyWithin(0, HOP_SIZE);
    for (let j = 0; j < HOP_SIZE; j++) {
      const index = inputBase + j;
      analysis[WINDOW_SIZE - HOP_SIZE + j] = index < length ? source[index] : 0;
    }

    // Forward transform of the windowed 768 samples, zero-padded to 1024.
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

    // Input panel: the windowed analysis spectrum that enters the model.  The
    // column is emitted later, when the matching output window is available.
    if (onSpectrum) {
      fillDbColumn(
        curRe,
        curIm,
        SPEC_BINS,
        SPECTRUM_NORM,
        inputHistory[frame % SPECTRUM_HISTORY]
      );
    }

    // ln(power + epsilon) features, float32 at every rounding boundary.
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
      const power = Math.fround(Math.fround(Math.fround(re * re) + Math.fround(im * im)) + EPSILON);
      const magnitude = Math.fround(Math.sqrt(power));
      const m = Math.fround(magnitude * MASK_GAIN);
      // Apply the maximum-attenuation floor.  A −80 dB floor is exactly the
      // historical fixed floor; a 0 dB floor is exactly unity.
      mask[k] = applyMaskFloor(m, maskFloor);
    }

    // The mask from this frame is applied to the spectrum from two hops ago.
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

    // The current spectra become the look-ahead history for later frames.
    targetRe.set(curRe);
    targetIm.set(curIm);

    const outputBase = HOP_SIZE + frame * HOP_SIZE;
    const available = Math.min(HOP_SIZE, totalLength - outputBase);
    for (let i = 0; i < available; i++) output[outputBase + i] = overlap[i];

    if (onSpectrum) {
      // Remember the hop that is emitted this frame.  The ring always holds
      // the last 768 output samples, oldest first.
      const ring = outputRing;
      let write = outputRingWrite;
      for (let i = 0; i < HOP_SIZE; i++) {
        ring[write] = overlap[i];
        write += 1;
        if (write === WINDOW_SIZE) write = 0;
      }
      outputRingWrite = write;

      if (frame >= SPECTRUM_ALIGN_HOPS) {
        const column = frame - SPECTRUM_ALIGN_HOPS;

        // Output panel: window and transform the real emitted waveform.  This
        // is the overlap-add output after the synthesis window, not the masked
        // analysis spectrum.
        fftRe.fill(0);
        fftIm.fill(0);
        let read = outputRingWrite;
        for (let i = 0; i < WINDOW_SIZE; i++) {
          fftRe[i] = Math.fround(ring[read] * WINDOW[i]);
          read += 1;
          if (read === WINDOW_SIZE) read = 0;
        }
        FFT_1024.forward(fftRe, fftIm);
        fillDbColumn(fftRe, fftIm, SPEC_BINS, SPECTRUM_NORM, spectrumOutput);

        // Input panel: the frame whose analysis window covers the same audio
        // span as those 768 output samples.
        spectrumInput.set(inputHistory[(frame - SPECTRUM_ALIGN_HOPS) % SPECTRUM_HISTORY]);

        for (let k = 0; k < SPEC_BINS; k++) {
          spectrumAccIn[k] += spectrumInput[k];
          spectrumAccOut[k] += spectrumOutput[k];
        }
        spectrumPending += 1;
        if (spectrumPending >= spectrumStride) {
          const inv = 1 / spectrumPending;
          for (let k = 0; k < SPEC_BINS; k++) {
            spectrumInput[k] = Math.fround(spectrumAccIn[k] * inv);
            spectrumOutput[k] = Math.fround(spectrumAccOut[k] * inv);
            spectrumAccIn[k] = 0;
            spectrumAccOut[k] = 0;
          }
          onSpectrum(column, spectrumInput, spectrumOutput);
          spectrumPending = 0;
        }
      }
    }

    overlap.copyWithin(0, HOP_SIZE);
    overlap.fill(0, WINDOW_SIZE - HOP_SIZE);

    if (onProgress && ((frame + 1) % progressEvery === 0 || frame + 1 === frames)) {
      onProgress(frame + 1, frames);
    }
  }

  const processingMs = clock() - started;
  const cleaned = output.slice(LATENCY_SAMPLES, LATENCY_SAMPLES + length);
  return {
    samples: cleaned,
    frames,
    durationSeconds,
    processingMs,
    rtf: durationSeconds > 0 ? processingMs / 1000 / durationSeconds : 0,
    rmsBefore: rms(source),
    rmsAfter: rms(cleaned),
    peakBefore: peak(source),
    peakAfter: peak(cleaned),
    clippedSamples: clippedCount(cleaned),
    latencySamples: LATENCY_SAMPLES,
  };
}

// ---------------------------------------------------------------------------
// WAV encoding and browser decoding
// ---------------------------------------------------------------------------

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

/**
 * Encode mono float samples as 16-bit PCM WAV.
 * @param {Float32Array} samples
 * @param {number} [sampleRate]
 * @returns {ArrayBuffer}
 */
export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i++) {
    const value = samples[i];
    const finite = Number.isFinite(value) ? value : 0;
    const clamped = finite < -1 ? -1 : finite > 1 ? 1 : finite;
    let quantized = Math.round(clamped * 32768);
    if (quantized > 32767) quantized = 32767;
    if (quantized < -32768) quantized = -32768;
    view.setInt16(44 + i * 2, quantized, true);
  }
  return buffer;
}

function getOfflineAudioContext() {
  if (typeof window === 'undefined') return null;
  return window.OfflineAudioContext || window.webkitOfflineAudioContext || null;
}

/**
 * Decode a File/Blob with the browser's Web Audio decoder and return mono
 * float32 samples at 16 kHz.  The decode context itself runs at 16 kHz, so the
 * browser performs the sample-rate conversion; an OfflineAudioContext render
 * is the explicit fallback if a browser returns a different rate.
 *
 * Browser-only: `OfflineAudioContext` is not available in workers.
 *
 * @param {Blob} file
 * @returns {Promise<Float32Array>}
 */
export async function decodeAudioFile(file) {
  const Ctx = getOfflineAudioContext();
  if (!Ctx || typeof file?.arrayBuffer !== 'function') {
    throw fail('This browser cannot decode audio. Please use a current browser.');
  }
  if (typeof location !== 'undefined' && location.protocol === 'file:') {
    throw fail('Open this page through a static web server; browsers do not allow loading WebAssembly from a local file.');
  }
  const bytes = await file.arrayBuffer();
  let decoded;
  try {
    // Decoding in a 16 kHz offline context resamples during decode.
    const decodeContext = new Ctx(1, 1, SAMPLE_RATE);
    decoded = await decodeContext.decodeAudioData(bytes);
  } catch {
    throw fail('Could not decode this audio file. Try another file, or convert it to WAV.');
  }
  if (!decoded || decoded.length === 0) throw fail('The audio file is empty.');

  if (Math.abs(decoded.sampleRate - SAMPLE_RATE) < 1e-6) {
    return mixToMono(decoded);
  }

  // Fallback: render the decoded buffer through a 16 kHz mono context.
  const targetLength = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE));
  const resampleContext = new Ctx(1, targetLength, SAMPLE_RATE);
  const source = resampleContext.createBufferSource();
  source.buffer = decoded;
  source.connect(resampleContext.destination);
  source.start(0);
  const rendered = await resampleContext.startRendering();
  return mixToMono(rendered);
}

function mixToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  if (channels === 1) {
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }
  for (let c = 0; c < channels; c++) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += data[i];
  }
  const scale = 1 / channels;
  for (let i = 0; i < length; i++) mono[i] *= scale;
  return mono;
}

/** Format seconds as m:ss.s for the UI. */
export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const total = Math.round(seconds * 10) / 10;
  const minutes = Math.floor(total / 60);
  const rest = total - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, '0')}`;
}
