/**
 * CleanField Voice 1.0 — Web Worker entry point.
 *
 * The worker owns one wasm instance and one voice-model handle for the lifetime
 * of the page.  Decoding and resampling stay on the main thread (Web Audio's
 * OfflineAudioContext is not available in workers); the DSP pipeline and WAV
 * encoding run here so the UI stays responsive.
 *
 * One request type: `process` — denoise one 16 kHz mono recording with the
 * voice engine.  Maximum attenuation (the mask floor, a model-level parameter)
 * is read once per request; output gain (post-processing) is applied after the
 * pipeline.
 */

import { loadVoiceEngine, processPcm, encodeWav, SAMPLE_RATE } from './dsp.js';
import { FILE_HISTORY, SPEC_BINS, SpectrogramBuffer } from './spectrogram-core.js';

let enginePromise = null;
let wasmBytes = 0;

function fetchWasm() {
  const url = new URL('./cf_wasm.wasm', import.meta.url);
  return fetch(url).then((response) => {
    if (!response.ok) {
      throw new Error(`Could not load the engine (HTTP ${response.status}).`);
    }
    return response.arrayBuffer();
  });
}

async function getEngine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      let bytes;
      try {
        bytes = await fetchWasm();
      } catch {
        throw new Error(
          'Could not load the denoising engine. Make sure cf_wasm.wasm is in the same folder as the page.'
        );
      }
      wasmBytes = bytes.byteLength;
      return loadVoiceEngine(bytes);
    })().catch((error) => {
      enginePromise = null;
      throw error;
    });
  }
  return enginePromise;
}

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

function applyGain(samples, gain) {
  if (!Number.isFinite(gain) || gain === 1) return;
  const scale = Math.min(8, Math.max(0, gain));
  for (let i = 0; i < samples.length; i++) samples[i] *= scale;
}

async function handleProcess(message) {
  const id = message.id;
  try {
    const engine = await getEngine();
    if (!(message.samples instanceof ArrayBuffer)) throw new Error('The audio data format is invalid.');
    const samples = new Float32Array(message.samples);
    if (message.sampleRate !== SAMPLE_RATE) throw new Error('The audio sample rate is not 16 kHz.');

    // Spectrogram columns are accumulated as the offline DSP advances.  Batches
    // are posted with the progress messages so the page can paint while the
    // worker is still processing; the final `done` message carries all columns.
    const spectrogram = new SpectrogramBuffer(FILE_HISTORY);
    let sentColumns = 0;
    const flushSpectrum = () => {
      if (spectrogram.length <= sentColumns) return;
      const count = spectrogram.length - sentColumns;
      const packed = spectrogram.pack(sentColumns, spectrogram.length);
      sentColumns = spectrogram.length;
      post({ type: 'spectrum', id, count, bins: SPEC_BINS, columns: packed }, [packed.buffer]);
    };

    let lastReported = -1;
    const result = processPcm(samples, engine, {
      maxAttenuationDb: message.maxAttenuationDb,
      maxSpectrumColumns: FILE_HISTORY,
      onSpectrum: (index, inputDb, outputDb) => {
        spectrogram.pushDb(inputDb, outputDb);
        if (spectrogram.length - sentColumns >= 32) flushSpectrum();
      },
      onProgress: (done, total) => {
        const percent = Math.floor((done / total) * 100);
        if (percent !== lastReported) {
          lastReported = percent;
          flushSpectrum();
          post({ type: 'progress', id, done, total, percent });
        }
      },
    });
    flushSpectrum();
    applyGain(result.samples, message.gain);

    const wav = encodeWav(result.samples, SAMPLE_RATE);
    const finalSpectrum = spectrogram.pack(0, spectrogram.length);
    post(
      {
        type: 'done',
        id,
        wav,
        frames: result.frames,
        processingMs: result.processingMs,
        rtf: result.rtf,
        durationSeconds: result.durationSeconds,
        rmsBefore: result.rmsBefore,
        rmsAfter: result.rmsAfter,
        peakBefore: result.peakBefore,
        peakAfter: result.peakAfter,
        clippedSamples: result.clippedSamples,
        wasmBytes,
        spectrogram: {
          columns: finalSpectrum,
          count: spectrogram.length,
          bins: SPEC_BINS,
        },
      },
      [wav, finalSpectrum.buffer]
    );
  } catch (error) {
    post({
      type: 'error',
      id,
      message: error && error.message ? error.message : 'An unknown error occurred while processing the audio.',
    });
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  if (!message || typeof message !== 'object') return;
  if (message.type === 'warmup') {
    try {
      const engine = await getEngine();
      post({
        type: 'ready',
        wasmBytes,
        modelBytes: engine.modelImageBytes,
      });
    } catch (error) {
      post({
        type: 'error',
        id: message.id,
        message: error && error.message ? error.message : 'Could not prepare the denoising engine.',
      });
    }
    return;
  }
  if (message.type === 'process') {
    await handleProcess(message);
  }
};
