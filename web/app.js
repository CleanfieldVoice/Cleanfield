/**
 * CleanField Voice 1.0 — page controller.
 *
 * One signal path, two inputs:
 *
 *   Live — microphone -> AudioWorklet -> wasm voice engine -> speakers.
 *   File — decode -> worker (or main-thread fallback) -> 16-bit PCM WAV.
 *
 * The wasm module embeds the voice model, so the page has no second model
 * download.  Maximum attenuation (the mask floor, a model-level parameter) and
 * output gain (a post-processing parameter) are the only user parameters; both
 * are forwarded to a running worklet without restarting audio, and the file
 * path reads them once when Process is pressed.
 */

import {
  SAMPLE_RATE,
  decodeAudioFile,
  encodeWav,
  formatDuration,
  loadVoiceEngine,
  processPcm,
} from './dsp.js';
import { FILE_HISTORY, SPEC_BINS, SpectrogramBuffer } from './spectrogram-core.js';
import { SpectrogramView } from './spectrogram-view.js';

const MIC_CONSTRAINTS = {
  audio: {
    channelCount: 1,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  },
};

const WASM_URL = new URL('./cf_wasm.wasm', import.meta.url).href;

const ui = {
  card: document.getElementById('card'),
  tabLive: document.getElementById('tabLive'),
  tabFile: document.getElementById('tabFile'),
  livePanel: document.getElementById('livePanel'),
  filePanel: document.getElementById('filePanel'),
  startButton: document.getElementById('startButton'),
  liveStatus: document.getElementById('liveStatus'),
  liveError: document.getElementById('liveError'),
  liveHint: document.getElementById('liveHint'),
  inputMeter: document.getElementById('inputMeter'),
  outputMeter: document.getElementById('outputMeter'),
  inputDb: document.getElementById('inputDb'),
  outputDb: document.getElementById('outputDb'),
  denoisedButton: document.getElementById('denoisedButton'),
  rawButton: document.getElementById('rawButton'),
  frameTiming: document.getElementById('frameTiming'),
  rtfValue: document.getElementById('rtfValue'),
  frameCount: document.getElementById('frameCount'),
  sampleRateValue: document.getElementById('sampleRateValue'),
  latencyNote: document.getElementById('latencyNote'),
  attenuationSlider: document.getElementById('attenuationSlider'),
  attenuationValue: document.getElementById('attenuationValue'),
  gainSlider: document.getElementById('gainSlider'),
  gainValue: document.getElementById('gainValue'),
  dropZone: document.getElementById('dropZone'),
  dropTitle: document.getElementById('dropTitle'),
  chooseButton: document.getElementById('chooseButton'),
  fileInput: document.getElementById('fileInput'),
  fileName: document.getElementById('fileName'),
  processButton: document.getElementById('processButton'),
  status: document.getElementById('status'),
  progress: document.getElementById('progress'),
  error: document.getElementById('error'),
  results: document.getElementById('results'),
  beforeAudio: document.getElementById('beforeAudio'),
  afterAudio: document.getElementById('afterAudio'),
  beforeLabel: document.getElementById('beforeLabel'),
  beforeNote: document.getElementById('beforeNote'),
  afterLabel: document.getElementById('afterLabel'),
  afterNote: document.getElementById('afterNote'),
  downloadLink: document.getElementById('downloadLink'),
  stats: document.getElementById('stats'),
  specInput: document.getElementById('specInput'),
  specOutput: document.getElementById('specOutput'),
  specInputEmpty: document.getElementById('specInputEmpty'),
  specOutputEmpty: document.getElementById('specOutputEmpty'),
  specFreeze: document.getElementById('specFreeze'),
  specClear: document.getElementById('specClear'),
  specRamp: document.getElementById('specRamp'),
  specSpan: document.getElementById('specSpan'),
  specXStart: document.getElementById('specXStart'),
  specXEnd: document.getElementById('specXEnd'),
};

const state = {
  tab: 'live',
  live: null,
  liveStarting: false,
  liveAttempt: 0,
  liveBypass: false,
  gainDb: 0,
  maxAttenuationDb: -80,
  liveStats: null,
  file: null,
  busy: false,
  worker: null,
  workerBroken: false,
  warmed: false,
  nextId: 1,
  pending: null,
  beforeUrl: null,
  afterUrl: null,
  mainEnginePromise: null,
};

const spectrogram = new SpectrogramView({
  inputCanvas: ui.specInput,
  outputCanvas: ui.specOutput,
  inputEmpty: ui.specInputEmpty,
  outputEmpty: ui.specOutputEmpty,
  freezeButton: ui.specFreeze,
  clearButton: ui.specClear,
  ramp: ui.specRamp,
  span: ui.specSpan,
  xStart: ui.specXStart,
  xEnd: ui.specXEnd,
});

// ---------------------------------------------------------------------------
// tabs
// ---------------------------------------------------------------------------

function selectTab(name) {
  const next = name === 'file' ? 'file' : 'live';
  if (next === 'file' && (state.live || state.liveStarting)) {
    void stopLive();
  }
  state.tab = next;
  const live = next === 'live';
  ui.tabLive.classList.toggle('is-active', live);
  ui.tabFile.classList.toggle('is-active', !live);
  ui.tabLive.setAttribute('aria-selected', live ? 'true' : 'false');
  ui.tabFile.setAttribute('aria-selected', live ? 'false' : 'true');
  ui.tabLive.tabIndex = live ? 0 : -1;
  ui.tabFile.tabIndex = live ? -1 : 0;
  ui.livePanel.hidden = !live;
  ui.filePanel.hidden = live;
  updateLatencyNote();
}

function handleTabKey(event) {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
  event.preventDefault();
  const next = state.tab === 'live' ? 'file' : 'live';
  selectTab(next);
  (next === 'live' ? ui.tabLive : ui.tabFile).focus();
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

function setStatus(text) {
  ui.status.textContent = text;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.textContent = '';
  ui.error.hidden = true;
}

function showProgress(percent) {
  ui.progress.hidden = false;
  if (percent === null || percent === undefined) {
    ui.progress.removeAttribute('value');
  } else {
    ui.progress.value = Math.max(0, Math.min(100, percent));
  }
}

function hideProgress() {
  ui.progress.hidden = true;
  ui.progress.value = 0;
}

function formatMs(ms) {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatDb(db) {
  if (!Number.isFinite(db) || db <= -99.5) return '−100 dB';
  const rounded = Math.round(db);
  return `${rounded < 0 ? '−' : ''}${Math.abs(rounded)} dB`;
}

function formatSignedDb(db) {
  return `${db > 0 ? '+' : db < 0 ? '−' : ''}${Math.abs(db)} dB`;
}

function setMeter(element, db) {
  const clamped = Number.isFinite(db) ? Math.max(-60, Math.min(0, db)) : -100;
  const percent = ((clamped + 60) / 60) * 100;
  element.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function dbToGain(db) {
  return 10 ** (Number(db) / 20);
}

function updateProcessButton() {
  ui.processButton.disabled = state.busy || !state.file;
  ui.processButton.textContent = state.busy ? 'Processing…' : 'Process';
}

function setBusy(busy) {
  state.busy = busy;
  ui.chooseButton.disabled = busy;
  ui.fileInput.disabled = busy;
  ui.dropZone.classList.toggle('is-busy', busy);
  ui.dropZone.setAttribute('aria-disabled', busy ? 'true' : 'false');
  ui.card.setAttribute('aria-busy', busy ? 'true' : 'false');
  updateProcessButton();
}

// ---------------------------------------------------------------------------
// latency note
// ---------------------------------------------------------------------------

function updateLatencyNote() {
  const session = state.live;
  const rate =
    session && session.ctx && Number.isFinite(session.ctx.sampleRate)
      ? session.ctx.sampleRate
      : SAMPLE_RATE;
  const resampled = Math.abs(rate - SAMPLE_RATE) > 1e-6;
  const rateText = `${(rate / 1000).toFixed(1)} kHz`;

  if (state.tab === 'file') {
    ui.latencyNote.textContent =
      'Algorithmic latency: 80 ms — a 768-sample window plus 2 × 256-sample look-ahead ' +
      '(1,280 samples at 16 kHz), removed from the file output.';
    return;
  }
  ui.latencyNote.textContent = resampled
    ? `Algorithmic latency: 80 ms — a 768-sample window plus 2 × 256-sample look-ahead ` +
      `(1,280 samples at 16 kHz). This context runs at ${rateText} and is ` +
      `resampled inside the worklet, which adds up to one 256-sample hop (16 ms) of buffering.`
    : 'Algorithmic latency: 80 ms — a 768-sample window plus 2 × 256-sample look-ahead ' +
      '(1,280 samples at 16 kHz), before the device buffer.';
}

// ---------------------------------------------------------------------------
// live mode
// ---------------------------------------------------------------------------

function liveSupported() {
  return (
    typeof navigator !== 'undefined' &&
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === 'function' &&
    typeof AudioContext !== 'undefined' &&
    typeof AudioWorkletNode !== 'undefined'
  );
}

function setLiveStatus(text) {
  ui.liveStatus.textContent = text;
}

function showLiveError(message) {
  ui.liveError.textContent = message;
  ui.liveError.hidden = false;
}

function clearLiveError() {
  ui.liveError.textContent = '';
  ui.liveError.hidden = true;
}

function resetLiveMeters() {
  setMeter(ui.inputMeter, -100);
  setMeter(ui.outputMeter, -100);
  ui.inputDb.textContent = formatDb(-100);
  ui.outputDb.textContent = formatDb(-100);
  ui.frameTiming.textContent = '— ms / frame';
  ui.rtfValue.textContent = 'RTF —';
  ui.frameCount.textContent = '0 frames';
  ui.sampleRateValue.textContent = '16 kHz';
  state.liveStats = null;
}

function updateLiveControls() {
  const running = Boolean(state.live);
  const starting = state.liveStarting;
  ui.startButton.disabled = starting;
  ui.startButton.textContent = starting ? 'Starting…' : running ? 'Stop' : 'Start';
  ui.startButton.classList.toggle('is-stop', running);
}

function describeMicError(error) {
  const name = error && error.name ? error.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return (
      'Microphone access was denied. Allow microphone access for this site in your ' +
      'browser settings, then press Start again. You can also use the File tab.'
    );
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No microphone was found. Connect one and press Start again, or use the File tab.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'The microphone is already in use by another application. Close it and try again.';
  }
  if (name === 'OverconstrainedError') {
    return 'This microphone does not support the requested capture settings.';
  }
  return error && error.message
    ? `Could not start live mode: ${error.message}`
    : 'Could not start live mode. Check your microphone and try again.';
}

function failLive(message) {
  showLiveError(message);
  setLiveStatus('Live mode stopped');
  void stopLive({ updateStatus: false, keepError: true });
}

function stopStream(stream) {
  if (!stream) return;
  try {
    stream.getTracks().forEach((track) => track.stop());
  } catch {
    // Ignore.
  }
}

/** Send one parameter message to the running worklet; ignore when idle. */
function postLive(message) {
  const session = state.live;
  if (!session) return;
  try {
    session.node.port.postMessage(message);
  } catch {
    // The node is already gone.
  }
}

function pushLiveParams() {
  postLive({ type: 'attenuation', value: state.maxAttenuationDb });
  postLive({ type: 'gain', value: dbToGain(state.gainDb) });
}

function handleLiveMessage(event) {
  const message = event.data;
  if (!message || typeof message !== 'object') return;
  const session = state.live;
  if (!session) return;

  if (message.type === 'need-wasm') {
    void supplyWasm(message.url || WASM_URL, session);
    return;
  }
  if (message.type === 'spectrum') {
    spectrogram.appendPacked(message.columns, message.count);
    return;
  }
  if (message.type === 'ready') {
    session.ready = true;
    session.latencySamples = message.latencySamples;
    const rate = message.contextSampleRate || session.ctx.sampleRate;
    const resampled = Math.abs(rate - SAMPLE_RATE) > 1e-6;
    ui.sampleRateValue.textContent = `${(rate / 1000).toFixed(1)} kHz${resampled ? ' · resampled' : ''}`;
    updateLatencyNote();
    setLiveStatus(
      `Live · ${(rate / 1000).toFixed(1)} kHz` + (resampled ? ' (resampled)' : '')
    );
    // Re-apply the current values, including changes made while the engine was
    // still loading.
    pushLiveParams();
    updateLiveControls();
    return;
  }
  if (message.type === 'stats') {
    session.frames = message.frames;
    session.frameMs = message.frameMs;
    session.rtf = message.rtf;
    session.levelIn = message.levelIn;
    session.levelOut = message.levelOut;
    session.peakIn = message.peakIn;
    session.peakOut = message.peakOut;
    state.liveStats = message;
    const frameMs = Number.isFinite(message.frameMs) ? message.frameMs : 0;
    const rtf = Number.isFinite(message.rtf) ? message.rtf : 0;
    ui.frameTiming.textContent = `${frameMs.toFixed(1)} ms / frame`;
    ui.rtfValue.textContent = `RTF ${rtf.toFixed(2)}`;
    ui.frameCount.textContent = `${Number(message.frames || 0).toLocaleString('en-US')} frames`;
    ui.inputDb.textContent = formatDb(message.levelIn);
    ui.outputDb.textContent = formatDb(message.levelOut);
    setMeter(ui.inputMeter, message.levelIn);
    setMeter(ui.outputMeter, message.levelOut);
    return;
  }
  if (message.type === 'error') {
    const text = message.message || 'The live processor stopped unexpectedly.';
    if (message.stage === 'engine') {
      failLive(
        `${text} Check that cf_wasm.wasm is next to worklet-processor.js, ` +
          'or use the File tab.'
      );
    } else {
      failLive(text);
    }
  }
}

async function supplyWasm(url, session) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength === 0) throw new Error('empty response');
    if (state.live !== session) return;
    session.node.port.postMessage({ type: 'wasm', bytes }, [bytes]);
  } catch (error) {
    const message = `Could not load the denoising engine (${error && error.message ? error.message : 'network error'}).`;
    try {
      session.node.port.postMessage({ type: 'wasm-error', message });
    } catch {
      // The node is already gone.
    }
    failLive(`${message} Use the File tab if this keeps happening.`);
  }
}

async function startLiveSession() {
  state.liveStarting = true;
  updateLiveControls();
  setLiveStatus('Requesting microphone…');
  const attempt = ++state.liveAttempt;

  let stream = null;
  let ctx = null;
  try {
    // Create and resume the context inside the click task so the browser keeps
    // the user activation; awaiting getUserMedia first can leave it suspended.
    ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: 'interactive' });
    if (!ctx.audioWorklet) {
      throw Object.assign(new Error('AudioWorklet is not available in this browser.'), {
        name: 'AudioWorkletUnavailable',
      });
    }
    void ctx.resume();
    stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
    if (attempt !== state.liveAttempt) {
      stopStream(stream);
      if (ctx && ctx.state !== 'closed') {
        try {
          await ctx.close();
        } catch {
          // Ignore.
        }
      }
      return;
    }
    setLiveStatus('Starting audio…');
    await ctx.audioWorklet.addModule(new URL('./worklet-processor.js', import.meta.url));
    if (attempt !== state.liveAttempt) {
      throw Object.assign(new Error('Live start was cancelled.'), { name: 'Cancelled' });
    }
    const node = new AudioWorkletNode(ctx, 'cleanfield-live', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
      channelCountMode: 'explicit',
      processorOptions: {
        bypass: state.liveBypass,
        gain: dbToGain(state.gainDb),
        maxAttenuationDb: state.maxAttenuationDb,
        wasmUrl: WASM_URL,
      },
    });
    const source = ctx.createMediaStreamSource(stream);

    const session = {
      stream,
      ctx,
      node,
      sources: [source],
      ready: false,
    };
    state.live = session;
    state.liveStarting = false;
    spectrogram.begin('live');
    node.port.onmessage = handleLiveMessage;
    node.onprocessorerror = () => {
      failLive('The live audio processor stopped unexpectedly. Press Start to try again.');
    };
    source.connect(node);
    node.connect(ctx.destination);
    await ctx.resume();
    if (attempt !== state.liveAttempt) {
      throw Object.assign(new Error('Live start was cancelled.'), { name: 'Cancelled' });
    }
    setLiveStatus('Loading denoising engine…');
    updateLiveControls();
    resetLiveMeters();
    ui.sampleRateValue.textContent = `${(ctx.sampleRate / 1000).toFixed(1)} kHz`;
  } catch (error) {
    state.live = null;
    state.liveStarting = false;
    updateLiveControls();
    stopStream(stream);
    if (ctx) {
      try {
        await ctx.close();
      } catch {
        // Ignore.
      }
    }
    if (error && error.name === 'AudioWorkletUnavailable') {
      showLiveError(
        'This browser does not expose AudioWorklet, so live mode cannot run. Use the File tab instead.'
      );
      setLiveStatus('Live mode unavailable');
      selectTab('file');
      setStatus('Live mode is unavailable in this browser. You can still process a file.');
      showError(
        'Live mode is unavailable in this browser. You can still denoise an audio file below.'
      );
      return;
    }
    if (error && error.name === 'Cancelled') {
      updateLiveControls();
      return;
    }
    const message = describeMicError(error);
    showLiveError(message);
    setLiveStatus('Could not start');
    if (typeof console !== 'undefined') console.error('[CleanField]', error);
  }
}

function startLive() {
  if (state.live || state.liveStarting) return;
  clearLiveError();

  if (!liveSupported()) {
    const message =
      'Live mode needs a secure context (HTTPS or localhost) plus microphone access ' +
      'and the AudioWorklet API, which this browser does not provide. ' +
      'Use the File tab instead.';
    showLiveError(message);
    setLiveStatus('Live mode unavailable');
    selectTab('file');
    setStatus('Live mode is unavailable in this browser. You can still process a file.');
    showError('Live mode is unavailable in this browser. You can still process a file.');
    return;
  }
  return startLiveSession();
}

function stopLive(options = {}) {
  const session = state.live;
  state.live = null;
  state.liveStarting = false;
  state.liveAttempt += 1;
  const closing = [];
  if (session) {
    try {
      session.node.port.postMessage({ type: 'stop' });
    } catch {
      // Ignore.
    }
    for (const source of session.sources || []) {
      try {
        source.disconnect();
      } catch {
        // Ignore.
      }
    }
    try {
      session.node.disconnect();
    } catch {
      // Ignore.
    }
    stopStream(session.stream);
    if (session.ctx && session.ctx.state !== 'closed') {
      closing.push(
        session.ctx.close().catch(() => {
          // Ignore.
        })
      );
    }
  }
  resetLiveMeters();
  if (options.keepError !== true) clearLiveError();
  if (options.updateStatus !== false) setLiveStatus(session ? 'Stopped' : 'Idle');
  updateLiveControls();
  return Promise.all(closing);
}

function setLiveBypass(bypass) {
  state.liveBypass = Boolean(bypass);
  ui.rawButton.classList.toggle('is-active', state.liveBypass);
  ui.denoisedButton.classList.toggle('is-active', !state.liveBypass);
  ui.rawButton.setAttribute('aria-pressed', state.liveBypass ? 'true' : 'false');
  ui.denoisedButton.setAttribute('aria-pressed', state.liveBypass ? 'false' : 'true');
  postLive({ type: 'bypass', value: state.liveBypass });
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------

function updateGainLabel() {
  ui.gainValue.textContent = formatSignedDb(state.gainDb);
}

function setGainDb(value) {
  const db = Math.max(-12, Math.min(12, Math.round(Number(value))));
  state.gainDb = Number.isFinite(db) ? db : 0;
  ui.gainSlider.value = String(state.gainDb);
  updateGainLabel();
  // Only the output stage changes; the engine, its history and the model are
  // untouched, so a live change needs no restart.
  postLive({ type: 'gain', value: dbToGain(state.gainDb) });
}

/**
 * Set the mask floor as a maximum attenuation in dB.  −80 dB applies the full
 * learned mask, 0 dB is a passthrough, and intermediate values keep a guard
 * floor under the mask so the original noise bed cannot be mixed back in.
 */
function setMaxAttenuationDb(value) {
  const db = Math.max(-80, Math.min(0, Math.round(Number(value) / 5) * 5));
  state.maxAttenuationDb = Number.isFinite(db) ? db : -80;
  ui.attenuationSlider.value = String(state.maxAttenuationDb);
  ui.attenuationValue.textContent = formatSignedDb(state.maxAttenuationDb);
  // The worklet slews the dB value over 100 ms, so this is click-free and never
  // reloads or resets the model.
  postLive({ type: 'attenuation', value: state.maxAttenuationDb });
}

function currentParams() {
  return {
    maxAttenuationDb: state.maxAttenuationDb,
    gain: dbToGain(state.gainDb),
    gainDb: state.gainDb,
  };
}

// ---------------------------------------------------------------------------
// object URLs
// ---------------------------------------------------------------------------

function revokeBefore() {
  if (state.beforeUrl) {
    URL.revokeObjectURL(state.beforeUrl);
    state.beforeUrl = null;
  }
  ui.beforeAudio.pause();
  ui.beforeAudio.removeAttribute('src');
  ui.beforeAudio.load();
}

function clearOutcome() {
  if (state.afterUrl) {
    URL.revokeObjectURL(state.afterUrl);
    state.afterUrl = null;
  }
  ui.afterAudio.pause();
  ui.afterAudio.removeAttribute('src');
  ui.afterAudio.load();
  ui.downloadLink.removeAttribute('href');
  ui.downloadLink.removeAttribute('download');
  ui.stats.textContent = '';
  ui.results.hidden = true;
  hideProgress();
}

function cleanupUrls() {
  if (state.beforeUrl) URL.revokeObjectURL(state.beforeUrl);
  if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
  state.beforeUrl = null;
  state.afterUrl = null;
}

// ---------------------------------------------------------------------------
// worker orchestration
// ---------------------------------------------------------------------------

function ensureWorker() {
  if (state.workerBroken || typeof Worker === 'undefined') return null;
  if (state.worker) return state.worker;
  try {
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
    worker.onerror = handleWorkerError;
    state.worker = worker;
    return worker;
  } catch {
    state.workerBroken = true;
    return null;
  }
}

function warmWorker() {
  if (state.warmed) return;
  const worker = ensureWorker();
  if (!worker) return;
  state.warmed = true;
  try {
    worker.postMessage({ type: 'warmup', id: 0 });
  } catch {
    state.workerBroken = true;
    state.worker = null;
  }
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (!message || typeof message !== 'object') return;
  if (message.type === 'progress') {
    if (state.pending && message.id === state.pending.id) {
      showProgress(message.percent);
      setStatus(`Processing audio… ${message.percent}%`);
    }
    return;
  }
  if (message.type === 'spectrum') {
    const pending = state.pending;
    if (!pending || message.id !== pending.id) return;
    spectrogram.appendPacked(message.columns, message.count);
    return;
  }
  if (message.type === 'done') {
    const pending = state.pending;
    if (!pending || message.id !== pending.id) return;
    state.pending = null;
    pending.resolve(message);
    return;
  }
  if (message.type === 'error') {
    const pending = state.pending;
    if (!pending || message.id !== pending.id) return;
    state.pending = null;
    pending.reject(new Error(message.message || 'An unknown error occurred while processing the audio.'));
    return;
  }
  // `ready` from the warm-up is intentionally silent.
}

function handleWorkerError(event) {
  if (event && typeof event.preventDefault === 'function') event.preventDefault();
  const pending = state.pending;
  state.pending = null;
  state.workerBroken = true;
  if (state.worker) {
    try {
      state.worker.terminate();
    } catch {
      // Already gone.
    }
    state.worker = null;
  }
  if (pending) {
    const error = new Error('The background worker is unavailable; switched to main-thread processing.');
    error.workerFailure = true;
    pending.reject(error);
  }
}

function runWorker(message, transfer) {
  return new Promise((resolve, reject) => {
    if (!state.worker) {
      const error = new Error('The background worker is unavailable.');
      error.workerFailure = true;
      reject(error);
      return;
    }
    const id = state.nextId++;
    state.pending = { id, resolve, reject };
    try {
      state.worker.postMessage({ ...message, id }, transfer);
    } catch {
      state.pending = null;
      const error = new Error('Could not start background processing; switched to the main thread.');
      error.workerFailure = true;
      reject(error);
    }
  });
}

function runInWorker(samples, params) {
  // Transfer a copy so the original is still available for a fallback.
  const copy = samples.slice();
  return runWorker(
    {
      type: 'process',
      samples: copy.buffer,
      sampleRate: SAMPLE_RATE,
      maxAttenuationDb: params.maxAttenuationDb,
      gain: params.gain,
    },
    [copy.buffer]
  );
}

// ---------------------------------------------------------------------------
// main-thread fallback
// ---------------------------------------------------------------------------

async function getMainEngine() {
  if (!state.mainEnginePromise) {
    if (typeof location !== 'undefined' && location.protocol === 'file:') {
      throw new Error(
        'Open this page through a static web server; browsers do not allow loading WebAssembly from a local file.'
      );
    }
    state.mainEnginePromise = (async () => {
      const response = await fetch(WASM_URL);
      if (!response.ok) {
        throw new Error(`Could not load the denoising engine (HTTP ${response.status}).`);
      }
      return loadVoiceEngine(await response.arrayBuffer());
    })().catch((error) => {
      state.mainEnginePromise = null;
      throw error;
    });
  }
  return state.mainEnginePromise;
}

async function runOnMainThread(samples, params) {
  setStatus('Preparing the engine (compatibility mode)…');
  const engine = await getMainEngine();
  setStatus('Processing audio (compatibility mode)…');
  // Give the browser one frame to paint the status before the blocking loop.
  await new Promise((resolve) => setTimeout(resolve, 30));
  const collector = new SpectrogramBuffer(FILE_HISTORY);
  const result = processPcm(samples, engine, {
    maxAttenuationDb: params.maxAttenuationDb,
    maxSpectrumColumns: FILE_HISTORY,
    onSpectrum: (index, inputDb, outputDb) => collector.pushDb(inputDb, outputDb),
    onProgress: (done, total) => {
      const percent = Math.floor((done / total) * 100);
      showProgress(percent);
      setStatus(`Processing audio (compatibility mode)… ${percent}%`);
    },
  });
  if (Number.isFinite(params.gain) && params.gain !== 1) {
    for (let i = 0; i < result.samples.length; i++) result.samples[i] *= params.gain;
  }
  const columns = collector.pack(0, collector.length);
  return {
    ...result,
    wav: encodeWav(result.samples),
    mode: 'main',
    spectrogram: { columns, count: collector.length, bins: SPEC_BINS },
  };
}

// ---------------------------------------------------------------------------
// file selection and processing
// ---------------------------------------------------------------------------

function looksLikeAudio(file) {
  if (file.type && file.type.startsWith('audio/')) return true;
  return /\.(wav|wave|mp3|m4a|aac|ogg|oga|flac|opus|webm|aif|aiff)$/i.test(file.name || '');
}

function downloadName(name) {
  const base = (name || 'audio').replace(/\.[^./\\]+$/, '').trim() || 'audio';
  return `${base}-denoised.wav`;
}

function selectFile(file) {
  if (state.busy || !file) return;
  if (!looksLikeAudio(file)) {
    showError('Please choose an audio file (WAV, MP3, M4A, OGG and similar).');
    return;
  }
  if (file.size === 0) {
    showError('This file is empty. Please choose another audio file.');
    return;
  }

  clearOutcome();
  revokeBefore();
  state.file = file;

  ui.fileName.textContent = `Selected: ${file.name}`;
  ui.fileName.hidden = false;
  clearError();
  setStatus('Ready — press Process to start.');
  updateProcessButton();

  try {
    state.beforeUrl = URL.createObjectURL(file);
    ui.beforeAudio.src = state.beforeUrl;
  } catch {
    state.beforeUrl = null;
  }

  warmWorker();
}

function finish(outcome, params) {
  if (!outcome || !outcome.wav) throw new Error('The processing result is invalid.');
  if (outcome.spectrogram && outcome.spectrogram.columns instanceof Uint8Array) {
    spectrogram.setPacked(outcome.spectrogram.columns, outcome.spectrogram.count);
  }
  const blob = new Blob([outcome.wav], { type: 'audio/wav' });
  if (state.afterUrl) URL.revokeObjectURL(state.afterUrl);
  state.afterUrl = URL.createObjectURL(blob);

  ui.afterAudio.src = state.afterUrl;
  ui.afterAudio.load();
  ui.downloadLink.href = state.afterUrl;
  ui.downloadLink.download = downloadName(state.file ? state.file.name : 'audio');

  const duration = Number.isFinite(outcome.durationSeconds) ? outcome.durationSeconds : 0;
  const rtf = Number.isFinite(outcome.rtf) ? outcome.rtf.toFixed(3) : '—';
  const parts = [
    `Duration ${formatDuration(duration)}`,
    `Processed in ${formatMs(outcome.processingMs)}`,
    `RTF ${rtf}`,
    `${Number(outcome.frames || 0).toLocaleString('en-US')} frames`,
    `Max attenuation ${formatSignedDb(params.maxAttenuationDb)}`,
    `Gain ${formatSignedDb(params.gainDb)}`,
  ];
  ui.stats.textContent = parts.join(' · ');

  ui.results.hidden = false;
  showProgress(100);
  setStatus(outcome.clippedSamples > 0 ? 'Done (the output peak was limited).' : 'Done.');
}

async function processCurrentFile() {
  if (state.busy || !state.file) return;
  setBusy(true);
  clearError();
  clearOutcome();
  spectrogram.begin('file');
  setStatus('Decoding audio…');
  showProgress(null);
  // Snapshot the controls: a change made during processing applies next time.
  const params = currentParams();

  try {
    const samples = await decodeAudioFile(state.file);
    if (!(samples instanceof Float32Array) || samples.length === 0) {
      throw new Error('The audio file is empty.');
    }

    showProgress(0);
    setStatus('Processing audio… 0%');

    let outcome;
    if (state.worker) {
      try {
        outcome = await runInWorker(samples, params);
      } catch (error) {
        if (!error || !error.workerFailure) throw error;
        outcome = await runOnMainThread(samples, params);
      }
    } else {
      outcome = await runOnMainThread(samples, params);
    }

    finish(outcome, params);
  } catch (error) {
    const message = error && error.message ? error.message : 'An unknown error occurred while processing the audio.';
    if (typeof console !== 'undefined') console.error('[CleanField]', error);
    showError(message);
    setStatus('Processing failed.');
    hideProgress();
  } finally {
    setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------

ui.tabLive.addEventListener('click', () => selectTab('live'));
ui.tabFile.addEventListener('click', () => selectTab('file'));
ui.tabLive.addEventListener('keydown', handleTabKey);
ui.tabFile.addEventListener('keydown', handleTabKey);

ui.startButton.addEventListener('click', () => {
  if (state.live) {
    void stopLive();
  } else {
    void startLive();
  }
});

ui.rawButton.addEventListener('click', () => setLiveBypass(true));
ui.denoisedButton.addEventListener('click', () => setLiveBypass(false));

ui.attenuationSlider.addEventListener('input', () =>
  setMaxAttenuationDb(ui.attenuationSlider.value)
);
ui.gainSlider.addEventListener('input', () => setGainDb(ui.gainSlider.value));

/** Wire the drop zone, its choose button and its hidden file input. */
function wireFileZone(zone, button, input, onFile) {
  button.addEventListener('click', (event) => {
    event.stopPropagation();
    if (!state.busy) input.click();
  });

  zone.addEventListener('click', (event) => {
    if (state.busy) return;
    if (event.target.closest('button')) return;
    input.click();
  });

  zone.addEventListener('keydown', (event) => {
    if (state.busy) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });

  let depth = 0;
  zone.addEventListener('dragenter', (event) => {
    event.preventDefault();
    if (state.busy) return;
    depth += 1;
    zone.classList.add('is-dragging');
  });
  zone.addEventListener('dragover', (event) => {
    event.preventDefault();
    if (state.busy) return;
    event.dataTransfer.dropEffect = 'copy';
    zone.classList.add('is-dragging');
  });
  zone.addEventListener('dragleave', (event) => {
    event.preventDefault();
    depth = Math.max(0, depth - 1);
    if (depth === 0) zone.classList.remove('is-dragging');
  });
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    depth = 0;
    zone.classList.remove('is-dragging');
    if (state.busy) return;
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) onFile(file);
  });

  input.addEventListener('change', (event) => {
    const files = event.target.files;
    const file = files && files[0];
    event.target.value = '';
    if (file) onFile(file);
  });
}

wireFileZone(ui.dropZone, ui.chooseButton, ui.fileInput, selectFile);

ui.processButton.addEventListener('click', () => {
  void processCurrentFile();
});

window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());
window.addEventListener('pagehide', () => {
  cleanupUrls();
  if (state.live) void stopLive({ updateStatus: false });
});

// Initial UI state.
updateLiveControls();
updateGainLabel();
setMaxAttenuationDb(-80);
resetLiveMeters();
selectTab('live');

// Diagnostics hook used by the optional browser check; it does not affect the UI.
window.__CLEANFIELD__ = {
  selectTab,
  startLive,
  stopLive,
  setLiveBypass,
  setGainDb,
  setMaxAttenuationDb,
  spectrogram: () => spectrogram.diagnostics(),
  clearSpectrogram: () => spectrogram.clear(),
  setSpectrogramFrozen: (value) => spectrogram.setFrozen(value),
  getState: () => ({
    tab: state.tab,
    liveStatus: ui.liveStatus.textContent,
    liveError: ui.liveError.hidden ? null : ui.liveError.textContent,
    fileStatus: ui.status.textContent,
    params: {
      maxAttenuationDb: state.maxAttenuationDb,
      gainDb: state.gainDb,
    },
    spectrogram: {
      mode: spectrogram.mode,
      frozen: spectrogram.frozen,
      columns: spectrogram.buffer.length,
    },
    live: state.live
      ? {
          ready: state.live.ready,
          contextRate: state.live.ctx.sampleRate,
          frames: state.live.frames,
          frameMs: state.live.frameMs,
          rtf: state.live.rtf,
          levelIn: state.live.levelIn,
          levelOut: state.live.levelOut,
          peakIn: state.live.peakIn,
          peakOut: state.live.peakOut,
        }
      : null,
  }),
};
