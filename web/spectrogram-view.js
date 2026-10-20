/**
 * CleanField Voice 1.0 — main-thread spectrogram renderer.
 *
 * Keeps the audio thread free: the worklet posts small quantised column
 * batches, the offline worker posts batches with its progress messages, and
 * this module only paints at most 30 times a second.
 *
 * Two stacked canvases share one SpectrogramBuffer and one dBFS scale.  Each
 * column is max-pooled from 513 bins to 128 rows, then the visible columns are
 * area-averaged onto the display width with constant total weight, so the
 * horizontal scaling cannot introduce periodic vertical bands.
 */

import {
  FILE_HISTORY,
  LIVE_HISTORY,
  SPEC_BINS,
  SPEC_IMAGE_ROWS,
  SpectrogramBuffer,
  aggregateColumns,
  buildColormapRgba32,
  buildRowBounds,
  colormapCss,
  poolBinsToRows,
} from './spectrogram-core.js';

const MAX_FPS = 30;
const FRAME_INTERVAL = 1000 / MAX_FPS;
const EMPTY_IDLE = 'Start live mode or process a file to see the spectrogram.';
const EMPTY_LIVE = 'Waiting for audio…';
const EMPTY_FILE = 'Computing spectrogram…';

export class SpectrogramPainter {
  constructor(canvas, plane) {
    this.canvas = canvas;
    this.plane = plane;
    this.ctx = canvas.getContext('2d', { alpha: false });
    // Small canvas that receives the aggregated columns before one drawImage.
    this.aggImage = document.createElement('canvas');
    this.aggImage.width = 1;
    this.aggImage.height = SPEC_IMAGE_ROWS;
    this.aggCtx = this.aggImage.getContext('2d', { alpha: false });
    this.imageData = null;
    this.pixels = null;
    this.capacity = 0;
    this.rowBins = 0;
    this.rowLow = null;
    this.rowHigh = null;
    this.pooled = null;
    this.aggBytes = null;
    this.aggColumns = 0;
    this.columns = 0;
    this.lut = buildColormapRgba32();
  }

  blank() {
    const ctx = this.ctx;
    ctx.fillStyle = '#fbfafc';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  resizeToDisplay() {
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const cssWidth = this.canvas.clientWidth || 320;
    const cssHeight = this.canvas.clientHeight || 132;
    const width = Math.max(1, Math.round(cssWidth * dpr));
    const height = Math.max(1, Math.round(cssHeight * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.blank();
    }
  }

  /** Allocate the per-column row pool and the bin-to-row bounds (once). */
  ensureCapacity(capacity, bins) {
    if (this.pooled && this.capacity === capacity && this.rowBins === bins) return;
    this.capacity = capacity;
    this.rowBins = bins;
    this.pooled = new Uint8Array(capacity * SPEC_IMAGE_ROWS);
    const bounds = buildRowBounds(bins, SPEC_IMAGE_ROWS);
    this.rowLow = bounds.low;
    this.rowHigh = bounds.high;
  }

  /** Allocate the destination-width image when the canvas width changes. */
  ensureAggregate(target) {
    if (this.imageData && this.aggColumns === target) return;
    this.aggColumns = target;
    this.aggImage.width = target;
    this.aggImage.height = SPEC_IMAGE_ROWS;
    this.imageData = this.aggCtx.createImageData(target, SPEC_IMAGE_ROWS);
    this.pixels = new Uint32Array(this.imageData.data.buffer);
    this.aggBytes = new Uint8Array(target * SPEC_IMAGE_ROWS);
  }

  draw(buffer) {
    this.resizeToDisplay();
    const columns = buffer.length;
    if (columns === 0) {
      this.blank();
      this.columns = 0;
      return;
    }

    const rows = SPEC_IMAGE_ROWS;
    this.ensureCapacity(buffer.capacity, buffer.bins);
    const data = this.plane === 'output' ? buffer.output : buffer.input;
    const pooled = this.pooled;
    const rowLow = this.rowLow;
    const rowHigh = this.rowHigh;

    // Row 0 is the top of the panel (8 kHz); the 513 bins are max-pooled so
    // narrow tonal peaks stay visible after the vertical downscale.
    for (let c = 0; c < columns; c++) {
      poolBinsToRows(data, buffer.columnOffset(c), rowLow, rowHigh, pooled, c * rows);
    }

    // One destination column per visible screen column.  aggregateColumns uses
    // a constant total weight for every destination column, so the horizontal
    // area filter does not alternate between one and two source columns.
    const target = Math.max(1, Math.min(this.canvas.width, columns));
    this.ensureAggregate(target);
    aggregateColumns(pooled, columns, rows, target, this.aggBytes);

    const pixels = this.pixels;
    const lut = this.lut;
    const count = target * rows;
    for (let i = 0; i < count; i++) pixels[i] = lut[this.aggBytes[i]];
    this.aggCtx.putImageData(this.imageData, 0, 0);

    const ctx = this.ctx;
    // Interpolation only helps when shrinking.  When a column occupies more
    // than one device pixel, smoothing draws a soft band at every column edge,
    // so nearest-neighbour keeps the columns crisp instead.
    const upscaling = this.canvas.width > target;
    ctx.imageSmoothingEnabled = !upscaling;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.aggImage, 0, 0, target, rows, 0, 0, this.canvas.width, this.canvas.height);
    this.columns = columns;
  }
}

export class SpectrogramView {
  constructor(elements, options = {}) {
    this.elements = elements;
    this.inputPainter = new SpectrogramPainter(elements.inputCanvas, 'input');
    this.outputPainter = new SpectrogramPainter(elements.outputCanvas, 'output');
    this.buffer = new SpectrogramBuffer(
      Number.isFinite(options.capacity) ? options.capacity : LIVE_HISTORY
    );
    this.mode = 'idle';
    this.frozen = false;
    this.dirty = true;
    this.forcePaint = true;
    this.lastPaint = 0;
    this.paintCount = 0;
    this.raf = 0;

    if (elements.ramp) elements.ramp.style.background = colormapCss();

    elements.freezeButton.addEventListener('click', () => this.setFrozen(!this.frozen));
    elements.clearButton.addEventListener('click', () => this.clear());

    this.resizeListener = () => {
      this.dirty = true;
    };
    if (typeof ResizeObserver === 'function') {
      this.resizeObserver = new ResizeObserver(this.resizeListener);
      this.resizeObserver.observe(elements.inputCanvas);
      this.resizeObserver.observe(elements.outputCanvas);
    } else {
      window.addEventListener('resize', this.resizeListener);
    }

    this.loop = this.loop.bind(this);
    this.raf = requestAnimationFrame(this.loop);
    this.updateLabels();
  }

  loop(timestamp) {
    this.raf = requestAnimationFrame(this.loop);
    if (this.frozen && !this.forcePaint) return;
    if (!this.dirty) return;
    if (timestamp - this.lastPaint < FRAME_INTERVAL - 1) return;
    this.lastPaint = timestamp;
    this.dirty = false;
    this.forcePaint = false;
    this.paint();
  }

  paint() {
    this.inputPainter.draw(this.buffer);
    this.outputPainter.draw(this.buffer);
    this.paintCount += 1;
    this.updateLabels();
  }

  begin(mode) {
    this.mode = mode === 'file' ? 'file' : 'live';
    const capacity = this.mode === 'file' ? FILE_HISTORY : LIVE_HISTORY;
    this.buffer = new SpectrogramBuffer(capacity);
    this.setFrozen(false);
    this.forcePaint = true;
    this.dirty = true;
    this.updateLabels();
  }

  /** Append a packed column batch: [input plane | output plane]. */
  appendPacked(columns, count) {
    if (this.frozen) return;
    this.buffer.appendPacked(columns, count);
    this.dirty = true;
    this.updateLabels();
  }

  /** Replace the buffer with a completed file spectrogram. */
  setPacked(columns, count) {
    this.buffer.clear();
    this.buffer.appendPacked(columns, count);
    if (!this.frozen) this.dirty = true;
    this.updateLabels();
  }

  clear() {
    this.buffer.clear();
    this.forcePaint = true;
    this.dirty = true;
    this.updateLabels();
  }

  setFrozen(value) {
    this.frozen = Boolean(value);
    const button = this.elements.freezeButton;
    button.setAttribute('aria-pressed', this.frozen ? 'true' : 'false');
    button.classList.toggle('is-active', this.frozen);
    button.textContent = this.frozen ? 'Frozen' : 'Freeze';
    if (!this.frozen) {
      this.dirty = true;
      this.forcePaint = true;
    }
    this.updateLabels();
  }

  updateLabels() {
    const { inputEmpty, outputEmpty, span, xStart, xEnd } = this.elements;
    const columns = this.buffer.length;
    const hasData = columns > 0;
    inputEmpty.hidden = hasData;
    outputEmpty.hidden = hasData;
    let emptyText = EMPTY_IDLE;
    if (this.mode === 'live') emptyText = EMPTY_LIVE;
    else if (this.mode === 'file') emptyText = EMPTY_FILE;
    if (!hasData) {
      inputEmpty.textContent = emptyText;
      outputEmpty.textContent = emptyText;
    }

    if (xStart.textContent !== (this.mode === 'file' ? 'start' : 'older')) {
      xStart.textContent = this.mode === 'file' ? 'start' : 'older';
    }
    if (xEnd.textContent !== (this.mode === 'file' ? 'end' : 'newest')) {
      xEnd.textContent = this.mode === 'file' ? 'end' : 'newest';
    }

    let text;
    if (this.mode === 'idle') {
      text = EMPTY_IDLE;
    } else if (this.mode === 'live') {
      const seconds = (columns * 0.016).toFixed(1);
      text = `${columns} frames · ${seconds} s · newest at right${this.frozen ? ' · frozen' : ''}`;
    } else {
      text = `full file · ${columns} columns${this.frozen ? ' · frozen' : ''}`;
    }
    if (span.textContent !== text) span.textContent = text;
  }

  diagnostics() {
    return {
      mode: this.mode,
      frozen: this.frozen,
      columns: this.buffer.length,
      capacity: this.buffer.capacity,
      bins: this.buffer.bins,
      expectedBins: SPEC_BINS,
      inputMeanByte: this.buffer.meanByte('input', 5),
      outputMeanByte: this.buffer.meanByte('output', 5),
      paintCount: this.paintCount,
      canvas: {
        width: this.elements.inputCanvas.width,
        height: this.elements.inputCanvas.height,
      },
    };
  }
}


