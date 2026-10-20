/**
 * CleanField Voice 1.0 — spectrogram core.
 *
 * Dependency-free helpers shared by the offline DSP (`dsp.js`), the live DSP
 * (`live-dsp.js`), the AudioWorklet, the file worker, the DOM renderer and the
 * Node tests.  Nothing here touches the DOM, so it is safe to import from an
 * AudioWorkletGlobalScope or from Node.
 *
 * Frequency axis: linear 0–8 kHz.  A 1024-point FFT at 16 kHz has 15.625 Hz
 * per bin, so all 513 bins map directly onto 0…8 kHz with no aggregation or
 * log-frequency interpolation.  The two panels use one fixed dBFS range
 * (−100…0 dB) so they are directly comparable.
 */

/** 0…8 kHz at 16 kHz / 1024-point FFT: bins 0..512 inclusive. */
export const SPEC_BINS = 513;
export const SPEC_MIN_DB = -100;
export const SPEC_MAX_DB = 0;
/** Rolling history used by the live panel (200 × 16 ms ≈ 3.2 s). */
export const LIVE_HISTORY = 1024;
/** Enough columns to show a whole file; processPcm decimates beyond this. */
export const FILE_HISTORY = 4096;
/** How many columns the worklet batches into one postMessage (~128 ms). */
export const SPEC_BATCH_COLUMNS = 8;
/** Rows in the rendered image; 513 bins are max-pooled down to this. */
export const SPEC_IMAGE_ROWS = 128;

export function clampDb(db) {
  if (!Number.isFinite(db)) return SPEC_MIN_DB;
  if (db < SPEC_MIN_DB) return SPEC_MIN_DB;
  if (db > SPEC_MAX_DB) return SPEC_MAX_DB;
  return db;
}

/**
 * Fill one dBFS magnitude column from a complex spectrum.
 *
 * `norm` converts the FFT magnitude to full-scale amplitude.  Both panels use
 * the windowed analysis transform, whose normalisation is 2 / sum(window), so a
 * bin-centred full-scale sine reads 0 dBFS.
 */
export function fillDbColumn(re, im, count, norm, out) {
  for (let k = 0; k < count; k++) {
    const value = re[k];
    const imag = im[k];
    const magnitude = Math.sqrt(value * value + imag * imag) * norm;
    const db = magnitude > 0 ? 20 * Math.log10(magnitude) : SPEC_MIN_DB;
    out[k] = db < SPEC_MIN_DB ? SPEC_MIN_DB : db > SPEC_MAX_DB ? SPEC_MAX_DB : db;
  }
}

/** Quantise a dBFS value to 0..255 over the fixed display range. */
export function quantizeDb(db) {
  const scaled = ((db - SPEC_MIN_DB) * 255) / (SPEC_MAX_DB - SPEC_MIN_DB);
  if (!(scaled > 0)) return 0;
  if (scaled >= 255) return 255;
  return Math.round(scaled);
}

/** Quantise a whole dBFS column into `out` at `offset`. */
export function writeQuantizedColumn(db, out, offset) {
  for (let k = 0; k < db.length; k++) {
    const scaled = ((db[k] - SPEC_MIN_DB) * 255) / (SPEC_MAX_DB - SPEC_MIN_DB);
    out[offset + k] = scaled <= 0 ? 0 : scaled >= 255 ? 255 : Math.round(scaled);
  }
}

// ---------------------------------------------------------------------------
// painter scaling (pure, so the Node check can exercise it)
// ---------------------------------------------------------------------------

/**
 * Precompute which FFT bins each display row covers.  Row 0 is the top of the
 * panel (8 kHz), and the 513 bins are max-pooled into `rows` bands so narrow
 * peaks stay visible.
 */
export function buildRowBounds(bins, rows) {
  const low = new Uint16Array(rows);
  const high = new Uint16Array(rows);
  for (let r = 0; r < rows; r++) {
    let lo = Math.floor(((rows - 1 - r) * bins) / rows);
    let hi = Math.floor(((rows - r) * bins) / rows);
    if (hi <= lo) hi = lo + 1;
    if (hi > bins) hi = bins;
    low[r] = lo;
    high[r] = hi;
  }
  return { low, high };
}

/** Max-pool one column of quantised bins into one column of display rows. */
export function poolBinsToRows(data, offset, rowLow, rowHigh, out, outOffset) {
  const rows = rowLow.length;
  for (let r = 0; r < rows; r++) {
    let value = 0;
    const high = rowHigh[r];
    for (let b = rowLow[r]; b < high; b++) {
      const byte = data[offset + b];
      if (byte > value) value = byte;
    }
    out[outOffset + r] = value;
  }
}

/**
 * Horizontal downscale of a pooled column image.
 *
 * Destination column `c` is the area average of the source columns that
 * overlap `[c * columns / target, (c + 1) * columns / target)`, with fractional
 * weights at the two edges.  Every destination column therefore has the same
 * total weight (`group`), which is what prevents the evenly spaced vertical
 * lines the former integer group sizes produced: a group of one source column
 * and a group of two source columns no longer receive different scaling.
 *
 * `pooled` is column-major: column `s`, row `r` is at `s * rows + r`.
 */
export function aggregateColumns(pooled, columns, rows, target, out) {
  const group = columns / target;
  for (let c = 0; c < target; c++) {
    const start = c * group;
    const end = start + group;
    const first = Math.floor(start);
    const last = Math.ceil(end);
    for (let r = 0; r < rows; r++) {
      let sum = 0;
      for (let s = first; s < last; s++) {
        const weight = Math.min(end, s + 1) - Math.max(start, s);
        sum += pooled[s * rows + r] * weight;
      }
      let value = Math.round(sum / group);
      if (value < 0) value = 0;
      else if (value > 255) value = 255;
      out[r * target + c] = value;
    }
  }
}

/**
 * Ring buffer of quantised spectrogram columns.
 *
 * Columns are stored oldest → newest.  Each plane (`input`, `output`) holds one
 * byte per bin per column.  The live view uses a 200-column ring; the file view
 * uses a 4096-column buffer that is filled in one pass.
 */
export class SpectrogramBuffer {
  constructor(capacity = LIVE_HISTORY, bins = SPEC_BINS) {
    this.bins = bins >>> 0;
    this.capacity = Math.max(1, capacity >>> 0);
    this.input = new Uint8Array(this.capacity * this.bins);
    this.output = new Uint8Array(this.capacity * this.bins);
    this.start = 0;
    this.count = 0;
  }

  get length() {
    return this.count;
  }

  /** Byte offset of visible column `index` (0 = oldest) in each plane. */
  columnOffset(index) {
    if (index < 0 || index >= this.count) throw new RangeError('spectrogram column out of range');
    return ((this.start + index) % this.capacity) * this.bins;
  }

  clear() {
    this.start = 0;
    this.count = 0;
    this.input.fill(0);
    this.output.fill(0);
  }

  /** Add one column from Float32 dBFS arrays (reused by the DSP). */
  pushDb(inputDb, outputDb) {
    let offset;
    if (this.count < this.capacity) {
      offset = ((this.start + this.count) % this.capacity) * this.bins;
      this.count += 1;
    } else {
      offset = this.start * this.bins;
      this.start = (this.start + 1) % this.capacity;
    }
    const bins = this.bins;
    for (let k = 0; k < bins; k++) {
      const i = inputDb[k];
      const scaled = ((i - SPEC_MIN_DB) * 255) / (SPEC_MAX_DB - SPEC_MIN_DB);
      this.input[offset + k] = scaled <= 0 ? 0 : scaled >= 255 ? 255 : Math.round(scaled);
      const o = outputDb[k];
      const scaledOut = ((o - SPEC_MIN_DB) * 255) / (SPEC_MAX_DB - SPEC_MIN_DB);
      this.output[offset + k] = scaledOut <= 0 ? 0 : scaledOut >= 255 ? 255 : Math.round(scaledOut);
    }
  }

  /**
   * Append a packed batch: [input plane | output plane], each `columns * bins`.
   * Used for worklet batches and worker progress messages.
   */
  appendPacked(packed, columns) {
    if (!(packed instanceof Uint8Array) || columns <= 0) return 0;
    const bins = this.bins;
    const available = Math.min(columns, Math.floor(packed.length / (bins * 2)));
    for (let c = 0; c < available; c++) {
      let offset;
      if (this.count < this.capacity) {
        offset = ((this.start + this.count) % this.capacity) * bins;
        this.count += 1;
      } else {
        offset = this.start * bins;
        this.start = (this.start + 1) % this.capacity;
      }
      const inBase = c * bins;
      const outBase = available * bins + inBase;
      this.input.set(packed.subarray(inBase, inBase + bins), offset);
      this.output.set(packed.subarray(outBase, outBase + bins), offset);
    }
    return available;
  }

  /** Pack columns `[from, to)` as [input plane | output plane]. */
  pack(from = 0, to = this.count) {
    const begin = Math.max(0, Math.min(from, this.count));
    const end = Math.max(begin, Math.min(to, this.count));
    const columns = end - begin;
    const bins = this.bins;
    const packed = new Uint8Array(columns * bins * 2);
    for (let c = 0; c < columns; c++) {
      const offset = this.columnOffset(begin + c);
      packed.set(this.input.subarray(offset, offset + bins), c * bins);
      packed.set(this.output.subarray(offset, offset + bins), columns * bins + c * bins);
    }
    return packed;
  }

  /** Mean quantised level of one plane (0..255), for diagnostics/checks. */
  meanByte(plane = 'input', step = 1) {
    const data = plane === 'output' ? this.output : this.input;
    if (this.count === 0) return 0;
    const stride = Math.max(1, step | 0);
    let sum = 0;
    let n = 0;
    for (let c = 0; c < this.count; c++) {
      const offset = this.columnOffset(c);
      for (let k = 0; k < this.bins; k += stride) {
        sum += data[offset + k];
        n += 1;
      }
    }
    return n > 0 ? sum / n : 0;
  }
}

// ---------------------------------------------------------------------------
// colormap (purple → magenta → orange → white, inferno-like)
// ---------------------------------------------------------------------------

/** Small code table; no external assets. */
export const COLORMAP_STOPS = [
  [13, 6, 40], // near-black purple
  [45, 10, 94], // deep purple
  [94, 15, 118], // violet
  [160, 27, 109], // magenta
  [214, 59, 74], // red-magenta
  [242, 118, 42], // orange
  [251, 177, 60], // amber
  [253, 224, 138], // pale yellow
  [255, 251, 232], // warm white
];

function sampleStops(t) {
  const clamped = t <= 0 ? 0 : t >= 1 ? 1 : t;
  const scaled = clamped * (COLORMAP_STOPS.length - 1);
  const index = Math.min(COLORMAP_STOPS.length - 2, Math.floor(scaled));
  const frac = scaled - index;
  const a = COLORMAP_STOPS[index];
  const b = COLORMAP_STOPS[index + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ];
}

const LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const rgb = sampleStops(i / 255);
    lut[i * 3] = rgb[0];
    lut[i * 3 + 1] = rgb[1];
    lut[i * 3 + 2] = rgb[2];
  }
  return lut;
})();

const LITTLE_ENDIAN = (() => {
  const probe = new Uint8Array(new Uint32Array([0x0a0b0c0d]).buffer);
  return probe[0] === 0x0d;
})();

const RGBA_LUT = (() => {
  const lut = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const r = LUT[i * 3];
    const g = LUT[i * 3 + 1];
    const b = LUT[i * 3 + 2];
    lut[i] = LITTLE_ENDIAN
      ? (255 << 24) | (b << 16) | (g << 8) | r
      : (r << 24) | (g << 16) | (b << 8) | 255;
  }
  return lut;
})();

export function colormapRgb(t) {
  return sampleStops(t);
}

export function buildColormapLut() {
  return LUT.slice();
}

/** Packed RGBA words for ImageData; matches this platform's byte order. */
export function buildColormapRgba32() {
  return RGBA_LUT.slice();
}

export function colormapCss() {
  const stops = COLORMAP_STOPS.map((rgb, index) => {
    const percent = (index / (COLORMAP_STOPS.length - 1)) * 100;
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]}) ${percent.toFixed(1)}%`;
  });
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}
