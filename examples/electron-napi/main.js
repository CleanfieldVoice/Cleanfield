'use strict';
/**
 * EXAMPLE ONLY - not production code.
 *
 * Minimal Electron main-process integration skeleton for the CleanField voice
 * engine.  It shows where the denoiser goes in a native capture pipeline, how
 * a handle is owned and released, and how to self-check the addon against the
 * conformance pack.
 *
 * This file does NOT contain an audio front-end.  `featuresFromHop` is the
 * seam where a 256-sample hop of 16 kHz mono audio becomes the 513-value
 * log-power feature frame the engine consumes.  The reference implementation
 * of that stage is `../../web/dsp.js` (window, 1024-point FFT, feature scale);
 * copy or port it, or move the front-end into your addon.
 *
 * The Node addon is not built by this repository.  It must expose the small
 * N-API surface documented in README.md:
 *
 *   info()                         -> { abiVersion, inputLen, outputLen, workspaceBytes }
 *   create(modelBytes, backend?)   -> External handle
 *   process(handle, features)      -> Float32Array(1026)
 *   reset(handle)
 *   free(handle)
 *   backend(handle)                -> 'scalar' | 'avx2'
 */

const fs = require('node:fs');
const path = require('node:path');

const ADDON_PATH = process.env.CLEANFIELD_ADDON
  || path.join(__dirname, 'build', 'Release', 'cleanfield.node');
const MODEL_PATH = process.env.CLEANFIELD_MODEL
  || path.join(__dirname, '..', '..', 'models', 'voice-default.cfwm');

const HOP = 256;
const INPUT_LEN = 513;
const OUTPUT_LEN = 1026;

function loadAddon() {
  try {
    return require(ADDON_PATH);
  } catch (error) {
    throw new Error(
      `CleanField addon not found at ${ADDON_PATH}.\n` +
      'This example does not build the N-API addon; see README.md for the surface.\n' +
      `Underlying error: ${error.message}`
    );
  }
}

/** One engine handle.  Create one per capture stream; never share across streams. */
class FrameDenoiser {
  constructor(addon, modelBytes, backend = 0) {
    this.addon = addon;
    this.handle = addon.create(modelBytes, backend);
    const info = addon.info();
    if (info.inputLen !== INPUT_LEN || info.outputLen !== OUTPUT_LEN) {
      addon.free(this.handle);
      throw new Error(`unexpected frame shape ${info.inputLen} -> ${info.outputLen}`);
    }
    this.backend = addon.backend(this.handle);
  }

  process(features) {
    if (features.length !== INPUT_LEN) {
      throw new RangeError(`expected ${INPUT_LEN} features, got ${features.length}`);
    }
    return this.addon.process(this.handle, features);
  }

  reset() {
    this.addon.reset(this.handle);
  }

  close() {
    if (this.handle) {
      this.addon.free(this.handle);
      this.handle = null;
    }
  }
}

/** Turns arbitrary-length blocks into 256-sample hops; holds the remainder. */
class HopReblocker {
  constructor(onHop) {
    this.onHop = onHop;
    this.pending = new Float32Array(0);
  }

  push(block) {
    const merged = new Float32Array(this.pending.length + block.length);
    merged.set(this.pending);
    merged.set(block, this.pending.length);
    let offset = 0;
    while (merged.length - offset >= HOP) {
      this.onHop(merged.subarray(offset, offset + HOP));
      offset += HOP;
    }
    this.pending = merged.slice(offset);
  }

  /** Zero-fill the last partial hop at end of stream. */
  flush() {
    if (this.pending.length > 0) {
      const tail = new Float32Array(HOP);
      tail.set(this.pending);
      this.pending = new Float32Array(0);
      this.onHop(tail);
    }
  }
}

/**
 * The insertion point: capture callback -> downmix/resample to 16 kHz mono ->
 * this reblocker -> featuresFromHop -> engine -> mask -> your synthesis stage.
 */
function attachDenoiser(engine, { featuresFromHop, onMask }) {
  const reblocker = new HopReblocker((hop) => {
    const features = featuresFromHop(hop);
    if (!(features instanceof Float32Array) || features.length !== INPUT_LEN) {
      throw new Error('featuresFromHop must return Float32Array(513)');
    }
    onMask(engine.process(features), hop);
  });
  return {
    push: (block) => reblocker.push(block),
    flush: () => reblocker.flush(),
  };
}

// ---------------------------------------------------------------------------
// self-check against the conformance pack (engine boundary only)
// ---------------------------------------------------------------------------
function readF32(file) {
  const bytes = fs.readFileSync(file);
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
}

function selfTest() {
  const addon = loadAddon();
  const model = fs.readFileSync(MODEL_PATH);
  const pack = path.join(__dirname, '..', '..', 'conformance');
  const manifest = JSON.parse(fs.readFileSync(path.join(pack, 'manifest.json'), 'utf8'));
  const info = addon.info();
  console.log(`addon: abi=${info.abiVersion} input=${info.inputLen} ` +
              `output=${info.outputLen} workspace=${info.workspaceBytes}`);

  let total = 0;
  let exact = 0;
  for (const set of manifest.sets) {
    const engine = new FrameDenoiser(addon, model);
    engine.reset();
    console.log(`set: ${set.name} (${engine.backend})`);
    const inputs = readF32(path.join(pack, set.input_file));
    const expected = readF32(path.join(pack, set.output_file));
    const expectedBits = new Uint32Array(expected.buffer);
    let setExact = 0;
    for (let f = 0; f < set.frames; f++) {
      const features = inputs.subarray(f * INPUT_LEN, (f + 1) * INPUT_LEN);
      const output = engine.process(features);
      const bits = new Uint32Array(output.buffer, output.byteOffset, OUTPUT_LEN);
      for (let i = 0; i < OUTPUT_LEN; i++) {
        if (bits[i] === expectedBits[f * OUTPUT_LEN + i]) setExact += 1;
      }
    }
    console.log(`  ${set.name}: ${setExact}/${set.frames * OUTPUT_LEN} exact`);
    total += set.frames * OUTPUT_LEN;
    exact += setExact;
    engine.close();
  }
  const ok = exact === total;
  console.log(`RESULT: ${ok ? 'PASS' : 'FAIL'} (${exact}/${total})`);
  return ok;
}

function main() {
  if (process.argv.includes('--self-test')) {
    try {
      process.exitCode = selfTest() ? 0 : 1;
    } catch (error) {
      console.error(`${error.message}`);
      process.exitCode = 2;
    }
    return;
  }
  console.log('CleanField Electron example (skeleton).');
  console.log('  * wire attachDenoiser() into your native capture callback;');
  console.log('  * implement featuresFromHop() or port ../../web/dsp.js;');
  console.log('  * run `node main.js --self-test` once the addon is built.');
}

if (require.main === module) main();

module.exports = { FrameDenoiser, HopReblocker, attachDenoiser, loadAddon };
