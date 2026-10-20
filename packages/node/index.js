'use strict';

/**
 * CleanField voice engine for Node.js.
 *
 * The native addon (`cf-node.node`) is resolved in this order:
 *
 *   1. `CLEANFIELD_NODE_BINDING` (explicit path, for tests and custom builds);
 *   2. `prebuilds/<platform>-<arch>/cf-node.node` (downloaded from the release
 *      by `scripts/fetch-prebuilt.mjs`, or copied by `scripts/build-local.ps1`);
 *   3. an optional platform package (`@cleanfield/node-<platform>-<arch>`);
 *   4. `build/Release/cf_node.node` (a local CMake-style layout, if present);
 *   5. a local Cargo build in `target/<triple>/release` or `target/release`.
 *
 * There is no node-gyp step: the addon is a plain Rust cdylib renamed to
 * `.node`, and it resolves the N-API entry points from the host executable at
 * load time.
 */

const fs = require('node:fs');
const path = require('node:path');

const RUST_TRIPLES = {
  'win32-x64': 'x86_64-pc-windows-msvc',
  'win32-arm64': 'aarch64-pc-windows-msvc',
  'darwin-x64': 'x86_64-apple-darwin',
  'darwin-arm64': 'aarch64-apple-darwin',
  'linux-x64': 'x86_64-unknown-linux-gnu',
  'linux-arm64': 'aarch64-unknown-linux-gnu',
};

const LIBRARY_SUFFIX =
  process.platform === 'win32' ? '.dll' : process.platform === 'darwin' ? '.dylib' : '.so';

function platformTag() {
  const tag = `${process.platform}-${process.arch}`;
  if (!RUST_TRIPLES[tag]) {
    throw new Error(
      `CleanField does not publish a prebuilt Node addon for ${process.platform}/${process.arch}. ` +
        'Supported platforms: win32/linux x64 and arm64, darwin x64 and arm64. ' +
        'Build from source with `cargo build --release -p cf-node`.'
    );
  }
  return tag;
}

function bindingCandidates() {
  const tag = platformTag();
  const triple = RUST_TRIPLES[tag];
  const root = path.join(__dirname, '..', '..');
  const candidates = [];
  if (process.env.CLEANFIELD_NODE_BINDING) {
    candidates.push(process.env.CLEANFIELD_NODE_BINDING);
  }
  candidates.push(path.join(__dirname, 'prebuilds', tag, 'cf-node.node'));
  candidates.push(path.join(__dirname, 'build', 'Release', 'cf-node.node'));
  candidates.push(path.join(root, 'target', triple, 'release', `cf_node${LIBRARY_SUFFIX}`));
  candidates.push(path.join(root, 'target', 'release', `cf_node${LIBRARY_SUFFIX}`));
  return candidates;
}

/**
 * Load a native addon from `filename`.
 *
 * `require()` only treats a file as an addon when its extension is `.node`, so
 * a Cargo cdylib found in a build directory (`cf_node.dll`, `libcf_node.so`,
 * `libcf_node.dylib`) is loaded through `process.dlopen`, which is exactly what
 * `require()` uses for `.node` files.
 */
function loadNative(filename) {
  if (filename.endsWith('.node')) return require(filename);
  const loaded = new Module(filename, module);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  process.dlopen(loaded, filename);
  return loaded.exports;
}

function loadBinding() {
  const attempts = [];
  for (const candidate of bindingCandidates()) {
    if (!fs.existsSync(candidate)) {
      attempts.push(`${candidate} (missing)`);
      continue;
    }
    try {
      return loadNative(candidate);
    } catch (error) {
      attempts.push(`${candidate} (${error && error.message ? error.message : error})`);
    }
  }

  const tag = platformTag();
  const platformPackages = [`@cleanfield/node-${tag}`];
  if (process.platform === 'darwin') platformPackages.push('@cleanfield/node-darwin-universal');
  for (const name of platformPackages) {
    try {
      return require(name);
    } catch (error) {
      attempts.push(`${name} (${error && error.code ? error.code : error})`);
    }
  }

  throw new Error(
    'CleanField could not find its native addon. Tried:\n  ' +
      attempts.join('\n  ') +
      '\nBuild it with `cargo build --release -p cf-node`, copy it to ' +
      `prebuilds/${tag}/cf-node.node, or set CLEANFIELD_NODE_BINDING to its path.`
  );
}

function resolveModelPath(explicitPath) {
  if (explicitPath !== undefined && explicitPath !== null) {
    if (typeof explicitPath !== 'string' || explicitPath.length === 0) {
      throw new TypeError('createVoice: options.modelPath must be a non-empty string');
    }
    if (!fs.existsSync(explicitPath)) {
      throw new Error(`createVoice: the model file does not exist: ${explicitPath}`);
    }
    return explicitPath;
  }
  if (process.env.CLEANFIELD_VOICE_MODEL) {
    if (!fs.existsSync(process.env.CLEANFIELD_VOICE_MODEL)) {
      throw new Error(
        `createVoice: CLEANFIELD_VOICE_MODEL points at a missing file: ${process.env.CLEANFIELD_VOICE_MODEL}`
      );
    }
    return process.env.CLEANFIELD_VOICE_MODEL;
  }
  // 1. Bundled with the published package (copied by `npm pack`/prepack).
  // 2. A workspace checkout: packages/node/../../models.
  const candidates = [
    path.join(__dirname, 'models', 'voice-default.cfwm'),
    path.join(__dirname, '..', '..', 'models', 'voice-default.cfwm'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error(
    'createVoice: no voice model was found. Pass { modelPath } or set CLEANFIELD_VOICE_MODEL. ' +
      'The default container is `models/voice-default.cfwm` in the CleanField distribution.'
  );
}

const binding = loadBinding();

if (!binding || typeof binding.createVoice !== 'function') {
  throw new Error(
    'CleanField loaded its native addon but the addon does not export createVoice(). ' +
      'The binary and the JavaScript package versions may not match.'
  );
}

/**
 * Create one voice engine instance.
 *
 * @param {object} [options]
 * @param {string} [options.modelPath] `.cfwm` container to load. Defaults to the
 *   model bundled with the package (or `CLEANFIELD_VOICE_MODEL`).
 * @param {'auto'|'scalar'|'avx2'} [options.backend] Backend selection. `auto`
 *   uses AVX2+FMA when the CPU supports it and the scalar reference otherwise.
 *   An explicitly requested backend never falls back silently.
 * @returns {{ backend: string, process: (frame: Float32Array) => Float32Array, reset: () => void, close: () => void }}
 */
function createVoice(options = {}) {
  if (options === null || typeof options !== 'object') {
    throw new TypeError('createVoice(options) expects an options object');
  }
  const backend = options.backend === undefined ? 'auto' : options.backend;
  if (backend !== 'auto' && backend !== 'scalar' && backend !== 'avx2') {
    throw new TypeError('createVoice: options.backend must be "auto", "scalar" or "avx2"');
  }
  const modelPath = resolveModelPath(options.modelPath);
  return binding.createVoice({ modelPath, backend });
}

module.exports = { createVoice };
