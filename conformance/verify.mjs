#!/usr/bin/env node
/**
 * CleanField conformance verifier (WebAssembly).
 *
 * Loads the conformance pack, instantiates the shipped wasm module
 * (`web/cf_wasm.wasm`, voice model embedded), runs every input frame through
 * `cf_voice_process`, and compares the result with the expected values bit for
 * bit.  PASS means every one of the 123,120 expected float32 values matched,
 * on the first pass and after `cf_voice_reset()`.
 *
 * No dependencies: Node built-ins only.
 *
 * Usage:
 *   node conformance/verify.mjs
 *   node conformance/verify.mjs --wasm web/cf_wasm.wasm --json result.json
 *   node conformance/verify.mjs --model models/voice-default.cfwm   # external model
 *
 * Exit code 0 on PASS, 1 on FAIL or setup error.
 */

import { readFileSync, existsSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const INPUT_LEN = 513;
const OUTPUT_LEN = 1026;
const WORKSPACE_BYTES = 440416;
const MODEL_BYTES = 2554560;
const ABI_FORMAT = 'cleanfield-conformance/1';

function usage() {
  console.log(
    'usage: node verify.mjs [--pack DIR] [--wasm PATH] [--model PATH] [--json PATH] [--no-reset]'
  );
}

function parseArgs(argv) {
  const options = {
    pack: HERE,
    wasm: path.join(ROOT, 'web', 'cf_wasm.wasm'),
    model: null,
    json: null,
    reset: true,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const need = () => {
      if (i + 1 >= argv.length) throw new Error(`missing value for ${arg}`);
      return argv[++i];
    };
    switch (arg) {
      case '--pack':
        options.pack = path.resolve(need());
        break;
      case '--wasm':
        options.wasm = path.resolve(need());
        break;
      case '--model':
        options.model = path.resolve(need());
        break;
      case '--json':
        options.json = path.resolve(need());
        break;
      case '--no-reset':
        options.reset = false;
        break;
      case '--help':
      case '-h':
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`unknown option: ${arg}`);
    }
  }
  return options;
}

function sha256(file) {
  return createHash('sha256').update(readFileSync(file)).digest('hex');
}

function readF32(file) {
  const bytes = readFileSync(file);
  if (bytes.length % 4 !== 0) throw new Error(`${file}: length is not a multiple of 4`);
  const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(copy);
}

function compare(actual, expected) {
  const a = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const b = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let exact = 0;
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) {
      exact += 1;
      continue;
    }
    const diff = Math.abs(actual[i] - expected[i]);
    if (Number.isFinite(diff)) maxDiff = Math.max(maxDiff, diff);
    else maxDiff = Infinity;
  }
  return { exact, maxDiff };
}

function fail(message) {
  console.error(`error: ${message}`);
  process.exit(1);
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const manifestPath = path.join(options.pack, 'manifest.json');
  if (!existsSync(manifestPath)) fail(`missing ${manifestPath} (run generate_pack.py first)`);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (manifest.format !== ABI_FORMAT) fail(`unsupported pack format ${manifest.format}`);
  if (!existsSync(options.wasm)) {
    fail(`wasm module not found: ${options.wasm}\n` +
         'Build it with: cargo build --release --target wasm32-unknown-unknown -p cf-wasm');
  }

  const hashErrors = [];
  for (const entry of manifest.sets) {
    for (const [fileKey, hashKey] of [['input_file', 'input_sha256'], ['output_file', 'output_sha256']]) {
      const file = path.join(options.pack, entry[fileKey]);
      if (!existsSync(file)) hashErrors.push(`${entry.name}: missing ${file}`);
      else if (sha256(file) !== entry[hashKey]) hashErrors.push(`${entry.name}: ${fileKey} hash mismatch`);
    }
  }

  const wasmBytes = readFileSync(options.wasm);
  let instance;
  try {
    instance = new WebAssembly.Instance(new WebAssembly.Module(wasmBytes), {});
  } catch (error) {
    fail(`could not instantiate ${options.wasm}: ${error.message}`);
  }
  const e = instance.exports;
  for (const name of ['memory', 'cf_alloc', 'cf_voice_new', 'cf_voice_process',
                      'cf_voice_reset', 'cf_voice_free', 'cf_voice_input_len', 'cf_voice_output_len']) {
    if (!(name in e)) fail(`wasm module does not export ${name}`);
  }
  const errors = [];
  if (e.cf_voice_input_len() !== INPUT_LEN) errors.push('cf_voice_input_len() != 513');
  if (e.cf_voice_output_len() !== OUTPUT_LEN) errors.push('cf_voice_output_len() != 1026');
  if (e.cf_voice_workspace_bytes && e.cf_voice_workspace_bytes() !== WORKSPACE_BYTES) {
    errors.push(`cf_voice_workspace_bytes() != ${WORKSPACE_BYTES}`);
  }
  if (e.cf_fma_self_test && e.cf_fma_self_test() !== 0) errors.push('cf_fma_self_test() failed');
  const fmaSoft = e.cf_fma_is_soft ? e.cf_fma_is_soft() : null;

  let handle;
  let modelBytes;
  if (options.model) {
    const image = readFileSync(options.model);
    const ptr = e.cf_alloc_u32(Math.ceil(image.length / 4));
    if (!ptr) fail('cf_alloc_u32 failed');
    new Uint8Array(e.memory.buffer, ptr, image.length).set(image);
    handle = e.cf_voice_new(ptr, image.length);
    modelBytes = image.length;
  } else {
    handle = e.cf_voice_new(0, 0);
    modelBytes = e.cf_voice_model_image_len ? e.cf_voice_model_image_len() : null;
    if (modelBytes !== null && modelBytes !== MODEL_BYTES) {
      errors.push(`embedded model is ${modelBytes} bytes, expected ${MODEL_BYTES}`);
    }
  }
  if (!handle) {
    const code = e.cf_last_error ? e.cf_last_error() : '?';
    fail(`cf_voice_new failed (cf_last_error=${code})`);
  }

  const inPtr = e.cf_alloc(INPUT_LEN * 4, 8);
  const outPtr = e.cf_alloc(OUTPUT_LEN * 4, 8);
  if (!inPtr || !outPtr) fail('cf_alloc failed for the frame buffers');

  console.log(`pack:    ${options.pack}`);
  console.log(`wasm:    ${options.wasm} (${wasmBytes.length} bytes)`);
  console.log(`model:   ${options.model || 'embedded'} (${modelBytes} bytes)`);
  console.log(`abi:     wasm C surface; backend: ${fmaSoft === 1 ? 'scalar (software FMA)' : 'scalar'}; ` +
              `workspace: ${e.cf_voice_workspace_bytes ? e.cf_voice_workspace_bytes() : '?'} bytes`);
  console.log();

  const header = ['set'.padEnd(18), 'frames'.padStart(7), 'values'.padStart(8),
                  'exact'.padStart(8), 'mism'.padStart(7), 'max|diff|'.padStart(12),
                  'reset replay'].join(' ');
  console.log(header);
  console.log('-'.repeat(header.length));

  const results = [];
  let failed = Boolean(errors.length || hashErrors.length);

  for (const entry of manifest.sets) {
    const frames = entry.frames;
    const inputs = readF32(path.join(options.pack, entry.input_file));
    const expected = readF32(path.join(options.pack, entry.output_file));
    if (inputs.length !== frames * INPUT_LEN || expected.length !== frames * OUTPUT_LEN) {
      errors.push(`${entry.name}: file sizes do not match the manifest`);
      failed = true;
      continue;
    }

    const run = () => {
      const values = new Float32Array(frames * OUTPUT_LEN);
      const out = new Float32Array(e.memory.buffer, outPtr, OUTPUT_LEN);
      for (let f = 0; f < frames; f++) {
        new Float32Array(e.memory.buffer, inPtr, INPUT_LEN)
          .set(inputs.subarray(f * INPUT_LEN, (f + 1) * INPUT_LEN));
        const rc = e.cf_voice_process(handle, inPtr, outPtr);
        if (rc !== 0) throw new Error(`${entry.name} frame ${f}: cf_voice_process returned ${rc}`);
        values.set(out, f * OUTPUT_LEN);
      }
      return values;
    };

    const resetRc = e.cf_voice_reset(handle);
    if (resetRc !== 0) fail(`${entry.name}: cf_voice_reset returned ${resetRc}`);

    const first = run();
    const { exact, maxDiff } = compare(first, expected);
    const result = {
      name: entry.name,
      frames,
      values: expected.length,
      exact,
      mismatches: expected.length - exact,
      max_abs_diff: maxDiff,
      reset_replay: null,
    };

    if (options.reset) {
      if (e.cf_voice_reset(handle) !== 0) fail(`${entry.name}: reset before replay failed`);
      const replay = run();
      const again = compare(replay, expected);
      result.reset_replay = {
        exact: again.exact,
        mismatches: expected.length - again.exact,
        max_abs_diff: again.maxDiff,
      };
    }

    if (result.mismatches || (result.reset_replay && result.reset_replay.mismatches)) failed = true;
    results.push(result);

    const replayText = result.reset_replay === null
      ? 'off'
      : result.reset_replay.mismatches === 0
        ? 'exact'
        : `${result.reset_replay.mismatches} MISMATCH`;
    console.log([
      result.name.padEnd(18),
      String(result.frames).padStart(7),
      String(result.values).padStart(8),
      String(result.exact).padStart(8),
      String(result.mismatches).padStart(7),
      result.max_abs_diff.toExponential(3).padStart(12),
      replayText,
    ].join(' '));
  }

  const values = results.reduce((sum, r) => sum + r.values, 0);
  const exact = results.reduce((sum, r) => sum + r.exact, 0);
  const maxDiff = results.reduce((max, r) => Math.max(max, r.max_abs_diff), 0);
  const replayOk = results.every(r => r.reset_replay === null || r.reset_replay.mismatches === 0);

  e.cf_voice_free(handle);
  e.cf_free(inPtr);
  e.cf_free(outPtr);

  console.log();
  console.log(`checks: hashes=${hashErrors.length ? 'FAIL' : 'ok'} queries=${errors.length ? 'FAIL' : 'ok'} ` +
              `exact=${exact}/${values} reset=${replayOk ? 'ok' : 'FAIL'}`);
  for (const message of [...errors, ...hashErrors]) console.log(`error: ${message}`);
  console.log();
  console.log(`RESULT: ${failed ? 'FAIL' : 'PASS'}`);

  if (options.json) {
    writeFileSync(options.json, JSON.stringify({
      format: 'cleanfield-conformance-result/1',
      verifier: 'conformance/verify.mjs',
      pass: !failed,
      wasm: options.wasm,
      model: options.model || 'embedded',
      backend: fmaSoft === 1 ? 'scalar-soft-fma' : 'scalar',
      workspace_bytes: e.cf_voice_workspace_bytes ? e.cf_voice_workspace_bytes() : null,
      values,
      exact,
      mismatches: values - exact,
      max_abs_diff: maxDiff,
      reset_replay_ok: replayOk,
      sets: results,
      errors: [...errors, ...hashErrors],
    }, null, 2) + '\n');
    console.log(`wrote ${options.json}`);
  }

  process.exitCode = failed ? 1 : 0;
}

try {
  main();
} catch (error) {
  fail(error && error.stack ? error.stack : String(error));
}
