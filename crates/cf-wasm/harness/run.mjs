#!/usr/bin/env node
// CleanField WASM verification harness.
//
// Loads the wasm module, replays the flat native-oracle dumps produced by
// `cf-cli dump`, and compares every output (and every optional intermediate
// buffer) bit for bit.  No npm dependencies: only Node built-ins.
//
// Usage:
//   node run.mjs --kind voice --wasm <cf_wasm.wasm> --dump <oracle.cfd>
//                [--dump ...] [--model <container.cfwm>] [--limit N]
//                [--embedded-check N] [--bench N]
//
// The wasm module ships the voice ABI only.  Echo cancellation lives in the
// separate `cf-echo` / `cf-echo-pipeline` crates and is not part of this
// harness.
//
// `--bench N` measures inference only: the first dump supplies N input
// frames, the module is warmed up, and the timer covers only
// `cf_voice_process` calls (no output comparison).  Reports ms/frame and fps
// for three trials and the best of them.
//
// Exit code 0 only when the software FMA self-test passes and every compared
// value matches.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const HERE = new URL('.', import.meta.url);
const DEFAULT_WASM = new URL('../../../target/wasm32-unknown-unknown/release/cf_wasm.wasm', HERE);
const DEFAULT_MODEL = new URL('../../../models/voice-default.cfwm', HERE);

function usage() {
  console.error(
    'usage: node run.mjs --kind voice --wasm <cf_wasm.wasm> --dump <oracle.cfd>\n' +
      '                    [--dump ...] [--model <container.cfwm>] [--limit N]\n' +
      '                    [--embedded-check N] [--bench N]'
  );
}

function parseArgs(argv) {
  const out = {
    kind: 'voice',
    wasm: null,
    model: null,
    dumps: [],
    limit: 0,
    embeddedCheck: 0,
    bench: 0,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const need = () => {
      if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`);
      return argv[++i];
    };
    switch (a) {
      case '--kind':
        out.kind = need();
        break;
      case '--wasm':
        out.wasm = need();
        break;
      case '--model':
        out.model = need();
        break;
      case '--dump':
        out.dumps.push(need());
        break;
      case '--limit':
        out.limit = Number.parseInt(need(), 10);
        break;
      case '--embedded-check':
        out.embeddedCheck = Number.parseInt(need(), 10);
        break;
      case '--bench':
        out.bench = Number.parseInt(need(), 10);
        break;
      case '--help':
      case '-h':
        usage();
        process.exit(0);
        break;
      default:
        throw new Error(`unknown argument: ${a}`);
    }
  }
  if (out.kind !== 'voice') {
    throw new Error(`unknown --kind ${out.kind} (this module ships the voice ABI only)`);
  }
  if (!out.wasm) out.wasm = fileURLToPath(DEFAULT_WASM);
  if (!out.model) {
    out.model = fileURLToPath(DEFAULT_MODEL);
  }
  if (out.dumps.length === 0) throw new Error('at least one --dump is required');
  return out;
}

// ---------------------------------------------------------------------------
// wasm loading and heap helpers
// ---------------------------------------------------------------------------
function instantiate(wasmPath) {
  const bytes = readFileSync(wasmPath);
  const module = new WebAssembly.Module(bytes);
  const imports = WebAssembly.Module.imports(module);
  if (imports.length !== 0) {
    throw new Error(
      `wasm module has ${imports.length} import(s): ` +
        imports.map((i) => `${i.module}.${i.name}`).join(', ')
    );
  }
  const instance = new WebAssembly.Instance(module, {});
  return { bytes, instance, e: instance.exports };
}

function makeHeap(e) {
  // Views are created on demand: any allocation can grow (and detach) the
  // memory buffer, so a cached view would be stale.
  return {
    u8: () => new Uint8Array(e.memory.buffer),
    f32: (ptr, n) => new Float32Array(e.memory.buffer, ptr, n),
    u32: (ptr, n) => new Uint32Array(e.memory.buffer, ptr, n),
    i16: (ptr, n) => new Int16Array(e.memory.buffer, ptr, n),
  };
}

function readCString(e, ptr, len) {
  return new TextDecoder().decode(new Uint8Array(e.memory.buffer, ptr, len));
}

// ---------------------------------------------------------------------------
// oracle dump reader (CFD1)
// ---------------------------------------------------------------------------
function readOracleDump(path, limit) {
  const raw = readFileSync(path);
  // Copy into an offset-zero ArrayBuffer so the typed-array offsets below are
  // guaranteed to be aligned.
  const bytes = new Uint8Array(raw.byteLength);
  bytes.set(raw);
  const dv = new DataView(bytes.buffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== 'CFD1') throw new Error(`${path}: bad oracle magic ${JSON.stringify(magic)}`);
  const version = dv.getUint16(4, true);
  if (version !== 1) throw new Error(`${path}: unsupported oracle version ${version}`);
  const flags = dv.getUint16(6, true);
  const declaredFrames = dv.getUint32(8, true);
  const inputLen = dv.getUint32(12, true);
  const outputLen = dv.getUint32(16, true);
  const layerCount = dv.getUint32(20, true);
  if (layerCount > 200) throw new Error(`${path}: implausible layer count ${layerCount}`);

  let at = 32;
  const layers = [];
  for (let i = 0; i < layerCount; i++) {
    if (at + 8 > bytes.length) throw new Error(`${path}: truncated layer table`);
    const kind = bytes[at];
    const nameLen = bytes[at + 1];
    const len = dv.getUint32(at + 4, true);
    at += 8;
    if (at + nameLen > bytes.length) throw new Error(`${path}: truncated layer name`);
    const name = new TextDecoder().decode(bytes.subarray(at, at + nameLen));
    at += nameLen;
    const pad = (4 - (nameLen % 4)) % 4;
    at += pad;
    if (kind !== 0 && kind !== 1) throw new Error(`${path}: bad layer kind ${kind}`);
    layers.push({ name, kind, len, elemSize: kind === 0 ? 4 : 2 });
  }
  const hasLayers = (flags & 1) !== 0;
  if (hasLayers !== layerCount > 0) {
    throw new Error(`${path}: layer flag/count disagree (flags=${flags}, count=${layerCount})`);
  }

  const frames = Math.min(declaredFrames, limit > 0 ? limit : declaredFrames);
  const framesOffset = at;
  // The dump always lays out all declared frames first; a --limit read only
  // consumes a prefix, so the section offsets use the declared frame count.
  const outputsOffset = framesOffset + declaredFrames * inputLen * 4;
  const layersOffset = outputsOffset + declaredFrames * outputLen * 4;
  const layerFrameBytes = layers.reduce((n, l) => n + l.len * l.elemSize, 0);
  const total = layersOffset + (hasLayers ? declaredFrames * layerFrameBytes : 0);
  if (total > bytes.length) {
    throw new Error(`${path}: payload shorter than declared (${total} > ${bytes.length})`);
  }

  return {
    path,
    flags,
    declaredFrames,
    frames,
    inputLen,
    outputLen,
    layers,
    layerFrameBytes,
    framesOffset,
    outputsOffset,
    layersOffset,
    bytes,
  };
}

// ---------------------------------------------------------------------------
// comparison
// ---------------------------------------------------------------------------
function compareU32(dump, frameIndex, offset, count, got, stats, label) {
  const want = new Uint32Array(dump.bytes.buffer, offset, count);
  let bad = 0;
  for (let i = 0; i < count; i++) {
    stats.compared++;
    if (got[i] !== want[i]) {
      bad++;
      stats.mismatch++;
      if (stats.examples.length < 4) {
        stats.examples.push(
          `frame ${frameIndex} ${label}[${i}]: wasm 0x${got[i].toString(16).padStart(8, '0')} ` +
            `native 0x${want[i].toString(16).padStart(8, '0')}`
        );
      }
    }
  }
  return bad;
}

function compareI16(dump, frameIndex, offset, count, got, stats, label) {
  const want = new Int16Array(dump.bytes.buffer, offset, count);
  let bad = 0;
  for (let i = 0; i < count; i++) {
    stats.compared++;
    if (got[i] !== want[i]) {
      bad++;
      stats.mismatch++;
      if (stats.examples.length < 4) {
        stats.examples.push(
          `frame ${frameIndex} ${label}[${i}]: wasm ${got[i]} native ${want[i]}`
        );
      }
    }
  }
  return bad;
}

function runDump({ e, heap, api, dump, handle, inputLen, outputLen, inPtr, outPtr, collectLayers }) {
  const inView = heap.f32(inPtr, inputLen);
  const outView = heap.u32(outPtr, outputLen);
  const outStats = { compared: 0, mismatch: 0, examples: [] };
  const layerStats = dump.layers.map((l) => ({
    name: l.name,
    compared: 0,
    mismatch: 0,
    examples: [],
  }));
  let layerOffsets = [];
  let acc = 0;
  for (const l of dump.layers) {
    layerOffsets.push(acc);
    acc += l.len * l.elemSize;
  }
  // Each oracle dump was generated from a freshly reset model.
  if (api.reset(handle) !== 0) throw new Error(`reset failed: ${e.cf_last_error()}`);

  const started = process.hrtime.bigint();
  for (let f = 0; f < dump.frames; f++) {
    const src = new Float32Array(
      dump.bytes.buffer,
      dump.framesOffset + f * inputLen * 4,
      inputLen
    );
    inView.set(src);
    const rc = api.process(handle, inPtr, outPtr);
    if (rc !== 0) throw new Error(`process failed at frame ${f}: rc=${rc}`);
    const gotOut = heap.u32(outPtr, outputLen);
    compareU32(
      dump,
      f,
      dump.outputsOffset + f * outputLen * 4,
      outputLen,
      gotOut,
      outStats,
      'out'
    );

    if (collectLayers) {
      for (let li = 0; li < dump.layers.length; li++) {
        const layer = dump.layers[li];
        const stats = layerStats[li];
        const p = e.cf_voice_diag_ptr(handle, li);
        if (!p) throw new Error(`diag ptr ${li} (${layer.name}) is null`);
        const wasmLen = e.cf_voice_diag_len(handle, li);
        if (wasmLen !== layer.len) {
          throw new Error(
            `diag ${li} (${layer.name}): wasm length ${wasmLen} != oracle length ${layer.len}`
          );
        }
        const kind = e.cf_voice_diag_kind(li);
        if (kind !== layer.kind) {
          throw new Error(`diag ${li} (${layer.name}): wasm kind ${kind} != oracle ${layer.kind}`);
        }
        const off =
          dump.layersOffset + f * dump.layerFrameBytes + layerOffsets[li];
        if (layer.kind === 0) {
          compareU32(dump, f, off, layer.len, heap.u32(p, layer.len), stats, layer.name);
        } else {
          compareI16(dump, f, off, layer.len, heap.i16(p, layer.len), stats, layer.name);
        }
      }
    }
  }
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  return { outStats, layerStats, elapsedMs };
}

function printStats(label, stats) {
  const verdict = stats.mismatch === 0 ? 'exact' : 'MISMATCH';
  console.log(
    `    ${label.padEnd(10)} compared ${String(stats.compared).padStart(11)}  ` +
      `mismatch ${String(stats.mismatch).padStart(3)}  ${verdict}`
  );
  for (const ex of stats.examples) console.log(`      ${ex}`);
}

// ---------------------------------------------------------------------------
// inference-only benchmark
// ---------------------------------------------------------------------------
// The timer covers only the exported `process` calls: no dump comparison, no
// typed-array reads after each frame.  Three trials, best reported.
function runBench({ e, heap, api, dump, handle, inputLen, frames, inPtr, outPtr }) {
  const inView = heap.f32(inPtr, inputLen);
  const step = (start, count) => {
    for (let f = start; f < start + count; f++) {
      const src = new Float32Array(
        dump.bytes.buffer,
        dump.framesOffset + f * inputLen * 4,
        inputLen
      );
      inView.set(src);
      const rc = api.process(handle, inPtr, outPtr);
      if (rc !== 0) throw new Error(`process failed at frame ${f}: rc=${rc}`);
    }
  };
  if (api.reset(handle) !== 0) throw new Error('reset failed');
  step(0, Math.min(200, frames)); // JIT / cache warm-up

  const trials = [];
  for (let t = 0; t < 3; t++) {
    if (api.reset(handle) !== 0) throw new Error('reset failed');
    const started = process.hrtime.bigint();
    step(0, frames);
    const ms = Number(process.hrtime.bigint() - started) / 1e6;
    trials.push(ms);
    console.log(
      `    trial ${t + 1}: ${ms.toFixed(1)} ms total, ${(ms / frames).toFixed(3)} ms/frame, ` +
        `${(frames / (ms / 1000)).toFixed(1)} fps`
    );
  }
  const best = Math.min(...trials);
  console.log(
    `    best: ${(best / frames).toFixed(3)} ms/frame, ${(frames / (best / 1000)).toFixed(1)} fps ` +
      `(${frames} frames, inference only)`
  );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
function main() {
  const opts = parseArgs(process.argv.slice(2));
  const { bytes: wasmBytes, e } = instantiate(opts.wasm);
  const heap = makeHeap(e);
  const api = {
    new: e.cf_voice_new,
    reset: e.cf_voice_reset,
    process: e.cf_voice_process,
    free: e.cf_voice_free,
    bytes: e.cf_voice_model_bytes,
    backendPtr: e.cf_voice_backend_name_ptr,
    backendLen: e.cf_voice_backend_name_len,
    imageLen: e.cf_voice_model_image_len,
    workspaceBytes: e.cf_voice_workspace_bytes,
    workspaceAlign: e.cf_voice_workspace_align,
    inputLen: e.cf_voice_input_len,
    outputLen: e.cf_voice_output_len,
    isScalar: e.cf_voice_backend_is_scalar,
    diagNamePtr: e.cf_voice_diag_name_ptr,
    diagNameLen: e.cf_voice_diag_name_len,
  };
  if (!api.new || !api.reset || !api.process || !api.free) {
    throw new Error(`module does not export the ${opts.kind} ABI`);
  }

  console.log(`kind: ${opts.kind}`);
  console.log(`wasm: ${opts.wasm} (${wasmBytes.length} bytes)`);
  console.log(`memory (initial): ${e.memory.buffer.byteLength} bytes`);
  console.log(`model image (embedded): ${api.imageLen()} bytes`);
  console.log(`workspace: ${api.workspaceBytes()} bytes, align ${api.workspaceAlign()}`);
  console.log(`input/output: ${api.inputLen()} / ${api.outputLen()}`);
  if (api.isScalar() !== 1) throw new Error('wasm backend is not scalar');

  const soft = e.cf_fma_is_soft();
  const selfTest = e.cf_fma_self_test() >>> 0;
  const selfCases = e.cf_fma_self_test_count();
  console.log(
    `fma: software=${soft} self-test=${selfTest === 0 ? 'PASS' : `FAIL(${selfTest})`} ` +
      `cases=${selfCases}`
  );
  if (soft !== 1) throw new Error('wasm build kept a non-software FMA path');
  if (selfTest !== 0) throw new Error(`software FMA self-test failed at case ${selfTest - 1}`);
  const probe = e.cf_fma_probe(3.0, 0.1, -0.3) >>> 0;
  console.log(`fma probe fma(3.0, 0.1, -0.3) = 0x${probe.toString(16).padStart(8, '0')}`);
  if (probe !== 0xb2000000) throw new Error('FMA probe did not match the expected bits');

  // Host-supplied model through the allocation API (the documented ABI path).
  const modelBytes = readFileSync(opts.model);
  if (modelBytes.length % 4 !== 0) throw new Error('model length is not a multiple of four');
  const modelWords = modelBytes.length / 4;
  const modelPtr = e.cf_alloc_u32(modelWords);
  if (!modelPtr) throw new Error('cf_alloc_u32 failed');
  heap.u8().set(modelBytes, modelPtr);
  const handle = api.new(modelPtr, modelBytes.length);
  if (!handle) throw new Error(`model new failed: error ${e.cf_last_error()}`);
  console.log(`model: ${opts.model} (${modelBytes.length} bytes, host-supplied)`);
  const modelResident = api.bytes(handle);
  console.log(`model resident: ${modelResident} bytes`);
  const backendPtr = api.backendPtr(handle);
  const backendLen = api.backendLen(handle);
  const backend = backendPtr ? readCString(e, backendPtr, backendLen) : '';
  console.log(`backend: ${backend}`);
  if (backend !== 'scalar') throw new Error(`wasm selected backend ${backend}, expected scalar`);

  const inputLen = api.inputLen();
  const outputLen = api.outputLen();
  const inPtr = e.cf_alloc(inputLen * 4, 4);
  const outPtr = e.cf_alloc(outputLen * 4, 4);
  if (!inPtr || !outPtr) throw new Error('failed to allocate I/O buffers');
  console.log(`heap after load: ${e.memory.buffer.byteLength} bytes`);

  if (opts.bench > 0) {
    const dump = readOracleDump(opts.dumps[0], opts.bench);
    if (dump.inputLen !== inputLen || dump.outputLen !== outputLen) {
      throw new Error(
        `${opts.dumps[0]}: oracle I/O shape ${dump.inputLen}/${dump.outputLen} != wasm ${inputLen}/${outputLen}`
      );
    }
    console.log(`[bench] ${opts.dumps[0]}: frames=${dump.frames}/${dump.declaredFrames}`);
    runBench({
      e,
      heap,
      api,
      dump,
      handle,
      inputLen,
      frames: dump.frames,
      inPtr,
      outPtr,
    });
    api.free(handle);
    e.cf_free(outPtr, outputLen * 4, 4);
    e.cf_free(inPtr, inputLen * 4, 4);
    console.log('');
    console.log('RESULT: bench only (no comparison; run without --bench for exactness)');
    return;
  }

  let totalOut = { compared: 0, mismatch: 0 };
  let totalLayers = { compared: 0, mismatch: 0 };
  let anyLayerDump = false;

  for (const path of opts.dumps) {
    const dump = readOracleDump(path, opts.limit);
    if (dump.inputLen !== inputLen || dump.outputLen !== outputLen) {
      throw new Error(
        `${path}: oracle I/O shape ${dump.inputLen}/${dump.outputLen} != wasm ${inputLen}/${outputLen}`
      );
    }
    // Verify the diagnostic table names/order before using the pointers.
    for (let i = 0; i < dump.layers.length; i++) {
      const p = api.diagNamePtr(i);
      const n = api.diagNameLen(i);
      const name = p ? readCString(e, p, n) : '';
      if (name !== dump.layers[i].name) {
        throw new Error(`${path}: diag ${i} name wasm ${JSON.stringify(name)} != oracle ${dump.layers[i].name}`);
      }
    }
    console.log(
      `[dump] ${path}: frames=${dump.frames}/${dump.declaredFrames}, layers=${dump.layers.length}`
    );
    const result = runDump({
      e,
      heap,
      api,
      dump,
      handle,
      inputLen,
      outputLen,
      inPtr,
      outPtr,
      collectLayers: dump.layers.length > 0,
    });
    printStats('out', result.outStats);
    for (const s of result.layerStats) {
      if (s.mismatch !== 0) printStats(s.name, s);
    }
    let layerCompared = 0;
    let layerMismatch = 0;
    for (const s of result.layerStats) {
      layerCompared += s.compared;
      layerMismatch += s.mismatch;
    }
    totalOut.compared += result.outStats.compared;
    totalOut.mismatch += result.outStats.mismatch;
    totalLayers.compared += layerCompared;
    totalLayers.mismatch += layerMismatch;
    if (dump.layers.length > 0) anyLayerDump = true;
    console.log(
      `    frames ${dump.frames} in ${result.elapsedMs.toFixed(1)} ms ` +
        `(${(dump.frames / (result.elapsedMs / 1000)).toFixed(1)} fps)`
    );
    const mism = result.outStats.mismatch + layerMismatch;
    console.log(`    ${mism === 0 ? 'RESULT: exact' : 'RESULT: MISMATCH'}`);

    if (opts.embeddedCheck > 0 && api.imageLen() === 0) {
      console.log('    embedded model: not compiled into this module (skipped)');
    } else if (opts.embeddedCheck > 0) {
      const n = Math.min(opts.embeddedCheck, dump.frames);
      const emb = api.new(0, 0);
      if (!emb) throw new Error(`embedded model new failed: ${e.cf_last_error()}`);
      const embResult = runDump({
        e,
        heap,
        api,
        dump: { ...dump, frames: n },
        handle: emb,
        inputLen,
        outputLen,
        inPtr,
        outPtr,
        collectLayers: dump.layers.length > 0,
      });
      let embLayers = 0;
      let embLayerBad = 0;
      for (const s of embResult.layerStats) {
        embLayers += s.compared;
        embLayerBad += s.mismatch;
      }
      const bad = embResult.outStats.mismatch + embLayerBad;
      console.log(
        `    embedded model: frames=${n} outputs=${embResult.outStats.compared} ` +
          `layers=${embLayers} mismatch=${bad} ${bad === 0 ? '(exact)' : '(MISMATCH)'}`
      );
      if (bad !== 0) throw new Error('embedded-model run diverged from the native oracle');
      api.free(emb);
    }
  }

  // Model-id selection: the unified ABI must reach the voice model and produce
  // the same outputs as the typed voice ABI on the first dump.
  {
    const dump = readOracleDump(opts.dumps[0], opts.limit);
    const n = Math.min(dump.frames, 5);
    const copyPtr = e.cf_alloc_u32(modelWords);
    if (!copyPtr) throw new Error('cf_alloc_u32 failed for the unified-model check');
    heap.u8().set(modelBytes, copyPtr);
    const unified = e.cf_model_new(1, copyPtr, modelBytes.length);
    if (!unified) throw new Error(`cf_model_new(1) failed: ${e.cf_last_error()}`);
    if (e.cf_model_id(unified) !== 1) throw new Error('cf_model_id did not report 1');
    if (e.cf_model_input_len(unified) !== inputLen) throw new Error('unified input length');
    if (e.cf_model_output_len(unified) !== outputLen) throw new Error('unified output length');
    const inView = heap.f32(inPtr, inputLen);
    const stats = { compared: 0, mismatch: 0, examples: [] };
    if (e.cf_model_reset(unified) !== 0) throw new Error('cf_model_reset failed');
    for (let f = 0; f < n; f++) {
      const src = new Float32Array(
        dump.bytes.buffer,
        dump.framesOffset + f * inputLen * 4,
        inputLen
      );
      inView.set(src);
      if (e.cf_model_process(unified, inPtr, outPtr) !== 0) {
        throw new Error(`cf_model_process failed at frame ${f}`);
      }
      compareU32(
        dump,
        f,
        dump.outputsOffset + f * outputLen * 4,
        outputLen,
        heap.u32(outPtr, outputLen),
        stats,
        'unified'
      );
    }
    e.cf_model_free(unified);
    console.log(
      `    unified model id=1: frames=${n} outputs=${stats.compared} mismatch=${stats.mismatch}`
    );
    if (stats.mismatch !== 0) throw new Error('unified model path diverged');
  }

  api.free(handle);
  e.cf_free(outPtr, outputLen * 4, 4);
  e.cf_free(inPtr, inputLen * 4, 4);

  console.log('');
  console.log(
    `outputs: ${totalOut.compared} compared, ${totalOut.mismatch} mismatch`
  );
  if (anyLayerDump) {
    console.log(
      `intermediates: ${totalLayers.compared} compared, ${totalLayers.mismatch} mismatch`
    );
  }
  console.log(`heap after free: ${e.memory.buffer.byteLength} bytes`);
  if (totalOut.mismatch !== 0 || totalLayers.mismatch !== 0) {
    console.log('RESULT: MISMATCH');
    process.exit(1);
  }
  console.log('RESULT: wasm matches the native scalar oracle bit for bit');
}

try {
  main();
} catch (err) {
  console.error(`error: ${err.message}`);
  process.exit(2);
}
