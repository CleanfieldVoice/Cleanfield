'use strict';

/**
 * Local smoke test for the Node addon.  Runs the real engine through the
 * public JavaScript surface:
 *
 *   node packages/node/test/smoke.js
 *
 * It does not need the package to be installed: the loader finds the workspace
 * Cargo build directly.  Exits non-zero on the first failure.
 */

const assert = require('node:assert');
const path = require('node:path');

const { createVoice } = require('../index.js');

const MODEL = path.join(__dirname, '..', '..', '..', 'models', 'voice-default.cfwm');
const INPUT_LEN = 513;
const OUTPUT_LEN = 1026;

function frame(seed) {
  const values = new Float32Array(INPUT_LEN);
  for (let i = 0; i < INPUT_LEN; i++) {
    values[i] = Math.fround(((seed * 37 + i * 13) % 211) * 0.01 - 1.0);
  }
  return values;
}

function bits(values) {
  return Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString('hex');
}

const checks = [];
function check(name, fn) {
  try {
    fn();
    checks.push(`ok   ${name}`);
  } catch (error) {
    checks.push(`FAIL ${name}: ${error && error.message ? error.message : error}`);
    throw error;
  }
}

const voice = createVoice();

check('backend is scalar or avx2', () => {
  assert.ok(voice.backend === 'scalar' || voice.backend === 'avx2', `backend=${voice.backend}`);
});

const first = voice.process(frame(1));
check('process returns Float32Array(1026)', () => {
  assert.ok(first instanceof Float32Array, 'not a Float32Array');
  assert.strictEqual(first.length, OUTPUT_LEN);
  assert.ok(first.every(Number.isFinite), 'non-finite output');
});

check('reset restores the initial state bit for bit', () => {
  voice.reset();
  const again = voice.process(frame(1));
  assert.strictEqual(bits(again), bits(first));
});

check('consecutive frames advance the engine state', () => {
  voice.reset();
  voice.process(frame(2));
  const second = voice.process(frame(3));
  voice.reset();
  const direct = voice.process(frame(3));
  assert.notStrictEqual(bits(second), bits(direct));
});

check('explicit modelPath and backend work', () => {
  const scalar = createVoice({ modelPath: MODEL, backend: 'scalar' });
  assert.strictEqual(scalar.backend, 'scalar');
  const out = scalar.process(frame(4));
  assert.strictEqual(out.length, OUTPUT_LEN);
  scalar.close();
});

check('two handles are independent', () => {
  const a = createVoice({ backend: 'scalar' });
  const b = createVoice({ backend: 'scalar' });
  const outA = a.process(frame(5));
  a.reset();
  const outB = b.process(frame(5));
  assert.strictEqual(bits(outA), bits(outB));
  a.close();
  b.close();
});

check('invalid backend is rejected', () => {
  assert.throws(() => createVoice({ backend: 'cuda' }), TypeError);
});

check('missing model file is reported', () => {
  assert.throws(() => createVoice({ modelPath: path.join(__dirname, 'nope.cfwm') }), /does not exist/);
});

check('wrong frame length is rejected', () => {
  assert.throws(() => voice.process(new Float32Array(10)), TypeError);
});

check('non-Float32Array input is rejected', () => {
  assert.throws(() => voice.process([0, 1, 2]), TypeError);
});

check('close is idempotent and blocks further use', () => {
  voice.close();
  voice.close();
  assert.throws(() => voice.process(frame(6)), /closed/);
});

console.log(checks.join('\n'));
console.log(`\nPASS: ${checks.length} checks (backend ${voice.backend})`);
