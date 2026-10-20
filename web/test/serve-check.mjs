#!/usr/bin/env node
/**
 * Static-host check: serve the web folder on a free loopback port, GET the
 * page, the worklet processor and the wasm module, verify 200 + Content-Type
 * + Content-Length, print the deployable payload size, then stop the server.
 *
 * Usage: node serve-check.mjs
 */

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.css': 'text/css; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const EXPECTED = {
  '/': { type: 'text/html', label: 'page' },
  '/index.html': { type: 'text/html', label: 'page' },
  '/cf_wasm.wasm': { type: 'application/wasm', label: 'engine' },
  '/worklet-processor.js': { type: 'text/javascript', label: 'worklet' },
  '/app.js': { type: 'text/javascript', label: 'controller' },
  '/dsp.js': { type: 'text/javascript', label: 'offline DSP' },
  '/live-dsp.js': { type: 'text/javascript', label: 'live DSP' },
  '/spectrogram-core.js': { type: 'text/javascript', label: 'spectrogram core' },
  '/spectrogram-view.js': { type: 'text/javascript', label: 'spectrogram view' },
  '/worker.js': { type: 'text/javascript', label: 'file worker' },
};

// The echo feature was removed from the demo.  These paths must not resolve,
// so a stale deployment that still ships the old files fails this check.
const FORBIDDEN = [
  '/echo-dsp.js',
  '/echo-live.js',
  '/echo-default.cfwm',
];

const PAYLOAD_FILES = [
  'index.html',
  'app.js',
  'dsp.js',
  'live-dsp.js',
  'spectrogram-core.js',
  'spectrogram-view.js',
  'worker.js',
  'worklet-processor.js',
  'cf_wasm.wasm',
];

const server = http.createServer(async (request, response) => {
  try {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' });
      response.end('method not allowed');
      return;
    }
    const requestUrl = new URL(request.url, 'http://127.0.0.1');
    let pathname = decodeURIComponent(requestUrl.pathname);
    if (pathname.endsWith('/')) pathname += 'index.html';
    const filePath = path.resolve(ROOT, '.' + pathname);
    const relative = path.relative(ROOT, filePath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      response.writeHead(403);
      response.end('forbidden');
      return;
    }
    const body = await readFile(filePath);
    response.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
      'Content-Length': body.byteLength,
      'Cache-Control': 'no-store',
    });
    response.end(request.method === 'HEAD' ? undefined : body);
  } catch (error) {
    const status = error && error.code === 'ENOENT' ? 404 : 500;
    response.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
    response.end(status === 404 ? 'not found' : 'server error');
  }
});

await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});

const address = server.address();
const base = `http://127.0.0.1:${address.port}`;
console.log(`serving ${ROOT} at ${base}`);

let failed = false;
const sizes = new Map();
for (const [route, expectation] of Object.entries(EXPECTED)) {
  const response = await fetch(base + route);
  const body = Buffer.from(await response.arrayBuffer());
  const contentLength = Number(response.headers.get('content-length'));
  const contentType = response.headers.get('content-type') || '';
  const lengthOk =
    Number.isFinite(contentLength) && contentLength === body.byteLength && body.byteLength > 0;
  const typeOk = contentType.toLowerCase().startsWith(expectation.type);
  const ok = response.status === 200 && lengthOk && typeOk;
  if (!ok) failed = true;
  if (body.byteLength > 0) sizes.set(route, body.byteLength);
  console.log(
    `${route}: status=${response.status} bytes=${body.byteLength} ` +
      `type=${contentType || '(none)'} expect=${expectation.type} ${ok ? 'OK' : 'FAIL'}`
  );
}

let total = 0;
console.log('');
console.log('removed echo routes (must not be served):');
for (const route of FORBIDDEN) {
  const response = await fetch(base + route);
  const body = Buffer.from(await response.arrayBuffer());
  const ok = response.status === 404;
  if (!ok) failed = true;
  console.log(`${route}: status=${response.status} bytes=${body.byteLength} ${ok ? 'absent OK' : 'FAIL (still served)'}`);
}

console.log('');
console.log('deployable payload:');
for (const name of PAYLOAD_FILES) {
  const bytes = sizes.get('/' + name) ?? 0;
  total += bytes;
  console.log(`  ${name.padEnd(22)} ${String(bytes).padStart(9)} bytes`);
}
console.log(`  ${'TOTAL'.padEnd(22)} ${String(total).padStart(9)} bytes (${(total / 1024 / 1024).toFixed(3)} MiB)`);

await new Promise((resolve) => {
  server.close(resolve);
  if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
});
console.log(`server ${base} stopped`);
if (failed) process.exitCode = 1;
