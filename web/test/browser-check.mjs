#!/usr/bin/env node
/**
 * Optional end-to-end browser check for live mode.
 *
 * Serves the static folder on a free loopback port, launches headless Chrome
 * with a fake microphone, starts live mode through the page, and verifies that
 * the AudioWorklet loads the wasm engine, processes frames continuously, and
 * survives Raw/Denoised and gain changes.  Requires Chrome at the default
 * Windows install path (override with CHROME_PATH).
 *
 * Usage: node test/browser-check.mjs
 */

import http from 'node:http';
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const CHROME =
  process.env.CHROME_PATH ||
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function createServer() {
  return http.createServer(async (request, response) => {
    try {
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
      response.end(body);
    } catch (error) {
      response.writeHead(error && error.code === 'ENOENT' ? 404 : 500);
      response.end();
    }
  });
}

async function main() {
  if (!existsSync(CHROME)) {
    console.error(`Chrome not found at ${CHROME}; set CHROME_PATH to run this check.`);
    process.exitCode = 2;
    return;
  }

  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const port = server.address().port;
  const base = `http://127.0.0.1:${port}`;
  console.log(`serving ${ROOT} at ${base}`);

  const profile = mkdtempSync(path.join(os.tmpdir(), 'cleanfield-browser-'));
  const chrome = spawn(
    CHROME,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${path.resolve(ROOT, '../../local-denoiser/samples/noisy.wav')}`,
      '--autoplay-policy=no-user-gesture-required',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      'about:blank',
    ],
    { stdio: 'ignore' }
  );

  let ws = null;
  let nextId = 0;
  const pending = new Map();
  const pageErrors = [];
  const consoleErrors = [];

  try {
    const activePortFile = path.join(profile, 'DevToolsActivePort');
    let devtoolsPort = 0;
    for (let i = 0; i < 60; i++) {
      if (existsSync(activePortFile)) {
        const text = readFileSync(activePortFile, 'utf8').split(/\r?\n/);
        if (text[0]) {
          devtoolsPort = Number(text[0]);
          break;
        }
      }
      await sleep(150);
    }
    if (!devtoolsPort) throw new Error('Chrome did not expose a DevTools port.');

    let page = null;
    for (let i = 0; i < 40; i++) {
      try {
        const pages = await (await fetch(`http://127.0.0.1:${devtoolsPort}/json/list`)).json();
        page = pages.find((p) => p.type === 'page');
        if (page) break;
      } catch {
        // Retry.
      }
      await sleep(150);
    }
    if (!page) throw new Error('Chrome did not expose a page target.');

    ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve, { once: true });
      ws.addEventListener('error', reject, { once: true });
    });
    ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const entry = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) entry.reject(new Error(JSON.stringify(message.error)));
        else entry.resolve(message.result);
        return;
      }
      if (message.method === 'Runtime.exceptionThrown') {
        pageErrors.push(
          message.params.exceptionDetails.exception?.description ||
            message.params.exceptionDetails.text ||
            'exception'
        );
      }
      if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
        consoleErrors.push(
          message.params.args
            .map((arg) => arg.value ?? arg.description ?? arg.type)
            .join(' ')
        );
      }
      if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
        consoleErrors.push(message.params.entry.text);
      }
    });

    const send = (method, params = {}) =>
      new Promise((resolve, reject) => {
        const id = ++nextId;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
      });

    const evaluate = async (expression, awaitPromise = false) => {
      const result = await send('Runtime.evaluate', {
        expression,
        awaitPromise,
        returnByValue: true,
      });
      if (result.exceptionDetails) {
        throw new Error(
          result.exceptionDetails.exception?.description || result.exceptionDetails.text
        );
      }
      return result.result ? result.result.value : undefined;
    };

    // Sample a spectrogram canvas: non-background pixels and distinct colours.
    const canvasStats = async (id) => {
      const value = await evaluate(`(() => {
        const canvas = document.getElementById(${JSON.stringify(id)});
        if (!canvas) return JSON.stringify({ missing: true });
        const ctx = canvas.getContext('2d');
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const colors = new Set();
        let nonWhite = 0;
        let sum = 0;
        let count = 0;
        for (let i = 0; i < data.length; i += 4 * 53) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          count += 1;
          sum += r + g + b;
          if (r < 248 || g < 248 || b < 248) nonWhite += 1;
          colors.add((r >> 4) + ',' + (g >> 4) + ',' + (b >> 4));
        }
        return JSON.stringify({
          width: canvas.width,
          height: canvas.height,
          samples: count,
          nonWhite,
          mean: count ? sum / (count * 3) : 0,
          colors: colors.size,
        });
      })()`);
      return JSON.parse(value);
    };

    await send('Runtime.enable');
    await send('Log.enable');
    await send('Page.enable');
    await send('DOM.enable');
    await send('Page.navigate', { url: base + '/' });

    for (let i = 0; i < 60; i++) {
      const ready = await evaluate(
        'Boolean(window.__CLEANFIELD__ && document.getElementById("startButton"))'
      );
      if (ready) break;
      await sleep(100);
    }

    const initial = await evaluate(
      'JSON.stringify({tab: window.__CLEANFIELD__.getState().tab, liveHidden: document.getElementById("livePanel").hidden, fileHidden: document.getElementById("filePanel").hidden})'
    );
    const initialParse = JSON.parse(initial);

    console.log('browser: headless Chrome, fake microphone');
    console.log(`initial view: tab=${initialParse.tab}, live hidden=${initialParse.liveHidden}, file hidden=${initialParse.fileHidden}`);

    const controls = JSON.parse(
      await evaluate(`JSON.stringify({
        attenuation: document.getElementById("attenuationSlider").value,
        gain: document.getElementById("gainSlider").value,
        attenuationHint: document.querySelector("#attenuationSlider").closest(".tune-control")
          .querySelector(".tune-hint").textContent,
        gainHint: document.querySelector("#gainSlider").closest(".tune-control")
          .querySelector(".tune-hint").textContent,
        visible: document.querySelector(".tune").getBoundingClientRect().width > 0
      })`)
    );
    console.log(
      `controls: max attenuation=${controls.attenuation} dB (hint "${controls.attenuationHint}"), ` +
        `gain=${controls.gain} dB (hint "${controls.gainHint}"), visible=${controls.visible}`
    );

    const startStarted = Date.now();
    await evaluate('window.__CLEANFIELD__.startLive()', true);

    let started = null;
    for (let i = 0; i < 120; i++) {
      const state = JSON.parse(
        await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())')
      );
      if (state.live && state.live.ready && state.live.frames >= 50) {
        started = state;
        break;
      }
      if (state.liveError) {
        throw new Error(`live mode error: ${state.liveError}`);
      }
      await sleep(200);
    }
    if (!started) throw new Error('live mode did not start within 24 s');
    const startupMs = Date.now() - startStarted;
    const contextRate = started.live.contextRate;
    const framesAtStart = started.live.frames;

    console.log(`startup: ${startupMs} ms to 50 frames; context ${contextRate} Hz`);
    console.log(
      `after startup: frames=${started.live.frames}, frameMs=${started.live.frameMs.toFixed(2)}, ` +
        `RTF=${started.live.rtf.toFixed(3)}, levels in/out=${started.live.levelIn.toFixed(1)}/` +
        `${started.live.levelOut.toFixed(1)} dB`
    );

    // Let it run, then verify the frame counter keeps advancing.
    const before = started.live.frames;
    await sleep(1200);
    let running = JSON.parse(await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())'));
    const framesAdvanced = running.live ? running.live.frames - before : 0;
    console.log(
      `after 1.2 s: frames=${running.live ? running.live.frames : 0}, ` +
        `levels in/out=${running.live ? running.live.levelIn.toFixed(1) : '?'}/` +
        `${running.live ? running.live.levelOut.toFixed(1) : '?'} dB, ` +
        `peaks in/out=${running.live ? running.live.peakIn.toFixed(4) : '?'}/` +
        `${running.live ? running.live.peakOut.toFixed(4) : '?'}`
    );

    // Spectrogram: live columns arrive from the worklet; freeze and clear must
    // operate on the display without stopping the audio path.
    const liveSpec = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    const liveInputCanvas = await canvasStats('specInput');
    const liveOutputCanvas = await canvasStats('specOutput');
    console.log(
      `spectrogram live: ${liveSpec.columns} columns × ${liveSpec.bins} bins, ` +
        `mean byte in/out=${liveSpec.inputMeanByte.toFixed(1)}/` +
        `${liveSpec.outputMeanByte.toFixed(1)}, painted ${liveSpec.paintCount}×`
    );
    console.log(
      `spectrogram live canvas: input ${liveInputCanvas.width}×${liveInputCanvas.height}, ` +
        `${liveInputCanvas.nonWhite} non-white samples, ${liveInputCanvas.colors} colours; ` +
        `output ${liveOutputCanvas.nonWhite} non-white, ${liveOutputCanvas.colors} colours`
    );

    const framesBeforeFreeze = running.live ? running.live.frames : 0;
    await evaluate('document.getElementById("specFreeze").click()');
    const frozenSpec = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    await sleep(400);
    const frozenAfter = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    const freezeHeld = frozenAfter.frozen && frozenAfter.columns === frozenSpec.columns;

    await evaluate('document.getElementById("specClear").click()');
    await sleep(250);
    const clearedSpec = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    const clearedCanvas = await canvasStats('specInput');

    await evaluate('document.getElementById("specFreeze").click()');
    await sleep(600);
    const resumedSpec = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    const liveResumed = resumedSpec.frozen === false && resumedSpec.columns > 0;
    console.log(
      `spectrogram freeze/clear: frozen held=${freezeHeld} (${frozenSpec.columns}→` +
        `${frozenAfter.columns} columns), cleared=${clearedSpec.columns} columns, ` +
        `${clearedCanvas.nonWhite} non-white samples, refilled=${resumedSpec.columns} columns`
    );
    const liveFramesAfter = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())')
    );
    if (liveFramesAfter.live) {
      running = liveFramesAfter;
    }

    // Raw/Denoised must not reset the engine: the frame counter keeps going.
    await evaluate('window.__CLEANFIELD__.setLiveBypass(true)');
    const beforeRaw = running.live ? running.live.frames : 0;
    await sleep(500);
    const rawState = JSON.parse(await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())'));
    const rawAdvanced = rawState.live ? rawState.live.frames - beforeRaw : 0;
    await evaluate('window.__CLEANFIELD__.setLiveBypass(false)');
    await evaluate('window.__CLEANFIELD__.setGainDb(6)');
    await sleep(300);

    // A live attenuation change must reach the worklet without restarting audio.
    const framesBeforeAttenuation = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())')
    ).live.frames;
    await evaluate('window.__CLEANFIELD__.setMaxAttenuationDb(-40)');
    await sleep(500);
    const attenuationState = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())')
    );
    const attenuationApplied =
      attenuationState.params.maxAttenuationDb === -40 &&
      attenuationState.live &&
      attenuationState.live.frames > framesBeforeAttenuation;
    console.log(
      `attenuation change: ${attenuationState.params.maxAttenuationDb} dB applied, ` +
        `+${attenuationState.live ? attenuationState.live.frames - framesBeforeAttenuation : 0} frames`
    );

    const latency = await evaluate(
      'document.getElementById("latencyNote").textContent'
    );

    await evaluate('window.__CLEANFIELD__.stopLive()', true);
    await sleep(300);
    const stopped = JSON.parse(await evaluate('JSON.stringify(window.__CLEANFIELD__.getState())'));
    const meterWidth = await evaluate('document.getElementById("inputMeter").style.width');

    // File mode: load the sample through the file input and run the offline path.
    await evaluate('window.__CLEANFIELD__.selectTab("file")');
    const tuneInFile = await evaluate(
      'Boolean(document.querySelector(".tune").getBoundingClientRect().width)'
    );
    // File parameters apply when Process is pressed: use −40 dB at 0 dB gain.
    await evaluate('window.__CLEANFIELD__.setMaxAttenuationDb(-40)');
    await evaluate('window.__CLEANFIELD__.setGainDb(0)');
    const documentNode = await send('DOM.getDocument');
    const inputNode = await send('DOM.querySelector', {
      nodeId: documentNode.root.nodeId,
      selector: '#fileInput',
    });
    await send('DOM.setFileInputFiles', {
      files: [path.resolve(ROOT, '../../local-denoiser/samples/noisy.wav')],
      nodeId: inputNode.nodeId,
    });
    await sleep(500);
    await evaluate('document.getElementById("processButton").click()');
    let fileDone = null;
    for (let i = 0; i < 120; i++) {
      const fileState = JSON.parse(
        await evaluate(
          'JSON.stringify({resultsHidden: document.getElementById("results").hidden, ' +
            'stats: document.getElementById("stats").textContent, ' +
            'download: document.getElementById("downloadLink").getAttribute("href"), ' +
            'status: document.getElementById("status").textContent, ' +
            'error: document.getElementById("error").hidden ? null : document.getElementById("error").textContent})'
        )
      );
      if (fileState.error) throw new Error(`file mode error: ${fileState.error}`);
      if (!fileState.resultsHidden && fileState.stats) {
        fileDone = fileState;
        break;
      }
      await sleep(250);
    }
    if (!fileDone) throw new Error('file mode did not finish within 30 s');
    console.log(`file mode: ${fileDone.stats}; download=${String(fileDone.download).slice(0, 20)}…`);

    const fileSpec = JSON.parse(
      await evaluate('JSON.stringify(window.__CLEANFIELD__.spectrogram())')
    );
    const fileInputCanvas = await canvasStats('specInput');
    const fileOutputCanvas = await canvasStats('specOutput');
    console.log(
      `spectrogram file: ${fileSpec.columns} columns × ${fileSpec.bins} bins, ` +
        `mean byte in/out=${fileSpec.inputMeanByte.toFixed(1)}/` +
        `${fileSpec.outputMeanByte.toFixed(1)}; input canvas ${fileInputCanvas.nonWhite} ` +
        `non-white, output canvas ${fileOutputCanvas.nonWhite} non-white`
    );

    const specFramesAdvance = running.live ? running.live.frames - framesBeforeFreeze : 0;
    const liveCanvasOk =
      liveInputCanvas.nonWhite > 50 &&
      liveInputCanvas.colors >= 3 &&
      liveOutputCanvas.nonWhite > 50 &&
      liveOutputCanvas.colors >= 3;
    const fileCanvasOk =
      fileInputCanvas.nonWhite > 50 &&
      fileInputCanvas.colors >= 3 &&
      fileOutputCanvas.nonWhite > 50 &&
      fileOutputCanvas.colors >= 3;

    const problems = [];
    if (initialParse.tab !== 'live') problems.push('Live is not the default tab');
    if (initialParse.liveHidden || !initialParse.fileHidden) problems.push('tab panels are swapped');
    if (controls.attenuation !== '-80' || controls.gain !== '0') {
      problems.push(
        `parameter defaults are ${controls.attenuation} dB / ${controls.gain} dB`
      );
    }
    if (!controls.visible) problems.push('denoise parameter row is not visible');
    if (!controls.attenuationHint.includes('−80…0 dB') || !controls.gainHint.includes('−12…+12 dB')) {
      problems.push('parameter range/default hints are missing');
    }
    if (!attenuationApplied) problems.push('live attenuation change did not reach the worklet');
    if (started.liveError) problems.push(`live error: ${started.liveError}`);
    if (Math.abs(contextRate - 16000) > 1) problems.push(`context rate is ${contextRate}, expected 16000`);
    if (!(started.live.frameMs > 0)) problems.push('frame timing was not reported');
    if (!(started.live.rtf > 0 && started.live.rtf < 1)) problems.push(`RTF out of range: ${started.live.rtf}`);
    if (framesAdvanced < 50) problems.push(`frames advanced by only ${framesAdvanced} in 1.2 s`);
    if (rawAdvanced < 20) problems.push(`Raw mode stalled the engine (${rawAdvanced} frames in 0.5 s)`);
    if (stopped.live !== null) problems.push('Stop left the live session active');
    if (!/Stopped|Idle/.test(stopped.liveStatus)) problems.push(`unexpected stopped status: ${stopped.liveStatus}`);
    if (meterWidth && meterWidth !== '0%' && meterWidth !== '0px') problems.push(`meter not reset: ${meterWidth}`);
    if (!fileDone.stats.includes('RTF') || !fileDone.stats.includes('frames')) {
      problems.push(`file result stats missing: ${fileDone.stats}`);
    }
    if (!fileDone.stats.includes('Max attenuation −40 dB')) {
      problems.push(`file result did not use the −40 dB setting: ${fileDone.stats}`);
    }
    if (!tuneInFile) problems.push('denoise parameter row is hidden in the File tab');
    if (!String(fileDone.download).startsWith('blob:')) problems.push('file download link is not a blob URL');
    if (!/Done/.test(fileDone.status)) problems.push(`unexpected file status: ${fileDone.status}`);
    if (liveSpec.mode !== 'live') problems.push(`live spectrogram mode is ${liveSpec.mode}`);
    if (liveSpec.columns < 40) problems.push(`live spectrogram has only ${liveSpec.columns} columns`);
    if (liveSpec.bins !== 513) problems.push(`live spectrogram has ${liveSpec.bins} bins`);
    if (!liveCanvasOk) problems.push('live spectrogram canvas is blank or monochrome');
    if (!freezeHeld) problems.push('Freeze did not hold the live spectrogram');
    if (clearedSpec.columns !== 0) problems.push(`Clear left ${clearedSpec.columns} columns`);
    if (clearedCanvas.nonWhite > 0) problems.push('Clear left pixels on the canvas');
    if (!liveResumed) problems.push('Unfreeze did not resume the live spectrogram');
    if (specFramesAdvance < 40) {
      problems.push(`audio stalled during freeze/clear (${specFramesAdvance} frames)`);
    }
    if (fileSpec.mode !== 'file') problems.push(`file spectrogram mode is ${fileSpec.mode}`);
    if (fileSpec.columns < 100) problems.push(`file spectrogram has only ${fileSpec.columns} columns`);
    if (!(fileSpec.outputMeanByte < fileSpec.inputMeanByte)) {
      problems.push(
        `file output spectrogram is not lower than input ` +
          `(${fileSpec.outputMeanByte.toFixed(1)} vs ${fileSpec.inputMeanByte.toFixed(1)})`
      );
    }
    if (!fileCanvasOk) problems.push('file spectrogram canvas is blank or monochrome');
    if (pageErrors.length) problems.push(`page exceptions: ${pageErrors.join(' | ')}`);
    if (consoleErrors.length) problems.push(`console errors: ${consoleErrors.join(' | ')}`);

    console.log(
      `run: +${framesAdvanced} frames in 1.2 s; Raw +${rawAdvanced} frames in 0.5 s; ` +
        `stop state=${stopped.liveStatus}`
    );
    console.log(`latency note: ${latency}`);
    console.log(
      `checks: ${problems.length === 0 ? 'all ok' : problems.join('; ')}`
    );

    const verdict = problems.length === 0 ? 'PASS' : 'FAIL';
    console.log(
      'RESULT ' +
        JSON.stringify({
          contextRate,
          startupMs,
          frames: started.live.frames,
          frameMs: started.live.frameMs,
          rtf: started.live.rtf,
          framesIn1200ms: framesAdvanced,
          rawFramesIn500ms: rawAdvanced,
          controls: {
            maxAttenuationDb: Number(controls.attenuation),
            gain: Number(controls.gain),
            attenuationApplied,
            tuneInFile,
          },
          spectrogram: {
            liveColumns: liveSpec.columns,
            liveBins: liveSpec.bins,
            freezeHeld,
            clearedColumns: clearedSpec.columns,
            resumedColumns: resumedSpec.columns,
            audioFramesDuringFreeze: specFramesAdvance,
            fileColumns: fileSpec.columns,
            fileInputMeanByte: fileSpec.inputMeanByte,
            fileOutputMeanByte: fileSpec.outputMeanByte,
            liveInputCanvas,
            liveOutputCanvas,
            fileOutputCanvas,
          },
          pageErrors,
          consoleErrors,
          verdict,
        })
    );
    if (verdict !== 'PASS') process.exitCode = 1;
  } catch (error) {
    console.error(`error: ${error.stack || error.message || error}`);
    if (pageErrors.length) console.error(`page exceptions: ${pageErrors.join('\n')}`);
    if (consoleErrors.length) console.error(`console errors: ${consoleErrors.join('\n')}`);
    process.exitCode = 1;
  } finally {
    try {
      if (ws) ws.close();
    } catch {
      // Ignore.
    }
    chrome.kill();
    await new Promise((resolve) => {
      server.close(resolve);
      if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
      setTimeout(resolve, 1500);
    });
    try {
      rmSync(profile, { recursive: true, force: true });
    } catch {
      // Ignore.
    }
    console.log(`server ${base} stopped`);
  }
}

main();
