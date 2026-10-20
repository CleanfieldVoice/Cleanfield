#!/usr/bin/env node
/**
 * Fetch the prebuilt native addon for this platform from the GitHub Release
 * that matches the package version.
 *
 * Resolution order in `index.js` is: explicit path, `prebuilds/`, platform
 * package, local build.  This script only fills `prebuilds/`; it never fails an
 * install.  Air-gapped machines can set `CLEANFIELD_SKIP_DOWNLOAD=1` (or pass
 * `--ignore-scripts`) and provide the addon themselves.
 *
 * Integrity: when the release publishes `SHA256SUMS`, the downloaded binary is
 * verified against it before it is moved into place.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.join(here, '..');
const manifest = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));

const SUPPORTED = new Set([
  'win32-x64',
  'win32-arm64',
  'darwin-x64',
  'darwin-arm64',
  'linux-x64',
  'linux-arm64',
]);

function skip(reason) {
  console.log(`[cleanfield] prebuilt addon not downloaded: ${reason}`);
  process.exit(0);
}

function warn(message) {
  console.warn(`[cleanfield] ${message}`);
  console.warn(
    '[cleanfield] The JavaScript package is installed, but the native addon is missing. ' +
      'Build it with `cargo build --release -p cf-node`, copy it to ' +
      `prebuilds/${process.platform}-${process.arch}/cf-node.node, or set CLEANFIELD_NODE_BINDING.`
  );
  process.exit(0);
}

if (process.env.CLEANFIELD_SKIP_DOWNLOAD === '1') skip('CLEANFIELD_SKIP_DOWNLOAD=1');

const tag = `${process.platform}-${process.arch}`;
if (!SUPPORTED.has(tag)) skip(`unsupported platform ${tag}`);

const asset =
  process.platform === 'darwin' ? 'cf-node-darwin-universal.node' : `cf-node-${tag}.node`;
const targetDir = path.join(packageRoot, 'prebuilds', tag);
const target = path.join(targetDir, 'cf-node.node');

if (fs.existsSync(target)) skip(`already present at ${path.relative(packageRoot, target)}`);

// A workspace build is enough; do not touch the network for local development.
const localCandidates = [
  path.join(packageRoot, 'build', 'Release', 'cf_node.node'),
  path.join(packageRoot, '..', '..', 'target', 'release', 'cf_node.dll'),
  path.join(packageRoot, '..', '..', 'target', 'release', 'libcf_node.so'),
  path.join(packageRoot, '..', '..', 'target', 'release', 'libcf_node.dylib'),
];
if (localCandidates.some((candidate) => fs.existsSync(candidate))) {
  skip('a local Cargo build is present');
}

const baseUrl =
  process.env.CLEANFIELD_RELEASE_BASE_URL ||
  (manifest.cleanfield && manifest.cleanfield.releaseBaseUrl);
if (!baseUrl) skip('no release base URL is configured');

const releaseUrl = `${baseUrl.replace(/\/+$/, '')}/v${manifest.version}`;
const assetUrl = `${releaseUrl}/${asset}`;

async function download(url) {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

async function main() {
  let bytes;
  try {
    bytes = await download(assetUrl);
  } catch (error) {
    warn(`could not download ${assetUrl}: ${error.message}`);
    return;
  }

  try {
    const sums = (await download(`${releaseUrl}/SHA256SUMS`)).toString('utf8');
    const expected = sums
      .split(/\r?\n/)
      .map((line) => line.trim().split(/\s+/))
      .find((parts) => parts.length >= 2 && parts[parts.length - 1] === asset);
    if (expected) {
      const actual = crypto.createHash('sha256').update(bytes).digest('hex');
      if (actual !== expected[0].toLowerCase()) {
        warn(`checksum mismatch for ${asset}: expected ${expected[0]}, got ${actual}`);
        return;
      }
      console.log(`[cleanfield] SHA256 verified: ${actual}`);
    } else {
      console.warn(`[cleanfield] ${asset} is not listed in SHA256SUMS; installing unverified`);
    }
  } catch (error) {
    console.warn(`[cleanfield] could not verify SHA256SUMS: ${error.message}`);
  }

  fs.mkdirSync(targetDir, { recursive: true });
  const temporary = `${target}.download`;
  fs.writeFileSync(temporary, bytes);
  fs.renameSync(temporary, target);
  console.log(
    `[cleanfield] installed ${asset} (${(bytes.length / 1024).toFixed(1)} KiB) into prebuilds/${tag}/`
  );
}

await main();
