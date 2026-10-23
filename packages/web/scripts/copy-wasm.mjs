// npm pack runs this first: the demo in `web/` owns the wasm, the package just
// ships a copy, so the two can never drift.
import { copyFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, '..', '..', '..', 'web', 'cf_wasm.wasm');
const target = join(here, '..', 'cf_wasm.wasm');

if (!existsSync(source)) {
  console.error(`[cleanfield] prebuilt module missing: ${source}`);
  process.exit(1);
}
copyFileSync(source, target);
console.log(`[cleanfield] copied ${source} -> ${target}`);
