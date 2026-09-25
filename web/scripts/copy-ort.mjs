// onnxruntime-web starts its WASM threads from ort-wasm-*.mjs. Serving those
// files as plain static assets (instead of bundling them) keeps the worker
// from importing any of the app's own chunks.
import { cpSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const src = join('node_modules', 'onnxruntime-web', 'dist');
const dst = join('public', 'ort');
mkdirSync(dst, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  cpSync(join(src, f), join(dst, f));
}
