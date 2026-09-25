import { defaultClientConditions, defineConfig } from 'vite';

// Cross-origin isolation lets onnxruntime-web use multi-threaded WASM.
// "credentialless" keeps Google Fonts loading without CORP headers.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  base: './',
  resolve: {
    // Picks onnxruntime-web's non-bundled build, which loads its WASM from
    // ort.env.wasm.wasmPaths (public/ort) instead of from our own chunks.
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  server: { headers: isolation },
  preview: { headers: isolation },
  build: { target: 'es2022', chunkSizeWarningLimit: 1200 },
  test: { environment: 'node', include: ['tests/**/*.test.ts'] },
});
