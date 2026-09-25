import type * as Ort from 'onnxruntime-web/webgpu';
import { decodeYolo, letterbox, type ModelMeta } from './yolo';
import type { Box, Detection } from './types';

export type Backend = 'webgpu' | 'wasm';

/** Runs the exported YOLO model in the browser on a video frame. */
export class Detector {
  backend: Backend = 'wasm';
  lastMs = 0;
  private ort!: typeof Ort;
  private session!: Ort.InferenceSession;
  private meta!: ModelMeta;
  private canvas = document.createElement('canvas');
  private ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
  private input!: Float32Array;

  async load(base: string, onProgress: (f: number) => void, onStage: (s: string) => void = () => {}): Promise<ModelMeta> {
    onStage('Loading engine…');
    const ort = (this.ort = await import('onnxruntime-web/webgpu'));
    ort.env.wasm.wasmPaths = new URL(`${base}ort/`, location.href).href;
    // Threads need SharedArrayBuffer, which only a cross-origin-isolated page gets.
    ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    this.meta = await (await fetch(`${base}models/meta.json`)).json();
    const bytes = await fetchWithProgress(`${base}models/${this.meta.file}`, (f, mb) => {
      onProgress(f);
      onStage(f ? `Downloading ${Math.round(f * 100)}%` : `Downloading ${mb.toFixed(1)} MB`);
    });
    const size = this.meta.input;
    this.canvas.width = this.canvas.height = size;
    this.input = new Float32Array(3 * size * size);

    for (const ep of ['webgpu', 'wasm'] as Backend[]) {
      if (ep === 'webgpu' && !(await hasGpu())) continue;
      onStage(ep === 'webgpu' ? 'Starting GPU…' : 'Starting CPU engine…');
      try {
        const start = async () => {
          const session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
          // The first run compiles GPU shaders and can take seconds; do it
          // behind the loading bar, not on the first frame of play.
          await this.warmUp(session);
          return session;
        };
        // Some GPU drivers never settle; fall back to WASM rather than hang.
        this.session = await (ep === 'webgpu' ? withTimeout(start(), 30_000) : start());
        this.backend = ep;
        break;
      } catch (err) {
        if (ep === 'wasm') throw err;
      }
    }
    return this.meta;
  }

  private async warmUp(session: Ort.InferenceSession): Promise<void> {
    const size = this.meta.input;
    const feeds = { [session.inputNames[0]]: new this.ort.Tensor('float32', this.input, [1, 3, size, size]) };
    const out = await session.run(feeds);
    for (const t of Object.values(out)) t.dispose();
  }

  /** Detects on the whole frame, or only inside `roi` (zoomed to fill the input). */
  async detect(src: CanvasImageSource, srcW: number, srcH: number, roi?: Box): Promise<Detection[]> {
    const t0 = performance.now();
    const size = this.meta.input;
    const r = roi ?? { x: 0, y: 0, w: srcW, h: srcH };
    const lb = letterbox(r.w, r.h, size);
    const ctx = this.ctx;
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(src, r.x, r.y, r.w, r.h, lb.padX, lb.padY, r.w * lb.scale, r.h * lb.scale);
    const px = ctx.getImageData(0, 0, size, size).data;
    const plane = size * size;
    for (let i = 0, j = 0; i < plane; i++, j += 4) {
      this.input[i] = px[j] / 255;
      this.input[i + plane] = px[j + 1] / 255;
      this.input[i + 2 * plane] = px[j + 2] / 255;
    }
    const feeds = { [this.session.inputNames[0]]: new this.ort.Tensor('float32', this.input, [1, 3, size, size]) };
    const out = (await this.session.run(feeds))[this.session.outputNames[0]];
    const [, channels, anchors] = out.dims as number[];
    const data = (await out.getData()) as Float32Array;
    out.dispose();
    if (!roi) this.lastMs = performance.now() - t0;
    const dets = decodeYolo(data, anchors, channels - 4, this.meta, lb);
    for (const d of dets) { d.x += r.x; d.y += r.y; }
    return dets;
  }
}

/** navigator.gpu can exist with no usable adapter; ORT then takes seconds to fail. */
async function hasGpu(): Promise<boolean> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return !!(await withTimeout(gpu.requestAdapter(), 5_000));
  } catch {
    return false;
  }
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => reject(new Error(`Timed out after ${ms} ms`)), ms);
    p.then(resolve, reject).finally(() => clearTimeout(id));
  });
}

/** Reports the fraction done, or 0 plus megabytes so far when the size is unknown. */
async function fetchWithProgress(url: string, onProgress: (f: number, mb: number) => void): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not load model (${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(total ? got / total : 0, got / 1e6);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  onProgress(1, got / 1e6);
  return buf;
}
