import { iou } from './geometry';
import type { Detection, Kind } from './types';

/** Describes the exported model; lives next to the .onnx file as meta.json. */
export interface ModelMeta {
  file: string;
  /** Human-readable model name shown in the masthead. */
  label?: string;
  input: number;
  names: string[];
  /** class index -> role in the app; classes not listed are ignored. */
  roles: Record<string, Kind>;
  thresholds?: Partial<Record<Kind, number>>;
}

export const DEFAULT_THRESHOLDS: Record<Kind, number> = {
  player: 0.35,
  goalkeeper: 0.35,
  referee: 0.35,
  ball: 0.12,
};

export interface Letterbox {
  scale: number;
  padX: number;
  padY: number;
}

export function letterbox(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  return { scale, padX: (size - srcW * scale) / 2, padY: (size - srcH * scale) / 2 };
}

/**
 * Decodes a YOLOv8 head of shape [1, 4 + nc, anchors] (channel-major) into
 * detections in source pixels. Only classes with a role are kept.
 */
export function decodeYolo(
  out: Float32Array,
  anchors: number,
  numClasses: number,
  meta: Pick<ModelMeta, 'roles' | 'thresholds'>,
  lb: Letterbox,
): Detection[] {
  const th = { ...DEFAULT_THRESHOLDS, ...meta.thresholds };
  const roleOf = new Map<number, Kind>();
  for (const [k, v] of Object.entries(meta.roles)) roleOf.set(Number(k), v);

  const dets: Detection[] = [];
  for (let a = 0; a < anchors; a++) {
    let best = -1;
    let bestScore = 0;
    for (const c of roleOf.keys()) {
      if (c >= numClasses) continue;
      const s = out[(4 + c) * anchors + a];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best < 0) continue;
    const kind = roleOf.get(best)!;
    if (bestScore < th[kind]) continue;
    const cx = out[a];
    const cy = out[anchors + a];
    const w = out[2 * anchors + a];
    const h = out[3 * anchors + a];
    dets.push({
      kind,
      score: bestScore,
      x: (cx - w / 2 - lb.padX) / lb.scale,
      y: (cy - h / 2 - lb.padY) / lb.scale,
      w: w / lb.scale,
      h: h / lb.scale,
    });
  }
  return nms(dets, 0.5);
}

/** Greedy non-maximum suppression, applied separately per kind. */
export function nms(dets: Detection[], threshold: number): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const d of sorted) {
    if (keep.every((k) => k.kind !== d.kind || iou(k, d) < threshold)) keep.push(d);
  }
  return keep;
}
