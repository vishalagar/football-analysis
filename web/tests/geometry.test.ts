import { describe, expect, it } from 'vitest';
import { homography, project, REFERENCES } from '../src/pitch';
import { histDistance, histogram } from '../src/source';
import { decodeYolo, letterbox, nms } from '../src/yolo';
import type { Detection } from '../src/types';

describe('homography', () => {
  it('maps the four clicked points exactly and interpolates between them', () => {
    const img = [{ x: 100, y: 50 }, { x: 900, y: 80 }, { x: 1000, y: 600 }, { x: 20, y: 580 }];
    const H = homography(img, REFERENCES.full.points)!;
    REFERENCES.full.points.forEach((q, i) => {
      const r = project(H, img[i]);
      expect(r.x).toBeCloseTo(q.x, 6);
      expect(r.y).toBeCloseTo(q.y, 6);
    });
  });

  it('rejects collinear points', () => {
    const line = [0, 1, 2, 3].map((i) => ({ x: i * 10, y: i * 10 }));
    expect(homography(line, REFERENCES.full.points)).toBeNull();
  });
});

describe('YOLO decoding', () => {
  it('undoes the letterbox and keeps only mapped classes', () => {
    const anchors = 2, nc = 33;
    const out = new Float32Array((4 + nc) * anchors);
    const set = (ch: number, a: number, v: number) => (out[ch * anchors + a] = v);
    // anchor 0: a person at the centre of the 640 input
    set(0, 0, 320); set(1, 0, 320); set(2, 0, 20); set(3, 0, 60); set(4 + 0, 0, 0.9);
    // anchor 1: a car (class 2) — not mapped, must be dropped
    set(0, 1, 100); set(1, 1, 100); set(2, 1, 50); set(3, 1, 50); set(4 + 2, 1, 0.95);
    const lb = letterbox(1280, 720, 640);
    const dets = decodeYolo(out, anchors, nc, { roles: { 0: 'player', 32: 'ball' } }, lb);
    expect(dets).toHaveLength(1);
    expect(dets[0].kind).toBe('player');
    expect(dets[0].x + dets[0].w / 2).toBeCloseTo(640);
    expect(dets[0].y + dets[0].h / 2).toBeCloseTo(360);
    expect(dets[0].h).toBeCloseTo(120);
  });

  it('suppresses overlapping boxes of the same kind only', () => {
    const d = (x: number, kind: Detection['kind'], score: number): Detection => ({ x, y: 0, w: 10, h: 10, kind, score });
    const kept = nms([d(0, 'player', 0.9), d(1, 'player', 0.8), d(1, 'ball', 0.5)], 0.5);
    expect(kept.map((k) => k.score)).toEqual([0.9, 0.5]);
  });
});

describe('cut detection histogram', () => {
  it('is 0 for identical frames and 1 for opposite ones', () => {
    const black = new Uint8ClampedArray(64).fill(0).map((_, i) => (i % 4 === 3 ? 255 : 0));
    const white = new Uint8ClampedArray(64).fill(255);
    expect(histDistance(histogram(black), histogram(black))).toBe(0);
    expect(histDistance(histogram(black), histogram(white))).toBeCloseTo(1);
  });
});
