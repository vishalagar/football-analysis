import type { Point } from './types';

export const PITCH = { length: 105, width: 68 };

/** 3×3 homography, row-major, h[8] = 1. */
export type Homography = number[];

/**
 * Solves the 8-unknown DLT system for 4 point pairs by Gaussian elimination.
 * Returns null if the points are degenerate (e.g. three collinear).
 */
export function homography(src: Point[], dst: Point[]): Homography | null {
  if (src.length !== 4 || dst.length !== 4) return null;
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: u, y: v } = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  for (let c = 0; c < 8; c++) {
    let piv = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (Math.abs(A[piv][c]) < 1e-10) return null;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k];
    }
  }
  return [...A.map((row, i) => row[8] / row[i]), 1];
}

export function project(h: Homography, p: Point): Point {
  const w = h[6] * p.x + h[7] * p.y + h[8];
  return { x: (h[0] * p.x + h[1] * p.y + h[2]) / w, y: (h[3] * p.x + h[4] * p.y + h[5]) / w };
}

/**
 * Reference shapes the user can click on. Points run clockwise from the
 * top-left as seen on a broadcast (far side at the top).
 */
export const REFERENCES: Record<string, { label: string; points: Point[] }> = {
  full: {
    label: 'Whole pitch — four corner flags',
    points: [{ x: 0, y: 0 }, { x: 105, y: 0 }, { x: 105, y: 68 }, { x: 0, y: 68 }],
  },
  leftBox: {
    label: 'Left penalty area',
    points: [{ x: 0, y: 13.84 }, { x: 16.5, y: 13.84 }, { x: 16.5, y: 54.16 }, { x: 0, y: 54.16 }],
  },
  rightBox: {
    label: 'Right penalty area',
    points: [{ x: 88.5, y: 13.84 }, { x: 105, y: 13.84 }, { x: 105, y: 54.16 }, { x: 88.5, y: 54.16 }],
  },
  centre: {
    label: 'Halfway line ends + centre-circle sides',
    points: [{ x: 52.5, y: 0 }, { x: 61.65, y: 34 }, { x: 52.5, y: 68 }, { x: 43.35, y: 34 }],
  },
};

export const clampToPitch = (p: Point): Point => ({
  x: Math.min(PITCH.length, Math.max(0, p.x)),
  y: Math.min(PITCH.width, Math.max(0, p.y)),
});
