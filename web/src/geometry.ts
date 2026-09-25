import type { Box, Point } from './types';

export function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

export const centre = (b: Box): Point => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });

/** Where a player touches the ground: the point we project onto the pitch. */
export const foot = (b: Box): Point => ({ x: b.x + b.w / 2, y: b.y + b.h });

export const dist = (a: Point, b: Point): number => Math.hypot(a.x - b.x, a.y - b.y);
