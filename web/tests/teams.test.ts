import { describe, expect, it } from 'vitest';
import { isGrass, jerseyColour, kmeans, TeamModel } from '../src/teams';
import type { Rgb } from '../src/types';

const jitter = (c: Rgb, i: number): Rgb => [c[0] + (i % 7) * 3, c[1] + (i % 5) * 3, c[2] + (i % 3) * 3];
const RED: Rgb = [200, 30, 40];
const WHITE: Rgb = [230, 230, 225];
const REF: Rgb = [20, 20, 20];

describe('teams', () => {
  it('recognises turf', () => {
    expect(isGrass([60, 130, 60])).toBe(true);
    expect(isGrass(RED)).toBe(false);
    expect(isGrass(WHITE)).toBe(false);
  });

  it('averages only the non-grass pixels of a crop', () => {
    const px = new Uint8ClampedArray([60, 130, 60, 255, 200, 30, 40, 255, 200, 30, 40, 255, 60, 130, 60, 255]);
    expect(jerseyColour(px)).toEqual([200, 30, 40]);
  });

  it('k-means separates two kits', () => {
    const pts = Array.from({ length: 40 }, (_, i) => jitter(i % 2 ? RED : WHITE, i));
    const c = kmeans(pts, 2).sort((a, b) => a[1] - b[1]);
    expect(c[0][0]).toBeGreaterThan(190);
    expect(c[1][1]).toBeGreaterThan(220);
  });

  it('labels kits consistently and flags the referee as an outlier', () => {
    const m = new TeamModel(60);
    for (let i = 0; i < 60; i++) m.add(jitter(i % 2 ? RED : WHITE, i));
    expect(m.ready).toBe(true);
    const red = m.classify(RED);
    const white = m.classify(WHITE);
    expect(new Set([red, white])).toEqual(new Set([0, 1]));
    expect(m.classify(jitter(RED, 3))).toBe(red);
    expect(m.classify(REF)).toBe(2);
    expect(m.classify(null)).toBe(-1);
  });

  it('learns the referee as a third cluster when they are in the samples', () => {
    const m = new TeamModel(66);
    for (let i = 0; i < 66; i++) m.add(jitter(i % 11 === 10 ? REF : i % 2 ? RED : WHITE, i));
    expect(m.officials).not.toBeNull();
    expect(m.classify(jitter(REF, 2))).toBe(2);
    expect(new Set([m.classify(RED), m.classify(WHITE)])).toEqual(new Set([0, 1]));
  });

  it('does not mistake a shading split of one kit for officials', () => {
    const m = new TeamModel(60);
    const shade = (c: Rgb, k: number): Rgb => [c[0] * k, c[1] * k, c[2] * k];
    for (let i = 0; i < 60; i++) m.add(i % 2 ? shade(RED, i % 4 === 1 ? 1 : 0.85) : WHITE);
    expect(m.officials).toBeNull();
    expect(m.classify(shade(RED, 0.85))).toBe(m.classify(RED));
  });
});
