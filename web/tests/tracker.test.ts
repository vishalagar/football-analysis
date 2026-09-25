import { describe, expect, it } from 'vitest';
import { Tracker } from '../src/tracker';
import type { Detection } from '../src/types';

const p = (x: number, y: number): Detection => ({ kind: 'player', score: 0.9, x, y, w: 20, h: 50 });

describe('Tracker', () => {
  it('keeps the same id for a player moving steadily', () => {
    const tr = new Tracker({ minHits: 1 });
    const ids = new Set<number>();
    for (let i = 0; i < 20; i++) {
      const [a] = tr.update([p(100 + i * 6, 200)], i * 0.1);
      ids.add(a.id);
    }
    expect(ids.size).toBe(1);
  });

  it('keeps two crossing players apart by motion', () => {
    const tr = new Tracker({ minHits: 1 });
    let first: number[] = [];
    for (let i = 0; i < 10; i++) {
      const out = tr.update([p(100 + i * 8, 100), p(100 + i * 8, 300)], i * 0.1);
      const top = out.find((t) => t.y < 200)!.id;
      const bottom = out.find((t) => t.y >= 200)!.id;
      if (i === 0) first = [top, bottom];
      expect([top, bottom]).toEqual(first);
    }
  });

  it('re-finds a fast mover through the centre-distance pass', () => {
    const tr = new Tracker({ minHits: 1 });
    const [a] = tr.update([p(100, 100)], 0);
    const [b] = tr.update([p(122, 100)], 0.1); // no box overlap, but close
    expect(b.id).toBe(a.id);
  });

  it('only confirms a track after minHits and drops it after maxMissed', () => {
    const tr = new Tracker({ minHits: 3, maxMissed: 2 });
    expect(tr.update([p(0, 0)], 0)).toHaveLength(0);
    tr.update([p(0, 0)], 0.1);
    expect(tr.update([p(0, 0)], 0.2)).toHaveLength(1);
    for (let i = 0; i < 3; i++) tr.update([], 0.3 + i * 0.1);
    expect(tr.tracks).toHaveLength(0);
  });

  it('ignores balls', () => {
    const tr = new Tracker({ minHits: 1 });
    expect(tr.update([{ ...p(0, 0), kind: 'ball' }], 0)).toHaveLength(0);
  });
});
