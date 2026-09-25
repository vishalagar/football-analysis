import { describe, expect, it } from 'vitest';
import { retry } from '../src/retry';

describe('retry', () => {
  it('returns the first success', async () => {
    let calls = 0;
    const v = await retry(async () => { calls++; if (calls < 3) throw new Error('flaky'); return 'ok'; }, 3, 0);
    expect(v).toBe('ok');
    expect(calls).toBe(3);
  });

  it('rethrows the last error once attempts run out', async () => {
    let calls = 0;
    await expect(retry(async () => { calls++; throw new Error(`fail ${calls}`); }, 2, 0)).rejects.toThrow('fail 2');
    expect(calls).toBe(2);
  });

  it('reports each failed attempt', async () => {
    const seen: number[] = [];
    await retry(async () => { if (seen.length < 1) throw new Error('x'); return 1; }, 3, 0, (n) => seen.push(n));
    expect(seen).toEqual([1]);
  });
});
