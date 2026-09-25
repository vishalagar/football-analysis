import { describe, expect, it } from 'vitest';
import { MatchStats } from '../src/stats';

describe('MatchStats', () => {
  it('splits possession by time on the ball', () => {
    const s = new MatchStats();
    for (let i = 0; i < 30; i++) s.step(i * 0.1, 0.1, [], null, 0, true);
    for (let i = 0; i < 10; i++) s.step(3 + i * 0.1, 0.1, [], null, 1, true);
    expect(s.possessionShare()).toEqual([75, 25]);
  });

  it('adds distance in metres and rejects impossible jumps', () => {
    const s = new MatchStats();
    const at = (x: number) => [{ id: 1, team: 0 as const, pos: { x, y: 10 }, px: { x: 0, y: 0 }, mpp: 1 }];
    s.step(0, 0, at(10), null, null, true);
    s.step(1, 1, at(15), null, null, true); // 5 m in 1 s
    s.step(1.5, 0.5, at(40), null, null, true); // 50 m/s: glitch
    expect(s.players.get(1)!.metres).toBeCloseTo(5);
  });

  it('uses player height for scale before calibration', () => {
    const s = new MatchStats();
    const at = (px: number) => [{ id: 1, team: 1 as const, pos: { x: 0, y: 0 }, px: { x: px, y: 0 }, mpp: 1.8 / 90 }];
    s.step(0, 0, at(0), null, null, false);
    s.step(1, 1, at(250), null, null, false);
    expect(s.teamMetres(1)).toBeCloseTo(5);
  });

  it('records passes as network edges and turnovers per side', () => {
    const s = new MatchStats();
    s.event({ t: 1, type: 'pass', team: 0, from: 1, to: 2 });
    s.event({ t: 2, type: 'pass', team: 0, from: 1, to: 2 });
    s.event({ t: 3, type: 'turnover', team: 1, from: 2, to: 5 });
    expect(s.passes).toEqual([2, 0]);
    expect(s.turnoversWon).toEqual([0, 1]);
    expect(s.edges.get('1>2')).toBe(2);
  });

  it('counts players on screen now, not every id ever seen', () => {
    const s = new MatchStats();
    const smp = (id: number, team: 0 | 1) => ({ id, team, pos: { x: 0, y: 0 }, px: { x: 0, y: 0 }, mpp: 1 });
    s.step(0, 0, [smp(1, 0), smp(2, 0), smp(3, 1)], null, null, true);
    s.step(0.1, 0.1, [smp(4, 0), smp(5, 1)], null, null, true);
    expect(s.players.size).toBe(5);
    expect(s.onScreen).toEqual([1, 1]);
  });
});
