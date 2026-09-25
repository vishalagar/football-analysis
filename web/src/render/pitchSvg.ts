import { PITCH } from '../pitch';

const L = PITCH.length;
const W = PITCH.width;

/** Chalk markings of a 105 × 68 pitch in metres, drawn to IFAB dimensions. */
export function pitchMarkings(): string {
  const box = (x: number, dir: 1 | -1) => {
    const bx = dir === 1 ? x : x - 16.5;
    const gx = dir === 1 ? x : x - 5.5;
    const spot = x + dir * 11;
    // The "D": the part of a 9.15 m circle round the spot outside the box.
    const dx = 16.5 - 11;
    const dy = Math.sqrt(9.15 ** 2 - dx ** 2);
    const edge = x + dir * 16.5;
    return `
      <rect x="${bx}" y="${W / 2 - 20.16}" width="16.5" height="40.32"/>
      <rect x="${gx}" y="${W / 2 - 9.16}" width="5.5" height="18.32"/>
      <circle cx="${spot}" cy="${W / 2}" r="0.25" class="dot"/>
      <path d="M ${edge} ${W / 2 - dy} A 9.15 9.15 0 0 ${dir === 1 ? 1 : 0} ${edge} ${W / 2 + dy}"/>
      <rect x="${dir === 1 ? x - 1.6 : x}" y="${W / 2 - 3.66}" width="1.6" height="7.32" class="goal"/>`;
  };
  return `
    <rect x="0" y="0" width="${L}" height="${W}"/>
    <line x1="${L / 2}" y1="0" x2="${L / 2}" y2="${W}"/>
    <circle cx="${L / 2}" cy="${W / 2}" r="9.15"/>
    <circle cx="${L / 2}" cy="${W / 2}" r="0.3" class="dot"/>
    ${box(0, 1)}${box(L, -1)}
    <path d="M 1 0 A 1 1 0 0 1 0 1 M ${L - 1} 0 A 1 1 0 0 0 ${L} 1
             M ${L - 1} ${W} A 1 1 0 0 1 ${L} ${W - 1} M 1 ${W} A 1 1 0 0 0 0 ${W - 1}"/>`;
}

export function pitchSvg(extra = ''): string {
  return `<svg viewBox="-2.5 -2.5 ${L + 5} ${W + 5}" class="pitch" role="img" aria-label="Pitch diagram">
    <g class="lines">${pitchMarkings()}</g>${extra}</svg>`;
}
