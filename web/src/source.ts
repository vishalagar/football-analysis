/** Loads footage into the shared <video> element from a file or a shared tab. */

export type SourceKind = 'file' | 'tab';

export function loadFile(video: HTMLVideoElement, file: File): void {
  stopStream(video);
  if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.load();
}

/** Phones and tablets have no getDisplayMedia, so tab sharing is desktop only. */
export const canShareTab = () => !!navigator.mediaDevices?.getDisplayMedia;

/** Chrome's CaptureController; not in every DOM typing yet. */
type FocusController = { setFocusBehavior?(b: 'focus-capturing-application' | 'no-focus-change'): void };

export async function loadTab(video: HTMLVideoElement): Promise<void> {
  if (!canShareTab()) throw new Error('unsupported');
  const Ctl = (window as unknown as { CaptureController?: new () => FocusController }).CaptureController;
  const controller = Ctl ? new Ctl() : undefined;
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: 30 },
    audio: false,
    controller,
    // Sharing this tab would analyse our own overlay.
    selfBrowserSurface: 'exclude',
  } as DisplayMediaStreamOptions);
  // Chrome jumps to the shared tab by default, hiding the analysis. Stay here;
  // the shared tab keeps playing in the background.
  try {
    controller?.setFocusBehavior?.('focus-capturing-application');
  } catch {
    // Too late or not a tab: the user switches back by hand.
  }
  stopStream(video);
  video.removeAttribute('src');
  video.srcObject = stream;
  video.muted = true;
  await video.play();
}

export function stopStream(video: HTMLVideoElement): void {
  const s = video.srcObject as MediaStream | null;
  s?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

/**
 * Copy-protected players (Hotstar, Netflix, Prime…) hand screen capture a
 * black frame. Mean brightness is sampled; two seconds of near-black while
 * playing is reported so the user is told why nothing is detected.
 */
export class DrmWatch {
  private darkSince: number | null = null;
  private c = document.createElement('canvas');
  private ctx = this.c.getContext('2d', { willReadFrequently: true })!;
  flagged = false;

  constructor() {
    this.c.width = 32;
    this.c.height = 18;
  }

  check(video: HTMLVideoElement, now: number): boolean {
    if (this.flagged || video.paused || video.readyState < 2) return this.flagged;
    this.ctx.drawImage(video, 0, 0, 32, 18);
    const px = this.ctx.getImageData(0, 0, 32, 18).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
    const mean = sum / (px.length / 4) / 3;
    if (mean > 6) this.darkSince = null;
    else if (this.darkSince === null) this.darkSince = now;
    else if (now - this.darkSince > 2000) this.flagged = true;
    return this.flagged;
  }

  reset(): void {
    this.darkSince = null;
    this.flagged = false;
  }
}

/**
 * Broadcast cuts break every track at once. A cut shows up as a large jump
 * in a coarse brightness histogram between consecutive analysed frames.
 */
export class CutDetector {
  private prev: Float32Array | null = null;
  private c = document.createElement('canvas');
  private ctx = this.c.getContext('2d', { willReadFrequently: true })!;

  constructor(private readonly threshold = 0.45) {
    this.c.width = 64;
    this.c.height = 36;
  }

  isCut(src: CanvasImageSource): boolean {
    this.ctx.drawImage(src, 0, 0, 64, 36);
    const hist = histogram(this.ctx.getImageData(0, 0, 64, 36).data);
    const cut = this.prev !== null && histDistance(this.prev, hist) > this.threshold;
    this.prev = hist;
    return cut;
  }

  reset(): void {
    this.prev = null;
  }
}

/** 16-bin brightness histogram, normalised to sum to 1. */
export function histogram(px: Uint8ClampedArray): Float32Array {
  const h = new Float32Array(16);
  const n = px.length / 4;
  for (let i = 0; i < px.length; i += 4) {
    const y = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
    h[Math.min(15, y >> 4)] += 1 / n;
  }
  return h;
}

/** Half the L1 distance: 0 for identical histograms, 1 for disjoint ones. */
export function histDistance(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
  return d / 2;
}
