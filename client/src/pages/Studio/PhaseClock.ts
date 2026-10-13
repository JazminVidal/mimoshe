/** One-step-delayed display clock; never extrapolates beyond received steps. */
export class PhaseClock {
  private start = 0;
  private target = 0;
  private lastPhase = 0;
  private startedAt = 0;
  private duration = 80;
  private frameDuration = 80;
  initialized = false;

  setFrameRate(frameRate: number) {
    if (Number.isFinite(frameRate) && frameRate > 0) {
      this.frameDuration = 1000 / frameRate;
    }
  }

  reset(phase: number, now: number) {
    this.start = phase;
    this.target = phase;
    this.lastPhase = phase;
    this.startedAt = now;
    this.initialized = true;
  }

  observe(phase: number, now: number) {
    if (!Number.isFinite(phase) || phase < 0 || phase >= 1) return;
    if (!this.initialized) {
      this.reset(phase, now);
      return;
    }
    // Duplicate acknowledgements must not keep restarting the animation.
    if (Math.abs(phase - this.lastPhase) < 1e-9) return;
    const delta = (phase - this.lastPhase + 1) % 1;
    this.start = this.readUnwrapped(now);
    this.target += delta;
    this.lastPhase = phase;
    this.startedAt = now;
    this.duration = this.frameDuration;
  }

  private readUnwrapped(now: number) {
    const fraction = Math.min(1, Math.max(0, (now - this.startedAt) / this.duration));
    return this.start + (this.target - this.start) * fraction;
  }

  sample(now: number) {
    const unwrapped = this.readUnwrapped(now);
    return { phase: unwrapped % 1, unwrapped };
  }
}
