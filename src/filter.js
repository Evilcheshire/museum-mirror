// One Euro Filter — adaptive low-pass filter for landmark smoothing.
// Preserves fast intentional movements while removing high-frequency jitter.
// Reference: https://gery.casiez.net/1euro/

class LowPassFilter {
  constructor(alpha) {
    this.alpha = alpha;
    this.y = null;
    this.s = null;
  }

  filter(value, alpha) {
    if (alpha !== undefined) this.alpha = alpha;
    const result = this.y === null
      ? value
      : this.alpha * value + (1 - this.alpha) * this.s;
    this.y = value;
    this.s = result;
    return result;
  }

  last() {
    return this.s;
  }
}

export class OneEuroFilter {
  // freq:      sampling rate (Hz) — recalculated on the fly from timestamps
  // minCutoff: smaller → more smoothing of slow motion
  // beta:      larger  → less lag on fast motion (tradeoff: less smooth)
  // dCutoff:   cutoff for derivative estimate
  constructor(freq = 30, minCutoff = 1.2, beta = 0.05, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter(this._alpha(minCutoff));
    this.dxFilter = new LowPassFilter(this._alpha(dCutoff));
    this.lastTime = null;
  }

  _alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  filter(value, timestamp) {
    if (timestamp !== undefined && this.lastTime !== null) {
      const dt = (timestamp - this.lastTime) / 1000;
      if (dt > 0) this.freq = 1.0 / dt;
    }
    this.lastTime = timestamp;

    const prev = this.xFilter.last();
    const dx = prev === null ? 0 : (value - prev) * this.freq;
    const edx = this.dxFilter.filter(dx, this._alpha(this.dCutoff));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this._alpha(cutoff));
  }
}
