// Pure host-side math ported from ming-tts-mnn 0.1.3 mathops.py.
// Shapes: RoPE tables are returned WITHOUT the batch dim the Python code
// carries, i.e. (S, headDim) instead of (1, S, headDim). Callers add the
// batch dim when building ORT tensors. All tables are fp32 (Float32 math
// via Math.f32round where it matters); the Python reference notes fp32
// tables reproduce torch to ~2e-6, while fp64 differs by ~5e-6 and compounds
// across 24 layers — do NOT "improve" this to fp64.

export const NEG = -1e4; // additive-mask "blocked" value

/** HF Qwen2 RoPE: half-split rotation. Returns (S, headDim) cos/sin. */
export function qwenRopeTables(
  positions: number[],
  headDim: number,
  theta: number,
): { cos: number[][]; sin: number[][] } {
  const half = headDim / 2;
  const invFreq: number[] = [];
  for (let i = 0; i < headDim; i += 2) {
    invFreq.push(Math.pow(theta, -i / headDim));
  }
  const cos: number[][] = [];
  const sin: number[][] = [];
  for (const pos of positions) {
    const freqs = invFreq.map((f) => pos * f);
    const emb = [...freqs, ...freqs];
    cos.push(emb.map((v) => Math.cos(v)));
    sin.push(emb.map((v) => Math.sin(v)));
  }
  return { cos, sin };
}

/**
 * x_transformers RoPE (DiT / Aggregator): interleaved pairs.
 * invFreq has length headDim/2; angles repeat each freq twice.
 */
export function xtRopeTables(
  invFreq: number[],
  positions: number[],
): { cos: number[][]; sin: number[][] } {
  const cos: number[][] = [];
  const sin: number[][] = [];
  for (const pos of positions) {
    const freqs = invFreq.map((f) => pos * f);
    const angles: number[] = [];
    for (const f of freqs) angles.push(f, f);
    cos.push(angles.map((v) => Math.cos(v)));
    sin.push(angles.map((v) => Math.sin(v)));
  }
  return { cos, sin };
}

/**
 * Additive causal mask (S, KV), matching transformers 4.46 Qwen2 eager mask.
 * A row may attend to kvIdx <= pos, and (when window != null) kvIdx > pos - window.
 */
export function causalMask(
  seqLen: number,
  kvLen: number,
  pastLen: number,
  window: number | null = null,
): number[][] {
  const out: number[][] = [];
  for (let s = 0; s < seqLen; s++) {
    const pos = pastLen + s;
    const row: number[] = [];
    for (let kv = 0; kv < kvLen; kv++) {
      let allowed = kv <= pos;
      if (window !== null) allowed = allowed && kv > pos - window;
      row.push(allowed ? 0 : NEG);
    }
    out.push(row);
  }
  return out;
}

export function fullOnesMask(seqLen: number, kvLen: number): number[][] {
  return Array.from({ length: seqLen }, () => new Array(kvLen).fill(0));
}

/** EPSS timesteps for n steps (CFM.py get_epss_timesteps). */
export function epssTimesteps(n: number): number[] {
  const dt = 1 / 32;
  const predefined: Record<number, number[]> = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
  };
  const t = predefined[n];
  if (!t) return Array.from({ length: n + 1 }, (_, i) => i / n);
  return t.map((v) => dt * v);
}

/** CFM.sample's timestep vector after the sway-resampling line (coef -1). */
export function swayTimesteps(steps: number, coef = -1.0): number[] {
  return epssTimesteps(steps).map(
    (t) => t + coef * (Math.cos((Math.PI / 2) * t) - 1 + t),
  );
}

/**
 * nn.Upsample(mode='linear', align_corners=False) on (C, T) -> (C, T*scale).
 * Ported from mathops.linear_upsample; index math is fp64 in Python, kept
 * in JS doubles here (the interpolation weights, not the audio, are doubles).
 */
export function linearUpsample(x: number[][], scale: number): number[][] {
  const c = x.length;
  if (c === 0) return [];
  const t = x[0].length;
  const outT = t * scale;
  const out: number[][] = [];
  for (let ch = 0; ch < c; ch++) {
    const row: number[] = [];
    for (let i = 0; i < outT; i++) {
      let src = (i + 0.5) * (t / outT) - 0.5;
      src = Math.min(Math.max(src, 0), t - 1);
      const lo = Math.floor(src);
      const hi = Math.min(lo + 1, t - 1);
      const w = src - lo;
      row.push(x[ch][lo] * (1 - w) + x[ch][hi] * w);
    }
    out.push(row);
  }
  return out;
}

/**
 * Non-streaming ISTFT with padding='same'.
 * specReal/specImag: (nFreq, T) where nFreq = nFft/2 + 1.
 * Returns (T*hop) samples. Window envelope division included; throws if the
 * envelope collapses (mirrors the Python assert).
 */
export function istftSame(
  specReal: number[][],
  specImag: number[][],
  nFft: number,
  hop: number,
  window: number[],
): Float32Array {
  const nFreq = specReal.length;
  const t = specReal[0].length;
  // irfft per frame (naive DFT — correct, not fast; the vocoder runs once
  // per fragment, and a real FFT can replace this without changing the API).
  const outLen = (t - 1) * hop + nFft;
  const y = new Float64Array(outLen);
  const env = new Float64Array(outLen);
  const winSq = window.map((v) => v * v);
  for (let frame = 0; frame < t; frame++) {
    // Real IFFT of the Hermitian spectrum: x[n] = (1/N) * sum_k Re/Im terms.
    for (let n = 0; n < nFft; n++) {
      let re = specReal[0][frame]; // k = 0
      let im = 0;
      for (let k = 1; k < nFreq - 1; k++) {
        const ang = (2 * Math.PI * k * n) / nFft;
        re += 2 * (specReal[k][frame] * Math.cos(ang) - specImag[k][frame] * Math.sin(ang));
        void im;
      }
      // k = Nyquist (nFft even): contributes once, real only.
      const nyq = nFreq - 1;
      re += specReal[nyq][frame] * Math.cos(Math.PI * n);
      const val = (re / nFft) * window[n];
      y[frame * hop + n] += val;
      env[frame * hop + n] += winSq[n];
    }
  }
  const pad = Math.floor((nFft - hop) / 2);
  const trimmed = outLen - 2 * pad;
  const out = new Float32Array(trimmed);
  for (let i = 0; i < trimmed; i++) {
    const e = env[pad + i];
    if (!(e > 1e-11)) throw new Error('istftSame: window envelope collapsed');
    out[i] = y[pad + i] / e;
  }
  return out;
}

/**
 * ISTFTHead: (specDim, T) raw head output -> waveform.
 * specDim = 2 * (nFft/2 + 1); first half = magnitude logits, second = phase.
 * mag = clip(exp(mag), max 1e2); spec = mag * (cos phase + i sin phase).
 */
export function specToWav(
  xPred: number[][],
  nFft: number,
  hop: number,
  window: number[],
): Float32Array {
  const half = xPred.length / 2;
  const mag = xPred.slice(0, half).map((row) =>
    row.map((v) => Math.min(Math.exp(v), 1e2)),
  );
  const phase = xPred.slice(half);
  const specReal = mag.map((row, k) =>
    row.map((m, t) => m * Math.cos(phase[k][t])),
  );
  const specImag = mag.map((row, k) =>
    row.map((m, t) => m * Math.sin(phase[k][t])),
  );
  return istftSame(specReal, specImag, nFft, hop, window);
}

export function softplus(x: number): number {
  // torch softplus, threshold 20, stable form.
  return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
}

export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}
