// Host-side inference loop, dependency-free (no ORT import here).
// Ported from ming-tts-mnn 0.1.3 harness.py: ar_generate + sample_latent +
// stop head + decode host layers + trim. Graph calls are injected, so this
// module is unit-testable with stub graphs and swappable for real ORT
// sessions in sessions.ts / ming-tts.ts without changing the math.
//
// RNG NOTE: the reference uses numpy's PCG64 default_rng with a shared
// stream (prompt-eps first, then per-step flow noise, in order). This JS
// RNG (mulberry32 + Box-Muller) is a DIFFERENT stream: same-seed JS runs
// will NOT be bit-identical to numpy runs, and speaker/quality comparisons
// must therefore use distribution metrics (log-mel L1, CAM++ cosine), not
// sample equality. Do not claim numpy bit-parity for this RNG.

import { causalMask, qwenRopeTables, swayTimesteps, xtRopeTables, linearUpsample, specToWav, softplus, softmaxRow } from './mathops.js';

export interface Dims {
  hidden: number;
  headDim: number;
  ropeTheta: number;
  lmLayers: number;
  kvHeads: number;
  lmWindow: number;
  latent: number;
  patchSizeDit: number;
  historyPatchSize: number;
  hop: number;
  patch: number;
  nFft: number;
  sampleRate: number;
  encWindow: number;
  decWindow: number;
}

export const DEFAULT_DIMS_0_5B: Dims = {
  hidden: 896, headDim: 64, ropeTheta: 1_000_000, lmLayers: 24, kvHeads: 2,
  lmWindow: 32768, latent: 64, patchSizeDit: 1, historyPatchSize: 1,
  hop: 882, patch: 4, nFft: 3528, sampleRate: 44100, encWindow: 64, decWindow: 64,
};

export interface HostWeights {
  /** Token embedding table: (vocab, hidden). Only rows used by the prompt are read. */
  embed: number[][];
  spkW: number[][]; // (hidden, 192)
  spkB: number[]; // (hidden,)
  stopW: number[][]; // (2, hidden)
  stopB: number[]; // (2,)
  fc1DecW: number[][]; // (decHidden, latent)
  fc1DecB: number[]; // (decHidden,)
  istftWindow: number[]; // (nFft,) hann
  xtInvFreq: number[]; // DiT RoPE inv freq (headDim/2)
  xtAggInvFreq: number[]; // Aggregator RoPE inv freq
  clsEmbed: number[]; // (encHidden,)
}

/** mulberry32 — small, seedable, deterministic. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Gaussian sampler (Box-Muller, with spare) over a uniform source. */
export function gaussianSampler(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = uniform();
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

export function matVec(w: number[][], x: number[]): number[] {
  // w: (out, in), x: (in,) -> (out,)
  return w.map((row) => {
    let s = 0;
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    return s;
  });
}

export function addBias(v: number[], b: number[]): number[] {
  return v.map((x, i) => x + b[i]);
}

export function spkProj(weights: HostWeights, spkEmb: number[]): number[] {
  // (1,192) @ spkW.T + spkB, with spkW stored (hidden,192).
  const out: number[] = [];
  for (let h = 0; h < weights.spkW.length; h++) {
    let s = weights.spkB[h];
    for (let i = 0; i < 192; i++) s += weights.spkW[h][i] * spkEmb[i];
    out.push(s);
  }
  return out;
}

/** Stop probability = softmax(stop_head(z))[1]. */
export function stopProb(weights: HostWeights, z: number[]): number {
  const logits = addBias(matVec(weights.stopW, z), weights.stopB);
  return softmaxRow(logits)[1];
}

export interface DitFn {
  (args: {
    x: number[][]; // (2, patch*latent) flattened per batch row — see note
    t: number[];
    cond: number[][];
    history: number[][];
    cos: number[][];
    sin: number[][];
  }): number[][];
}

// Latents are handled as flat rows of length latent (patch_size_dit = 1),
// so a "patch" is one row. DitFn therefore takes rows, not 3-D tensors;
// sessions.ts is responsible for reshaping rows to ORT tensors.

/**
 * One CFM sample: Euler integration of the flow with CFG.
 * Mirrors harness.sample_latent / upstream CFM.sample exactly:
 * 10 steps, EPSS + sway timesteps, velocity = pred + (pred - null) * cfg,
 * SDE term sigma * sqrt(temperature) * sqrt(|dt|) * sdeNoise[i] (vanishes
 * at temperature 0). ditFn is called once per step with the CFG-doubled
 * batch (noise row duplicated, cond = [z, zeros], history duplicated).
 */
export function sampleLatent(opts: {
  z: number[];
  history: number[];
  noise: number[];
  sdeNoise: number[][];
  cfg: number;
  sigma: number;
  temperature: number;
  ditFn: (x: number[][], t: number[], c: number[][], h: number[][]) => number[][];
}): number[] {
  const { z, history, cfg, sigma, temperature, ditFn } = opts;
  const steps = opts.sdeNoise.length;
  const t = swayTimesteps(steps);
  let y = [...opts.noise];
  const scale = sigma * Math.sqrt(temperature);
  for (let i = 0; i < steps; i++) {
    const dt = t[i + 1] - t[i];
    const x = [y, [...y]];
    const c = [z, new Array(z.length).fill(0)];
    const h = [history, [...history]];
    const out = ditFn(x, [t[i], t[i]], c, h);
    const pred = out[0];
    const nul = out[1];
    const vel = pred.map((p, j) => p + (p - nul[j]) * cfg);
    y = y.map((v, j) => v + vel[j] * dt);
    if (scale !== 0) {
      const sde = opts.sdeNoise[i];
      y = y.map((v, j) => v + scale * Math.sqrt(Math.abs(dt)) * sde[j]);
    }
  }
  return y;
}

/** Trailing-silence trim, per fragment (0.1 s frames, 1e-3, 0.3 s tail). */
export function trimTrailingSilence(
  speech: Float32Array,
  sampleRate: number,
  threshold = 1e-3,
  tailSeconds = 0.3,
): Float32Array {
  if (speech.length === 0) return speech;
  const frame = Math.floor(sampleRate * 0.1);
  const tail = Math.floor(sampleRate * tailSeconds);
  if (frame <= 0 || speech.length < frame) return speech.slice(0, tail);
  const numFrames = Math.floor((speech.length - frame) / frame) + 1;
  const truncated = speech.slice(0, numFrames * frame);
  const scores: number[] = [];
  for (let f = 0; f < numFrames; f++) {
    let s = 0;
    for (let i = 0; i < frame; i++) s += Math.abs(truncated[f * frame + i]);
    scores.push(s / frame);
  }
  let index = scores.length - 1;
  while (index >= 0 && scores[index] <= threshold) index--;
  if (index < 0) return truncated.slice(0, tail);
  return truncated.slice(0, index * frame + frame + tail);
}

/**
 * Vocoder host decode: latents (T, latent) -> fc1 -> linear upsample x patch
 * -> vocoder graph (injected) -> ISTFT -> waveform @ sampleRate.
 * vocFn takes the upsampled rows (T*patch, decHidden) and returns the raw
 * spec head output as (specDim, T*patch) rows.
 */
export function decodeLatents(opts: {
  latents: number[][];
  weights: HostWeights;
  dims: Dims;
  vocFn: (up: number[][], cos: number[][], sin: number[][], mask: number[][]) => number[][];
}): Float32Array {
  const { latents, weights, dims } = opts;
  if (latents.length === 0) return new Float32Array(0);
  const projected = latents.map((row) => addBias(matVec(weights.fc1DecW, row), weights.fc1DecB));
  // linearUpsample works on (C, T); transpose, upsample, transpose back.
  const channels = projected[0].length;
  const asChannels: number[][] = [];
  for (let c = 0; c < channels; c++) asChannels.push(projected.map((row) => row[c]));
  const upChannels = linearUpsample(asChannels, dims.patch);
  const up: number[][] = upChannels[0].map((_, t) => upChannels.map((ch) => ch[t]));
  const n = up.length;
  const positions = Array.from({ length: n }, (_, i) => i);
  const { cos, sin } = qwenRopeTables(positions, dims.headDim, dims.ropeTheta);
  const mask = causalMask(n, n, 0, dims.decWindow);
  const spec = opts.vocFn(up, cos, sin, mask);
  return specToWav(spec, dims.nFft, dims.hop, weights.istftWindow);
}

export interface LmFn {
  (args: {
    embeds: number[][];
    mask: number[][];
    cos: number[][];
    sin: number[][];
    pastK: number[][][][] | null;
    pastV: number[][][][] | null;
  }): { hidden: number[][]; pastK: number[][][][]; pastV: number[][][][] };
}

export interface AggregatorFn {
  (latent: number[], cos: number[][], sin: number[][]): number[];
}

/**
 * The autoregressive loop (harness.ar_generate), with injected graphs.
 * Returns latents (one row per step) + diagnostics. The stop-step latent IS
 * included, and stopping requires step > 3, exactly as the reference.
 */
export function arGenerate(opts: {
  embeds: number[][];
  promptLatent: number[][] | null;
  weights: HostWeights;
  dims: Dims;
  lmFn: LmFn;
  ditFn: (x: number[][], t: number[], c: number[][], h: number[][]) => number[][];
  aggregatorFn: AggregatorFn;
  cfg?: number;
  sigma?: number;
  temperature?: number;
  maxDecodeSteps?: number;
  gaussian: () => number;
}): { latents: number[][]; steps: number; stopStep: number | null; stopped: boolean } {
  const { dims, weights } = opts;
  const cfg = opts.cfg ?? 2.0;
  const sigma = opts.sigma ?? 0.25;
  const temperature = opts.temperature ?? 0.0;
  const maxDecodeSteps = opts.maxDecodeSteps ?? 1000;
  const n = opts.embeds.length;
  const MAX_CACHE_LEN = 512;
  const budget = Math.floor((MAX_CACHE_LEN - n) / dims.patchSizeDit);
  const steps = Math.max(1, Math.min(maxDecodeSteps, budget));
  const positions = Array.from({ length: n }, (_, i) => i);
  const { cos, sin } = qwenRopeTables(positions, dims.headDim, dims.ropeTheta);
  const mask = causalMask(n, n, 0, dims.lmWindow);
  let { hidden, pastK, pastV } = opts.lmFn({ embeds: opts.embeds, mask, cos, sin, pastK: null, pastV: null });
  let history = new Array(dims.latent).fill(0);
  if (opts.promptLatent !== null) {
    const pl = opts.promptLatent;
    const start = dims.historyPatchSize - pl.length;
    if (start < 0) {
      history = [...pl[pl.length - dims.historyPatchSize]];
    } else {
      const tail = pl[pl.length - 1];
      history = [...tail];
    }
    void start;
  }
  const ditPositions = Array.from({ length: 1 + dims.historyPatchSize + dims.patchSizeDit }, (_, i) => i);
  const ditRope = xtRopeTables(weights.xtInvFreq, ditPositions);
  void ditRope;
  const aggPositions = Array.from({ length: 1 + dims.patchSizeDit }, (_, i) => i);
  const aggRope = xtRopeTables(weights.xtAggInvFreq, aggPositions);
  const out: number[][] = [];
  let stopStep: number | null = null;
  for (let step = 0; step < steps; step++) {
    const z = hidden[hidden.length - 1];
    const noise = Array.from({ length: dims.latent }, () => opts.gaussian());
    const sde = Array.from({ length: 10 }, () =>
      Array.from({ length: dims.latent }, () => opts.gaussian()),
    );
    const latent = sampleLatent({
      z, history, noise, sdeNoise: sde, cfg, sigma, temperature,
      ditFn: opts.ditFn,
    });
    out.push(latent);
    if (stopProb(weights, z) > 0.5 && step > 3) {
      stopStep = step;
      break;
    }
    const nxt = opts.aggregatorFn(latent, aggRope.cos, aggRope.sin);
    const pos = n + step;
    const stepMask = causalMask(1, pos + 1, pos, dims.lmWindow);
    const stepRope = qwenRopeTables([pos], dims.headDim, dims.ropeTheta);
    const res = opts.lmFn({
      embeds: [nxt], mask: stepMask, cos: stepRope.cos, sin: stepRope.sin, pastK, pastV,
    });
    hidden = res.hidden;
    pastK = res.pastK;
    pastV = res.pastV;
    history = [...latent];
  }
  return { latents: out, steps: out.length, stopStep, stopped: stopStep !== null };
}
