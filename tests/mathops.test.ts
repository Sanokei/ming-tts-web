import { describe, expect, test } from 'bun:test';
import { qwenRopeTables, xtRopeTables, causalMask, swayTimesteps, epssTimesteps, linearUpsample, specToWav, softplus } from '../src/mathops.ts';
import { trimTrailingSilence, sampleLatent, stopProb } from '../src/host-loop.ts';

describe('mathops (ported from MNN mathops.py)', () => {
  test('qwen rope at position 0 is cos=1, sin=0', () => {
    const { cos, sin } = qwenRopeTables([0], 64, 1_000_000);
    expect(cos[0].every((v) => Math.abs(v - 1) < 1e-12)).toBe(true);
    expect(sin[0].every((v) => Math.abs(v) < 1e-12)).toBe(true);
  });
  test('qwen rope half-split: first half freqs repeat in second half', () => {
    const { cos } = qwenRopeTables([3], 8, 10000);
    // emb = [f0,f1,f2,f3, f0,f1,f2,f3] so cos halves are identical
    expect(cos[0].slice(0, 4)).toEqual(cos[0].slice(4, 8));
  });
  test('xt rope interleaves pairs', () => {
    const { cos } = xtRopeTables([0.5, 0.25], [2]);
    // angles = [1,1,0.5,0.5]
    expect(Math.abs(cos[0][0] - Math.cos(1)) < 1e-12).toBe(true);
    expect(Math.abs(cos[0][1] - Math.cos(1)) < 1e-12).toBe(true);
    expect(Math.abs(cos[0][2] - Math.cos(0.5)) < 1e-12).toBe(true);
  });
  test('causal mask blocks future and respects sliding window', () => {
    const m = causalMask(3, 3, 0, null);
    expect(m[0]).toEqual([0, -1e4, -1e4]);
    expect(m[2]).toEqual([0, 0, 0]);
    const w = causalMask(3, 3, 0, 2);
    // row pos=2, window 2: kv > 0 allowed -> [blocked, 0, 0]
    expect(w[2]).toEqual([-1e4, 0, 0]);
    const past = causalMask(1, 4, 3, null);
    expect(past[0]).toEqual([0, 0, 0, 0]);
  });
  test('epss timesteps for 10 steps match the documented table', () => {
    expect(epssTimesteps(10)).toEqual(
      [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32].map((v) => v / 32),
    );
  });
  test('sway timesteps start at 0 and end at 1', () => {
    const t = swayTimesteps(10);
    expect(t.length).toBe(11);
    expect(Math.abs(t[0]) < 1e-9).toBe(true);
    expect(Math.abs(t[10] - 1) < 1e-6).toBe(true);
    // sway with coef -1 pulls intermediate times down vs linear epss
    const e = epssTimesteps(10);
    expect(t[5] < e[5]).toBe(true);
  });
  test('linear upsample x2 of a ramp', () => {
    const out = linearUpsample([[0, 1, 2]], 2);
    expect(out[0].length).toBe(6);
    // align_corners=false: first output ~ first input, last ~ last input
    expect(Math.abs(out[0][0] - 0) < 1e-9).toBe(true);
    expect(Math.abs(out[0][5] - 2) < 1e-9).toBe(true);
  });
  test('softplus matches torch stable form', () => {
    expect(Math.abs(softplus(0) - Math.log(2)) < 1e-12).toBe(true);
    expect(Math.abs(softplus(20) - 20) < 1e-6).toBe(true);
  });
  test('specToWav with tiny fft round-trips a constant-ish spec without NaN', () => {
    // nFft=8, hop=2, specDim=10, T=4; hann window length 8.
    const win = Array.from({ length: 8 }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 7));
    const mag = Array.from({ length: 5 }, () => [0, 0, 0, 0]); // exp(0)=1
    const phase = Array.from({ length: 5 }, () => [0, 0, 0, 0]);
    const wav = specToWav([...mag, ...phase], 8, 2, win);
    expect(wav.length).toBe(4 * 2);
    expect([...wav].every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe('host-loop', () => {
  test('trimTrailingSilence keeps 0.3s tail after last voiced frame', () => {
    const sr = 44100;
    const frame = Math.floor(sr * 0.1);
    const audio = new Float32Array(frame * 5);
    audio.fill(0.5, 0, frame * 2); // 2 voiced frames, then silence
    const out = trimTrailingSilence(audio, sr);
    // last voiced frame index 1 -> 2*frame + 0.3s tail, capped by input length handling
    expect(out.length).toBeLessThanOrEqual(audio.length);
    expect(out.length).toBe(2 * frame + Math.floor(sr * 0.3));
  });
  test('trimTrailingSilence on all-silence returns the 0.3s tail', () => {
    const out = trimTrailingSilence(new Float32Array(44100), 44100);
    expect(out.length).toBe(Math.floor(44100 * 0.3));
  });
  test('stopProb is softmax of stop head, index 1', () => {
    const weights = {
      embed: [], spkW: [], spkB: [], stopW: [[0, 0], [10, 0]], stopB: [0, 0],
      fc1DecW: [], fc1DecB: [], istftWindow: [], xtInvFreq: [], xtAggInvFreq: [], clsEmbed: [],
    } as never;
    const p = stopProb(weights, [1, 0]);
    // logits [0, 10] -> softmax[1] ~ 0.99995
    expect(p).toBeGreaterThan(0.999);
  });
  test('sampleLatent with a zero-velocity dit returns the noise unchanged', () => {
    const z = [0.1, -0.2];
    const out = sampleLatent({
      z, history: [0, 0], noise: [0.5, -0.5],
      sdeNoise: Array.from({ length: 10 }, () => [0, 0]),
      cfg: 2, sigma: 0.25, temperature: 0,
      ditFn: () => [[0, 0], [0, 0]],
    });
    expect(out).toEqual([0.5, -0.5]);
  });
  test('sampleLatent CFG blends pred and null: vel = pred + (pred-null)*cfg', () => {
    const out = sampleLatent({
      z: [0], history: [0], noise: [0],
      sdeNoise: Array.from({ length: 10 }, () => [0]),
      cfg: 2, sigma: 0.25, temperature: 0,
      // pred=1, null=0 every step -> vel=3; sum of dt over sway steps = 1
      ditFn: () => [[1], [0]],
    });
    expect(Math.abs(out[0] - 3) < 1e-5).toBe(true);
  });
});
