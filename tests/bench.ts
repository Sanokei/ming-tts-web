// Benchmark script — WebGPU vs WASM, fp16 vs int8, RTF.
// This is the Step 2 gate runner (see notes/STEP2-GATE-PROTOCOL.md).
// It does NOT synthesize: it times single LM + DiT steps through an
// injected session factory, so it runs identically in a browser (ORT-web)
// and in Node (onnxruntime-node) once real sessions exist.
//
// Usage (browser, after Step 3 exports exist):
//   import { runBench } from './bench.ts'; // with a real factory
// Usage (Node, campplus smoke only — the one graph already available):
//   MING_BENCH_CAMPPLUS_ONNX=/path/to/campplus.onnx bun tests/bench.ts
//   (times campplus on CPU via onnxruntime-node; lm/dit need Step 3 exports)
//
// PASS BAR (decided before running): WebGPU median (lm decode + 1 dit step)
// at least 2x faster than WASM, both graphs, no hot-loop WASM fallback.
// Anything else = STOP, report numbers, do not build the full loop.

import { speakerFeatures } from '../src/audio.ts';

export interface BenchSession {
  run(feeds: Record<string, { data: Float32Array; dims: number[] }>): Promise<Record<string, { data: Float32Array; dims: number[] }>>;
}

export interface BenchFactory {
  createSession(path: string, providers: string[]): Promise<BenchSession>;
}

export interface BenchResult {
  label: string;
  medianMs: number;
  p90Ms: number;
  runs: number;
}

export function summarize(label: string, timesMs: number[]): BenchResult {
  const sorted = [...timesMs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const p90 = sorted[Math.floor(sorted.length * 0.9)] ?? sorted[sorted.length - 1];
  return { label, medianMs: median, p90Ms: p90, runs: timesMs.length };
}

export async function timeSession(
  label: string,
  session: BenchSession,
  feeds: Record<string, { data: Float32Array; dims: number[] }>,
  warmup = 5,
  runs = 20,
): Promise<BenchResult> {
  for (let i = 0; i < warmup; i++) await session.run(feeds);
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    await session.run(feeds);
    times.push(performance.now() - t0);
  }
  return summarize(label, times);
}

function printResult(r: BenchResult): void {
  console.log(
    `${r.label}: median ${r.medianMs.toFixed(2)} ms, p90 ${r.p90Ms.toFixed(2)} ms (n=${r.runs})`,
  );
}

// Node campplus smoke: real graph, real fbank, CPU only. This validates the
// bench harness end to end before lm/dit exports exist; it is NOT the gate.
async function campplusSmoke(): Promise<void> {
  const path = process.env.MING_BENCH_CAMPPLUS_ONNX;
  if (!path) {
    console.log('MING_BENCH_CAMPPLUS_ONNX not set.');
    console.log('Gate status: NOT RUN — lm/dit ONNX exports + a GPU browser are required.');
    console.log('See notes/STEP2-GATE-PROTOCOL.md for the exact protocol.');
    return;
  }
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(path, { executionProviders: ['cpu'] });
  const sr = 16000;
  const n = sr; // 1 s of 440 Hz sine
  const wav = new Float32Array(n);
  for (let i = 0; i < n; i++) wav[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / sr);
  const feats = speakerFeatures(wav, sr);
  const flat = new Float32Array(feats.flat());
  const feeds = { input: new ort.Tensor('float32', flat, [1, feats.length, 80]) };
  const times: number[] = [];
  for (let i = 0; i < 5; i++) await session.run(feeds);
  for (let i = 0; i < 20; i++) {
    const t0 = performance.now();
    const out = await session.run(feeds);
    times.push(performance.now() - t0);
    if (i === 0) {
      const emb = out['output'].data as Float32Array;
      let norm = 0;
      for (const v of emb) norm += v * v;
      console.log(`campplus output: dims [${out['output'].dims}], L2 norm ${Math.sqrt(norm).toFixed(4)}`);
    }
  }
  printResult(summarize('campplus/cpu (smoke, NOT the gate)', times));
  console.log('Gate status: NOT RUN — lm/dit exports + WebGPU browser still required.');
}

if (import.meta.main) {
  await campplusSmoke();
}
