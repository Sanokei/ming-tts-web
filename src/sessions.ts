// ORT-web session plumbing. The ONLY module that touches onnxruntime-web,
// and only via a lazy dynamic import inside loadOrtWeb() — never a
// top-level import (kokoro-js throws at import time in some environments;
// this package preserves the lazy-load pattern from the kinoinstrument
// previs export). Tests inject a SessionFactory and never touch ORT.

export interface OrtTensorLike {
  data: Float32Array | Int32Array | BigInt64Array;
  dims: number[];
}

export interface OrtSessionLike {
  run(feeds: Record<string, OrtTensorLike>): Promise<Record<string, OrtTensorLike>>;
}

export interface SessionFactory {
  createSession(
    pathOrUrl: string,
    options: { executionProviders: string[] },
  ): Promise<OrtSessionLike>;
}

export type Device = 'webgpu' | 'wasm';
export type Dtype = 'fp16' | 'int8' | 'int4';

export const GRAPH_NAMES = [
  'lm', 'dit', 'aggregator', 'enc_backbone', 'enc_agg', 'voc', 'campplus',
] as const;
export type GraphName = (typeof GRAPH_NAMES)[number];

export interface ModelPaths {
  /** Per-graph URL/path, e.g. models/int8/lm.onnx */
  graph: (name: GraphName) => string;
  /** Host assets: dims.json, host tensors, tokenizer.json */
  dimsUrl: string;
  tokenizerUrl: string;
}

/**
 * Default paths for a model directory laid out per RESEARCH §9.1:
 * models/<dtype>/{lm,dit,aggregator,enc_backbone,enc_agg,voc,campplus}.onnx
 * plus assets/dims.json and tokenizer/tokenizer.json. The caller supplies
 * the base URL (HF, CDN, or local static dir); nothing is downloaded here.
 */
export function defaultModelPaths(baseUrl: string, dtype: Dtype): ModelPaths {
  const base = baseUrl.replace(/\/$/, '');
  return {
    graph: (name) => `${base}/${dtype}/${name}.onnx`,
    dimsUrl: `${base}/assets/dims.json`,
    tokenizerUrl: `${base}/tokenizer/tokenizer.json`,
  };
}

export class ModelNotLoadedError extends Error {
  constructor(message = 'MingTTS is not ready for built-in synthesis: the real fragment engine and end-to-end gate are incomplete. Supply a verified FragmentEngine through options.engine. Installing ONNX Runtime or supplying a SessionFactory is not sufficient. For a working browser TTS path, see https://github.com/hexgrad/kokoro/tree/main/kokoro.js .') {
    super(message);
    this.name = 'ModelNotLoadedError';
  }
}

/**
 * Lazy ORT-web loader. In a browser this dynamic-imports 'onnxruntime-web'
 * on first use; in Node tests a SessionFactory is injected instead and this
 * is never called. When onnxruntime-web is absent, throws ModelNotLoadedError
 * — never fake audio, never silent zeros.
 */
export async function loadOrtWeb(): Promise<SessionFactory> {
  try {
    const ort = await import('onnxruntime-web');
    return {
      createSession: async (pathOrUrl, options) => {
        const session = await ort.InferenceSession.create(pathOrUrl, {
          executionProviders: options.executionProviders,
        });
        return {
          run: async (feeds) => {
            const ortFeeds: Record<string, unknown> = {};
            for (const [k, v] of Object.entries(feeds)) {
              ortFeeds[k] = new ort.Tensor(
                v.data instanceof BigInt64Array ? 'int64' : 'float32',
                v.data as Float32Array,
                v.dims,
              );
            }
            const out = await session.run(ortFeeds as never);
            const result: Record<string, OrtTensorLike> = {};
            for (const [k, v] of Object.entries(out)) {
              result[k] = { data: v.data as Float32Array, dims: [...v.dims] };
            }
            return result;
          },
        };
      },
    };
  } catch {
    throw new ModelNotLoadedError(
      'MingTTS: onnxruntime-web could not be loaded. In tests, inject a SessionFactory; in production, bundle onnxruntime-web.',
    );
  }
}

/** Execution providers for a device, in ORT-web order of preference. */
export function executionProvidersFor(device: Device): string[] {
  return device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
}

/**
 * Lazy per-graph session holder. campplus + enc_* are only created when a
 * prompt is supplied (cloning path); the TTS/IP path needs lm/dit/
 * aggregator/voc only. Sessions are created once and reused.
 */
export class GraphSessions {
  private cache = new Map<GraphName, OrtSessionLike>();
  constructor(
    private factory: SessionFactory,
    private paths: ModelPaths,
    private device: Device,
  ) {}

  async get(name: GraphName): Promise<OrtSessionLike> {
    const existing = this.cache.get(name);
    if (existing) return existing;
    const session = await this.factory.createSession(this.paths.graph(name), {
      executionProviders: executionProvidersFor(this.device),
    });
    this.cache.set(name, session);
    return session;
  }

  has(name: GraphName): boolean {
    return this.cache.has(name);
  }
}
