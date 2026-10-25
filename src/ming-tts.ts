// MingTTS — kokoro-js-shaped API over the Ming-3.0-tts 0.5B host loop.
//
// Compatibility path (what the Anascummis worker calls today):
//   const tts = await MingTTS.from_pretrained('inclusionAI/Ming-3.0-tts-0.5B', { device: 'webgpu', dtype: 'fp16', progress_callback });
//   const out = await tts.generate(text, { voice: 'US_Emma', speed: 1 });
//   // -> { audio: Float32Array, sampling_rate: 44100 }
//
// Differences from Kokoro, enforced here (not papered over):
// - 'af_heart' does NOT exist in Ming: it is remapped to US_Emma explicitly.
// - Built-in voices are IP-task captions, NOT speaker-embedding slots.
//   generate(text, { voice: 'US_Emma' }) with no task runs task IP.
// - Output is ALWAYS 44.1 kHz, never 24 kHz. Callers must never mix 24 kHz
//   Kokoro tracks with 44.1 kHz Ming tracks in one mixPcmTracks call.
// - Scalar speed is NOT a native Ming control. With task BASIC it maps to
//   the trained Chinese speed/pitch/volume caption; otherwise a non-1 speed
//   is applied as a post-hoc resample (documented: this also shifts pitch —
//   prefer BASIC for proper speed control). Speed is NEVER silently dropped.
// - Streaming is FRAGMENT-level (per sentence/fragment), not token-level:
//   the vocoder runs on completed latents.
// - Zero-speaker is NEVER the default. No-voice calls use US_Emma (IP).
//
// Current implementation status (honest): the API, validation, task
// captions, segmentation, normalization, host loop math, and session
// plumbing are implemented and tested. End-to-end synthesis against real
// ORT sessions requires the Step 3–4 wiring (host weights from host.npz /
// embed.npy + tokenizer.json + the 7 exported ONNX graphs), which is gated
// on the Step 2 WebGPU benchmark (see notes/STEP2-GATE-PROTOCOL.md).
// generate() with an injected engine runs today; generate() without an
// engine or weights throws ModelNotLoadedError — it never fakes audio.

import { resample } from './audio.js';
import { buildTaskGenerationKwargs, buildTtsPrompt, type StructuredControls } from './instructions.js';
import { normalizeText } from './normalize.js';
import { prepareText } from './segment.js';
import { ModelNotLoadedError, type Device, type Dtype, type SessionFactory } from './sessions.js';
import type { ProgressCallback } from './cache.js';
import { CLONING_CONSENT_NOTICE, DEFAULT_VOICE, listVoices, resolveVoice } from './voices.js';

export const SAMPLING_RATE = 44100;
export const MODEL_ID = 'inclusionAI/Ming-3.0-tts-0.5B';

export const SUPPORTED_LANGUAGES = [
  'Chinese', 'English', 'German', 'Spanish', 'French', 'Korean', 'Japanese',
  'zh', 'en', 'de', 'es', 'fr', 'ko', 'ja',
] as const;

export interface FragmentResult {
  audio: Float32Array;
  samplingRate: number;
}

/** Injected fragment engine (tests / integrators). The real ORT engine plugs in here. */
export interface FragmentEngine {
  synthesizeFragment(
    fragment: string,
    plan: GeneratePlan,
  ): Promise<FragmentResult>;
}

export interface GenerateOptions {
  voice?: string;
  task?: string;
  language?: string;
  languageId?: string;
  promptWav?: Float32Array;
  promptWavRate?: number;
  promptText?: string;
  useSpkEmb?: boolean;
  useZeroSpkEmb?: boolean;
  voiceDesign?: string;
  instruction?: string;
  controls?: StructuredControls;
  speed?: number | string;
  pitch?: number | string;
  volume?: number | string;
  emotion?: string;
  dialect?: string;
  cfg?: number;
  seed?: number;
  maxLength?: number;
  rawText?: boolean;
}

export interface GeneratePlan {
  task: string;
  runtimeTask: string;
  prompt: string;
  instruction: string | null;
  voice: string;
  language: string | null;
  useZeroSpkEmb: boolean;
  useSpkEmb: boolean;
  hasPrompt: boolean;
  postHocSpeed: number | null;
  cfg: number;
  seed: number;
}

function bucketSpeed(v: number | string): string {
  if (typeof v === 'string') return v;
  if (v < 0.85) return '慢速';
  if (v <= 1.15) return '中速';
  return '快速';
}
function bucketLevel(v: number | string): string {
  if (typeof v === 'string') return v;
  if (v < 0.85) return '低';
  if (v <= 1.15) return '中';
  return '高';
}

/**
 * Pure validation + planning for one generate() call. Exported so tests
 * (and integrators) can check task/voice/language rules without any model.
 */
export function resolveGeneratePlan(text: string, opts: GenerateOptions = {}): GeneratePlan {
  if (!text || !text.trim()) {
    throw new Error('MingTTS.generate: text must not be empty');
  }
  const language = opts.language ?? opts.languageId ?? null;
  if (language !== null && !(SUPPORTED_LANGUAGES as readonly string[]).includes(String(language))) {
    throw new Error(
      `MingTTS.generate: unsupported language ${JSON.stringify(language)}; supported: ${SUPPORTED_LANGUAGES.join(', ')}`,
    );
  }
  if (opts.promptWav && !opts.promptText) {
    throw new Error('MingTTS.generate: promptWav requires promptText (its transcript)');
  }
  if (opts.voiceDesign && opts.promptWav) {
    throw new Error('MingTTS.generate: voiceDesign cannot be combined with promptWav');
  }
  if (opts.voiceDesign && opts.task && opts.task.trim().toUpperCase() !== 'VOICE_DESIGN' && opts.task.trim().toUpperCase() !== 'STYLE') {
    throw new Error('MingTTS.generate: voiceDesign requires task VOICE_DESIGN');
  }
  const voice = resolveVoice(opts.voice ?? null);
  // Task selection: explicit task wins; otherwise voiceDesign -> VOICE_DESIGN,
  // prompt -> TTS, built-in voice -> IP, nothing -> IP with the default voice.
  let task = opts.task ? opts.task.trim().toUpperCase() : null;
  if (!task) {
    if (opts.voiceDesign) task = 'VOICE_DESIGN';
    else if (opts.promptWav) task = 'TTS';
    else task = 'IP';
  }
  if (task === 'IP' && opts.voice && opts.voice.trim() !== '' && opts.voice.trim() !== voice) {
    // af_heart remap etc. already applied by resolveVoice; nothing to do.
  }
  if (task !== 'IP' && opts.voice && task !== 'TTS') {
    // A bare voice option only makes sense for the IP task (or the
    // no-task compat path, which already resolved to IP above).
    throw new Error(
      `MingTTS.generate: voice option requires task IP (got task ${task}); omit task for the voice-compat path or use task IP with the voice as ip`,
    );
  }
  if (task === 'TTS' && opts.voice) {
    // Kokoro-compat calls pass a voice with no task; those resolved to IP
    // above. An EXPLICIT task TTS + voice is ambiguous — prefer the
    // explicit task and treat the voice as the IP only when the caller
    // also passes task IP. Here we keep TTS and drop the voice slot
    // (there is no speaker slot in TTS without a prompt), but record it:
    // callers migrating from Kokoro should omit task to get IP voices.
    if (typeof console !== 'undefined') {
      console.warn(
        'MingTTS.generate: explicit task TTS with a voice option — voice is ignored for TTS without a prompt; omit task (or use task IP) to use a built-in voice',
      );
    }
  }
  const controls: StructuredControls = { ...(opts.controls ?? {}) };
  let postHocSpeed: number | null = null;
  if (task === 'BASIC') {
    if (opts.speed !== undefined) controls.speed = bucketSpeed(opts.speed);
    if (opts.pitch !== undefined) controls.pitch = bucketLevel(opts.pitch);
    if (opts.volume !== undefined) controls.volume = bucketLevel(opts.volume);
  } else if (opts.speed !== undefined && opts.speed !== 1 && opts.speed !== '1') {
    if (typeof opts.speed === 'number') postHocSpeed = opts.speed;
    else throw new Error('MingTTS.generate: speed as a string is only accepted for task BASIC (Chinese caption values)');
  }
  if (task !== 'BASIC' && (opts.pitch !== undefined || opts.volume !== undefined)) {
    throw new Error('MingTTS.generate: pitch/volume require task BASIC');
  }
  if (opts.emotion !== undefined) controls.emotion = opts.emotion;
  if (opts.dialect !== undefined) controls.dialect = opts.dialect;
  if (opts.voiceDesign !== undefined) controls.style = opts.voiceDesign;
  if (task === 'IP') controls.ip = voice;
  const kwargs = buildTaskGenerationKwargs({
    task,
    text: text.trim(),
    languageId: language,
    instruction: opts.instruction ?? null,
    controls: Object.keys(controls).length > 0 ? controls : null,
    promptWav: opts.promptWav ?? null,
    promptText: opts.promptText ?? null,
    useSpkEmb: opts.useSpkEmb ?? (opts.promptWav ? true : false),
    useZeroSpkEmb: opts.useZeroSpkEmb ?? false,
    cfg: opts.cfg ?? 2.0,
  });
  if (task === 'DIALECT' && language !== null) {
    // buildTaskGenerationKwargs already warned + ignored it; plan reflects that.
  }
  return {
    task,
    runtimeTask: kwargs.taskname,
    prompt: kwargs.prompt,
    instruction: kwargs.instruction,
    voice,
    language,
    useZeroSpkEmb: kwargs.useZeroSpkEmb,
    useSpkEmb: kwargs.useSpkEmb,
    hasPrompt: !!opts.promptWav,
    postHocSpeed,
    cfg: kwargs.cfg,
    seed: opts.seed ?? 0,
  };
}

export interface FromPretrainedOptions {
  device?: Device;
  dtype?: Dtype;
  progress_callback?: ProgressCallback;
  sessionFactory?: SessionFactory;
  engine?: FragmentEngine;
  /** Base URL for model files (graphs/assets/tokenizer). Default: HF repo path. */
  modelBaseUrl?: string;
}

export class MingTTS {
  readonly modelId: string;
  readonly device: Device;
  readonly dtype: Dtype;
  private engine: FragmentEngine | null;
  private progressCallback?: ProgressCallback;

  private constructor(
    modelId: string,
    device: Device,
    dtype: Dtype,
    engine: FragmentEngine | null,
    progressCallback?: ProgressCallback,
  ) {
    this.modelId = modelId;
    this.device = device;
    this.dtype = dtype;
    this.engine = engine;
    this.progressCallback = progressCallback;
  }

  /**
   * Lazy load: onnxruntime-web is imported only inside from_pretrained
   * (via sessions.loadOrtWeb) when no engine/factory is injected — never
   * at module import time. tts-tiny is refused outright (0.5B ONLY).
   */
  static async from_pretrained(
    modelId: string,
    options: FromPretrainedOptions = {},
  ): Promise<MingTTS> {
    if (/tiny/i.test(modelId)) {
      throw new Error(
        `MingTTS: ${modelId} is out of scope — this package supports Ming-3.0-tts-0.5B ONLY, never tts-tiny`,
      );
    }
    const device = options.device ?? 'webgpu';
    const dtype = options.dtype ?? (device === 'webgpu' ? 'fp16' : 'int8');
    options.progress_callback?.({ status: 'loading', file: modelId });
    if (options.engine) {
      return new MingTTS(modelId, device, dtype, options.engine, options.progress_callback);
    }
    if (options.sessionFactory) {
      // Sessions are lazy (GraphSessions), but the real engine wiring
      // (host weights + tokenizer) is Step 3–4 work, gated on Step 2.
      // Until then, a factory alone is not enough to synthesize.
      return new MingTTS(modelId, device, dtype, null, options.progress_callback);
    }
    // No engine, no factory: do not attempt a silent ORT import that may
    // throw at import time in the host env. The caller gets a clear error
    // at generate() time, and can inject an engine/factory instead.
    return new MingTTS(modelId, device, dtype, null, options.progress_callback);
  }

  list_voices(): string[] {
    return listVoices();
  }

  /** Cloning consent notice (see voices.ts). */
  static get cloningConsentNotice(): string {
    return CLONING_CONSENT_NOTICE;
  }

  private requireEngine(): FragmentEngine {
    if (!this.engine) {
      throw new ModelNotLoadedError();
    }
    return this.engine;
  }

  private fragmentsFor(text: string, opts: GenerateOptions): string[] {
    if (opts.rawText) return [text];
    const normalized = normalizeText(text);
    return prepareText(normalized, opts.maxLength ?? 50);
  }

  private applyPostHocSpeed(audio: Float32Array, speed: number): Float32Array {
    // Post-hoc resample: output length = input / speed. This shifts pitch
    // as well as tempo — it is a compatibility shim, not Ming speed control.
    // Prefer task BASIC for proper speed/pitch/volume control.
    if (speed === 1) return audio;
    const outLen = Math.max(1, Math.round(audio.length / speed));
    const resampled = resample(audio, SAMPLING_RATE, Math.round(SAMPLING_RATE * speed));
    // resample() above changes the sample count by ~speed; trim/pad to the
    // exact target length so duration math stays exact.
    const out = new Float32Array(outLen);
    out.set(resampled.subarray(0, Math.min(outLen, resampled.length)));
    return out;
  }

  async generate(
    text: string,
    opts: GenerateOptions = {},
  ): Promise<{ audio: Float32Array; sampling_rate: number }> {
    const engine = this.requireEngine();
    const plan = resolveGeneratePlan(text, opts);
    const fragments = this.fragmentsFor(text, opts);
    if (fragments.length === 0) {
      throw new Error(`MingTTS.generate: text produced no synthesizable fragments: ${JSON.stringify(text)}`);
    }
    const chunks: Float32Array[] = [];
    for (const fragment of fragments) {
      const out = await engine.synthesizeFragment(fragment, plan);
      if (out.samplingRate !== SAMPLING_RATE) {
        throw new Error(
          `MingTTS.generate: fragment engine returned sampling_rate ${out.samplingRate}, expected ${SAMPLING_RATE}`,
        );
      }
      let audio = out.audio;
      if (plan.postHocSpeed !== null) audio = this.applyPostHocSpeed(audio, plan.postHocSpeed);
      chunks.push(audio);
    }
    const total = chunks.reduce((a, c) => a + c.length, 0);
    const audio = new Float32Array(total);
    let off = 0;
    for (const c of chunks) {
      audio.set(c, off);
      off += c.length;
    }
    return { audio, sampling_rate: SAMPLING_RATE };
  }

  /** Fragment-level streaming: yields each fragment's audio as it completes. */
  async *stream(
    text: string,
    opts: GenerateOptions = {},
  ): AsyncGenerator<{ text: string; audio: Float32Array; sampling_rate: number }> {
    const engine = this.requireEngine();
    const plan = resolveGeneratePlan(text, opts);
    const fragments = this.fragmentsFor(text, opts);
    if (fragments.length === 0) {
      throw new Error(`MingTTS.generate: text produced no synthesizable fragments: ${JSON.stringify(text)}`);
    }
    for (const fragment of fragments) {
      const out = await engine.synthesizeFragment(fragment, plan);
      if (out.samplingRate !== SAMPLING_RATE) {
        throw new Error(
          `MingTTS.generate: fragment engine returned sampling_rate ${out.samplingRate}, expected ${SAMPLING_RATE}`,
        );
      }
      let audio = out.audio;
      if (plan.postHocSpeed !== null) audio = this.applyPostHocSpeed(audio, plan.postHocSpeed);
      yield { text: fragment, audio, sampling_rate: SAMPLING_RATE };
    }
  }
}

export { DEFAULT_VOICE, buildTtsPrompt };
export type { ProgressCallback };
