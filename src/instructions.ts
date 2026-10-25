// Task + instruction contracts, ported verbatim from upstream
// inclusionAI/Ming-3.0-tts @ 028c1eace (tts_tasks.py).
// The strings and JSON key order here are part of the model input protocol:
// the checkpoint was trained against these exact captions. Do not paraphrase.

export const DEFAULT_TTS_PROMPT =
  'Please generate speech based on the following description.\n';

export const PARALINGUISTIC_TAGS = [
  '[Breathing]',
  '[Cough]',
  '[Sigh]',
  '[Laughter]',
  '[Surprise-ah]',
  '[Surprise-oh]',
  '[Dissatisfaction-hnn]',
  '[Uhm]',
  '[Confirmation-en]',
  '[Question-ei]',
  '[Question-ah]',
  '[Question-en]',
  '[Question-oh]',
  '[Question-huh]',
] as const;

export const TARGET_LANGUAGE_NAMES: Record<string, string> = {
  zh: 'Chinese', 'zh-cn': 'Chinese', chinese: 'Chinese',
  en: 'English', english: 'English',
  ja: 'Japanese', japanese: 'Japanese',
  ko: 'Korean', korean: 'Korean',
  de: 'German', german: 'German',
  fr: 'French', french: 'French',
  es: 'Spanish', spanish: 'Spanish',
  yue: 'Chinese', 'zh-yue': 'Chinese', cantonese: 'Chinese',
};

export const MULTILINGUAL_LANGUAGE_IDS: Record<string, string> = {
  ja: 'Japanese', ko: 'Korean', de: 'German', fr: 'French', es: 'Spanish',
};

/** BASIC attribute values are CHINESE words in the trained caption. */
export const BASIC_ATTRIBUTE_VALUES = {
  speed: ['慢速', '中速', '快速'],
  pitch: ['低', '中', '高'],
  volume: ['低', '中', '高'],
} as const;

export const EMOTION_VALUES = [
  '高兴', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶', '中性',
] as const;

export const DIALECT_VALUES = ['chuanyu dialect', 'cantonese'] as const;

export const TASKS = [
  'TTS', 'SEED', 'PLAIN', 'MULTILINGUAL', 'MULTILINGUAL_TTS', 'BASIC',
  'EMOTION', 'DIALECT', 'IP', 'VOICE_DESIGN', 'STYLE', 'FREEFORM',
] as const;
export type TaskName = (typeof TASKS)[number];

/** Runtime tasknames after alias resolution (upstream _TASK_ALIASES). */
export type RuntimeTask = 'TTS' | 'BASIC' | 'EMOTION' | 'DIALECT' | 'IP' | 'STYLE';

const TASK_ALIASES: Record<string, RuntimeTask> = {
  TTS: 'TTS', SEED: 'TTS', PLAIN: 'TTS',
  MULTILINGUAL: 'TTS', MULTILINGUAL_TTS: 'TTS',
  BASIC: 'BASIC', EMOTION: 'EMOTION', DIALECT: 'DIALECT', IP: 'IP',
  VOICE_DESIGN: 'STYLE', STYLE: 'STYLE', FREEFORM: 'STYLE',
};

export function resolveRuntimeTask(task: string): RuntimeTask {
  const key = task.trim().toUpperCase().replace(/-/g, '_');
  const rt = TASK_ALIASES[key];
  if (!rt) {
    throw new Error(
      `Unsupported Ming TTS task ${JSON.stringify(task)}; expected one of ${[...TASKS].sort().join(', ')}`,
    );
  }
  return rt;
}

export function buildTtsPrompt(targetLanguage?: string | null): string {
  if (targetLanguage === null || targetLanguage === undefined || !String(targetLanguage).trim()) {
    return DEFAULT_TTS_PROMPT;
  }
  const lang = String(targetLanguage).trim();
  const name = TARGET_LANGUAGE_NAMES[lang.toLowerCase()] ?? lang;
  return DEFAULT_TTS_PROMPT + `Target language: ${name}.\n`;
}

/** True when target text must bypass Chinese/English TN (kana safety). */
export function preserveMultilingualText(languageId?: string | null): boolean {
  if (!languageId || !String(languageId).trim()) return false;
  const lang = String(languageId).trim().toLowerCase().replace(/_/g, '-');
  return !['zh', 'zh-cn', 'chinese', 'en', 'english'].includes(lang);
}

export interface StructuredControls {
  dialect?: string;
  style?: string;
  speed?: string;
  pitch?: string;
  volume?: string;
  emotion?: string;
  ip?: string;
}

const ZH_FIELDS = ['序号', '说话人', '方言', '风格', '语速', '基频', '音量', '情感'] as const;
const EN_FIELDS = ['id', 'speaker', 'dialect', 'style', 'speed', 'pitch', 'volume', 'emotion'] as const;
const BGM_FIELDS = ['Genre', 'Mood', 'Instrument', 'Theme', 'ENV', 'SNR'] as const;

/**
 * Ordered assistant-side caption consumed during SFT. Key order matters
 * (upstream builds an OrderedDict); JSON.stringify preserves insertion order
 * for string keys, and this builder inserts in the trained order.
 */
export function buildStructuredInstruction(
  controls: StructuredControls,
  schema: 'zh' | 'en' = 'zh',
): string {
  const fields = schema === 'zh' ? ZH_FIELDS : EN_FIELDS;
  const ipKey = schema === 'zh' ? 'IP' : 'ip';
  const values: Record<string, unknown> = {};
  const vals = [1, 'speaker_1', controls.dialect ?? null, controls.style ?? null,
    controls.speed ?? null, controls.pitch ?? null, controls.volume ?? null,
    controls.emotion ?? null];
  fields.forEach((f, i) => {
    values[f] = vals[i];
  });
  const bgm: Record<string, null> = {};
  for (const f of BGM_FIELDS) bgm[f] = null;
  values['BGM'] = bgm;
  values[ipKey] = controls.ip ?? null;
  return JSON.stringify({ audio_sequence: [values] });
}

const ALLOWED_CONTROLS: Record<string, (keyof StructuredControls)[]> = {
  BASIC: ['speed', 'pitch', 'volume'],
  DIALECT: ['dialect'],
  EMOTION: ['emotion'],
  VOICE_DESIGN: ['style'],
  STYLE: ['style'],
  IP: ['ip'],
};

export function buildTaskInstruction(
  task: string,
  controls: StructuredControls,
  schema?: 'zh' | 'en',
): string {
  const t = task.trim().toUpperCase();
  const allowed = ALLOWED_CONTROLS[t];
  if (!allowed) throw new Error(`Unsupported TTS control task: ${JSON.stringify(task)}`);
  const unknown = (Object.keys(controls) as (keyof StructuredControls)[]).filter(
    (k) => controls[k] !== undefined && !allowed.includes(k),
  );
  if (unknown.length > 0) {
    throw new Error(`${t} does not accept control fields: ${JSON.stringify(unknown.sort())}`);
  }
  const resolvedSchema = schema ?? (t === 'DIALECT' ? 'en' : 'zh');
  return buildStructuredInstruction(controls, resolvedSchema);
}

export interface TaskRequest {
  prompt: string;
  text: string;
  useSpkEmb: boolean;
  useZeroSpkEmb: boolean;
  instruction: string | null;
  cfg: number;
  sigma: number;
  temperature: number;
  taskname: RuntimeTask;
  textLanguage: string | null;
}

/**
 * Resolve a task request into the fields consumed by the model.
 * Mirrors upstream build_task_generation_kwargs validation, including:
 * FREEFORM requires an instruction and forbids controls; structured tasks
 * forbid a raw instruction and require controls; MULTILINGUAL requires a
 * language_id; DIALECT ignores language_id; useSpkEmb + useZeroSpkEmb
 * together is an error; useSpkEmb needs a prompt wav or saved embedding.
 */
export function buildTaskGenerationKwargs(opts: {
  task: string;
  text: string;
  languageId?: string | null;
  targetLanguage?: string | null;
  instruction?: string | null;
  controls?: StructuredControls | null;
  instructionSchema?: 'zh' | 'en';
  promptWav?: unknown;
  promptText?: string | null;
  spkEmbPath?: unknown;
  useSpkEmb?: boolean;
  useZeroSpkEmb?: boolean;
  cfg?: number;
  sigma?: number;
  temperature?: number;
}): TaskRequest {
  const task = opts.task.trim().toUpperCase().replace(/-/g, '_');
  const runtimeTask = resolveRuntimeTask(task);
  let languageId = opts.languageId ?? null;
  let targetLanguage = opts.targetLanguage ?? null;
  let dialectWarned = false;
  if (task === 'DIALECT' && ((languageId && String(languageId).trim()) || (targetLanguage && String(targetLanguage).trim()))) {
    // Upstream warns and ignores; the browser API surfaces it as a flag on
    // the returned request via a console warning (callers that need it
    // strictly should check before calling).
    if (typeof console !== 'undefined') {
      console.warn('DIALECT does not use a language ID; ignoring language_id/target_language');
    }
    dialectWarned = true;
    languageId = null;
    targetLanguage = null;
  }
  void dialectWarned;
  if (languageId !== null && targetLanguage !== null) {
    if (String(languageId).trim().toLowerCase() !== String(targetLanguage).trim().toLowerCase()) {
      throw new Error('language_id and target_language must not conflict');
    }
  }
  languageId = languageId !== null ? languageId : targetLanguage;
  if ((task === 'MULTILINGUAL' || task === 'MULTILINGUAL_TTS') && (!languageId || !String(languageId).trim())) {
    throw new Error('MULTILINGUAL requires language_id');
  }
  const text = String(opts.text).trim();
  if (!text) throw new Error('text must not be empty');
  const useSpkEmb = !!opts.useSpkEmb;
  const useZeroSpkEmb = !!opts.useZeroSpkEmb;
  if (useSpkEmb && useZeroSpkEmb) {
    throw new Error('use_spk_emb and use_zero_spk_emb cannot both be true');
  }
  if (useSpkEmb && !(opts.promptWav || opts.spkEmbPath)) {
    throw new Error('use_spk_emb requires a prompt wav or a speaker embedding');
  }
  const controls = { ...(opts.controls ?? {}) };
  let taskInstruction: string | null;
  if (task === 'FREEFORM') {
    if (Object.keys(controls).length > 0) {
      throw new Error('FREEFORM uses instruction, not controls');
    }
    taskInstruction = opts.instruction && String(opts.instruction).trim() ? String(opts.instruction).trim() : null;
    if (taskInstruction === null) throw new Error('FREEFORM requires a non-empty instruction');
  } else if (['BASIC', 'EMOTION', 'DIALECT', 'IP', 'VOICE_DESIGN', 'STYLE'].includes(task)) {
    if (opts.instruction !== null && opts.instruction !== undefined) {
      throw new Error(`${task} builds its instruction from controls; do not pass instruction`);
    }
    if (Object.keys(controls).length === 0) throw new Error(`${task} requires task controls`);
    const promptTask = task === 'STYLE' ? 'VOICE_DESIGN' : task;
    taskInstruction = buildTaskInstruction(promptTask, controls, opts.instructionSchema);
  } else {
    if (Object.keys(controls).length > 0) {
      throw new Error(`${task} does not accept controls`);
    }
    taskInstruction =
      opts.instruction && String(opts.instruction).trim() ? String(opts.instruction).trim() : null;
  }
  if (task === 'BASIC') {
    for (const k of ['speed', 'pitch', 'volume'] as const) {
      if (controls[k] === undefined) {
        throw new Error(`BASIC requires all three of speed, pitch, volume (missing ${k})`);
      }
    }
  }
  return {
    prompt: buildTtsPrompt(languageId),
    text,
    useSpkEmb,
    useZeroSpkEmb,
    instruction: taskInstruction,
    cfg: opts.cfg ?? 2.0,
    sigma: opts.sigma ?? 0.25,
    temperature: opts.temperature ?? 0.0,
    taskname: runtimeTask,
    textLanguage: languageId,
  };
}
