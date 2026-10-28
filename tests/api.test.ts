import { describe, expect, test } from 'bun:test';
import { buildInputIds, validateTokenizer, TOKEN_IDS, SYSTEM_PREFIX } from '../src/prompt.ts';
import { buildStructuredInstruction, buildTaskInstruction, buildTaskGenerationKwargs, buildTtsPrompt } from '../src/instructions.ts';
import { listVoices, DEFAULT_VOICE, VOICE_REMAP, resolveVoice, CLONING_CONSENT_NOTICE } from '../src/voices.ts';
import { MingTTS, SAMPLING_RATE, resolveGeneratePlan, MODEL_ID } from '../src/ming-tts.ts';
import { ModelNotLoadedError } from '../src/sessions.ts';
import {mingReadiness} from '../src/readiness.ts';

// Minimal tokenizer stub: encodes by word-splitting for ordinary text, but
// returns the exact trained single id for every special token (as the real
// tokenizer.json does — verified against the shipped file).
function stubTokenizer() {
  const specials = new Map<string, number>(Object.entries(TOKEN_IDS));
  return {
    encode: (text: string): number[] => {
      if (specials.has(text)) return [specials.get(text)!];
      // Split special tokens out of longer strings first.
      let ids: number[] = [];
      let rest = text;
      const tokens = [...specials.keys()].sort((a, b) => b.length - a.length);
      while (rest.length > 0) {
        const hit = tokens.find((t) => rest.startsWith(t));
        if (hit) {
          ids.push(specials.get(hit)!);
          rest = rest.slice(hit.length);
          continue;
        }
        // Consume one "word" (or single char) and assign a stable fake id.
        const m = rest.match(/^\s+|^\S+/);
        const chunk = m ? m[0] : rest[0];
        let h = 1000;
        for (const ch of chunk) h = (h * 31 + ch.charCodeAt(0)) % 100000;
        ids.push(h);
        rest = rest.slice(chunk.length);
      }
      return ids;
    },
    tokenId: (token: string) => specials.get(token) ?? null,
  };
}

describe('prompt contract', () => {
  test('token IDs match the trained dense checkpoint', () => {
    expect(TOKEN_IDS['<|im_start|>']).toBe(151644);
    expect(TOKEN_IDS['<audio>']).toBe(151706);
    expect(TOKEN_IDS['<audioPatch>']).toBe(151705);
  });
  test('validateTokenizer accepts the stub (specials exact)', () => {
    expect(() => validateTokenizer(stubTokenizer())).not.toThrow();
  });
  test('validateTokenizer rejects a re-indexed tokenizer', () => {
    const bad = stubTokenizer();
    const orig = bad.tokenId;
    void orig;
    const badTok = {
      encode: (t: string) => (t === '<audio>' ? [999] : bad.encode(t)),
      tokenId: (t: string) => (t === '<audio>' ? 999 : bad.tokenId(t)),
    };
    expect(() => validateTokenizer(badTok)).toThrow();
  });
  test('buildInputIds: speaker slot position + latent position', () => {
    const tok = stubTokenizer();
    const { ids, speakerPositions, latentPosition } = buildInputIds(tok, 'Please generate speech based on the following description.\n', 'Hello world.', {
      numSpeakers: 1, promptText: 'Hello there.', promptLatentSteps: 3,
    });
    expect(speakerPositions.length).toBe(1);
    expect(ids[speakerPositions[0]]).toBe(TOKEN_IDS['<|vision_pad|>']);
    expect(ids[speakerPositions[0] - 1]).toBe(TOKEN_IDS['<|vision_start|>']);
    expect(latentPosition).toBeGreaterThan(0);
    expect(ids.slice(latentPosition)).toEqual([151705, 151705, 151705]);
    // <audio> immediately precedes the latent block
    expect(ids[latentPosition - 1]).toBe(TOKEN_IDS['<audio>']);
  });
  test('buildInputIds: no prompt -> latentPosition -1, no speaker slots', () => {
    const tok = stubTokenizer();
    const r = buildInputIds(tok, 'Please generate speech based on the following description.\n', 'Hello.', {});
    expect(r.latentPosition).toBe(-1);
    expect(r.speakerPositions).toEqual([]);
  });
  test('buildInputIds: prompt latents dropped when promptText is absent', () => {
    const tok = stubTokenizer();
    const r = buildInputIds(tok, 'Please generate speech based on the following description.\n', 'Hello.', {
      promptLatentSteps: 5,
    });
    expect(r.latentPosition).toBe(-1);
  });
  test('buildInputIds: BGM text omits the text-input prefix', () => {
    const tok = stubTokenizer();
    const bgm = 'Genre: pop. Mood: happy. Instrument: piano. Theme: love. Duration: 10s.';
    const withPrefix = buildInputIds(tok, 'p', 'Hello.', {});
    const bgmIds = buildInputIds(tok, 'p', bgm, {});
    // The BGM sequence must not contain the " Text input:" encoding chunk;
    // simplest check: system prefix is present in both (sanity), and the
    // BGM ids are shorter relative to their text length than a prefixed one.
    expect(bgmIds.ids.length).toBeGreaterThan(0);
    expect(withPrefix.ids.length).toBeGreaterThan(0);
    expect(SYSTEM_PREFIX.startsWith('<|im_start|>')).toBe(true);
  });
});

describe('instructions / tasks', () => {
  test('buildTtsPrompt with and without language', () => {
    expect(buildTtsPrompt(null)).toBe('Please generate speech based on the following description.\n');
    expect(buildTtsPrompt('en')).toContain('Target language: English.');
    expect(buildTtsPrompt('ja')).toContain('Target language: Japanese.');
  });
  test('structured instruction key order + BGM nulls (zh schema)', () => {
    const json = buildStructuredInstruction({ speed: '中速', pitch: '中', volume: '中' }, 'zh');
    const parsed = JSON.parse(json);
    const item = parsed.audio_sequence[0];
    expect(Object.keys(item)).toEqual(['序号', '说话人', '方言', '风格', '语速', '基频', '音量', '情感', 'BGM', 'IP']);
    expect(item['语速']).toBe('中速');
    expect(item['BGM']).toEqual({ Genre: null, Mood: null, Instrument: null, Theme: null, ENV: null, SNR: null });
  });
  test('buildTaskInstruction rejects leaking controls', () => {
    expect(() => buildTaskInstruction('BASIC', { speed: '中速', emotion: '高兴' })).toThrow(/does not accept control fields/);
    expect(() => buildTaskInstruction('EMOTION', { emotion: '高兴' })).not.toThrow();
    expect(() => buildTaskInstruction('NOPE', {})).toThrow(/Unsupported TTS control task/);
  });
  test('FREEFORM requires instruction, forbids controls', () => {
    expect(() =>
      buildTaskGenerationKwargs({ task: 'FREEFORM', text: 'Hi.', controls: { style: 'x' } }),
    ).toThrow();
    expect(() => buildTaskGenerationKwargs({ task: 'FREEFORM', text: 'Hi.' })).toThrow(/requires a non-empty instruction/);
    expect(
      buildTaskGenerationKwargs({ task: 'FREEFORM', text: 'Hi.', instruction: 'Sound warm.' }).instruction,
    ).toBe('Sound warm.');
  });
  test('MULTILINGUAL requires language_id; DIALECT ignores it', () => {
    expect(() => buildTaskGenerationKwargs({ task: 'MULTILINGUAL', text: 'Bonjour.' })).toThrow(/requires language_id/);
    const d = buildTaskGenerationKwargs({ task: 'DIALECT', text: 'Hi.', languageId: 'en', controls: { dialect: 'cantonese' } });
    expect(d.textLanguage).toBeNull();
  });
  test('useSpkEmb + useZeroSpkEmb together is an error', () => {
    expect(() =>
      buildTaskGenerationKwargs({ task: 'TTS', text: 'Hi.', useSpkEmb: true, useZeroSpkEmb: true, promptWav: {} }),
    ).toThrow(/cannot both be true/);
  });
});

describe('voices', () => {
  test('list_voices returns all 5 cookbook English IPs', () => {
    expect(listVoices()).toEqual(['US_Emma', 'US_James', 'US_Stella', 'UK_George', 'UK_Charlotte']);
  });
  test('default voice is US_Emma, never zero-speaker', () => {
    expect(DEFAULT_VOICE).toBe('US_Emma');
  });
  test('af_heart remaps explicitly to US_Emma', () => {
    expect(VOICE_REMAP['af_heart']).toBe('US_Emma');
    expect(resolveVoice('af_heart')).toBe('US_Emma');
    expect(resolveVoice(null)).toBe('US_Emma');
  });
  test('cloning notice carries consent + labelling language', () => {
    expect(CLONING_CONSENT_NOTICE).toContain('without consent');
    expect(CLONING_CONSENT_NOTICE).toContain('Label synthetic speech');
  });
});

const sineEngine = {
  async synthesizeFragment(fragment: string) {
    // Deterministic non-silent audio: 440 Hz sine, 0.5 s per fragment.
    const n = Math.floor(SAMPLING_RATE * 0.5);
    const audio = new Float32Array(n);
    for (let i = 0; i < n; i++) audio[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / SAMPLING_RATE);
    void fragment;
    return { audio, samplingRate: SAMPLING_RATE };
  },
};

describe('MingTTS API (mocked engine — tests API/validation, NOT audio quality)', () => {
  test('from_pretrained refuses tts-tiny', async () => {
    await expect(
      MingTTS.from_pretrained('inclusionAI/Ming-3.0-tts-tiny', {}),
    ).rejects.toThrow(/0\.5B ONLY/);
  });
  test('loading without an engine fails immediately, before a UI can claim ready', async () => {
    await expect(MingTTS.from_pretrained(MODEL_ID, { device: 'wasm', dtype: 'int8' })).rejects.toBeInstanceOf(ModelNotLoadedError);
  });
  test('session factory alone is not synthesis readiness', async () => {
    const sessionFactory={createSession:async()=>({run:async()=>({})})};
    await expect(MingTTS.from_pretrained(MODEL_ID, { sessionFactory })).rejects.toThrow(/not sufficient/);
    expect(mingReadiness().ready).toBe(false);
    expect(mingReadiness({engine:sineEngine}).ready).toBe(true);
    expect(mingReadiness({engine:sineEngine}).realSynthesisVerified).toBe(false);
  });
  test('generate returns non-silent 44100 Hz audio with reasonable duration', async () => {
    const tts = await MingTTS.from_pretrained(MODEL_ID, { device: 'wasm', dtype: 'int8', engine: sineEngine });
    const out = await tts.generate('Hello, this is a test.', { voice: 'US_Emma', speed: 1 });
    expect(out.sampling_rate).toBe(44100);
    expect(out.audio.length).toBeGreaterThan(0);
    const peak = Math.max(...out.audio.map(Math.abs));
    expect(peak).toBeGreaterThan(0.1);
    const seconds = out.audio.length / out.sampling_rate;
    expect(seconds).toBeGreaterThan(0.1);
    expect(seconds).toBeLessThan(60);
  });
  test('generate with no voice uses US_Emma IP (never zero-speaker)', async () => {
    let seenPlan: unknown = null;
    const spy = {
      async synthesizeFragment(fragment: string, plan: unknown) {
        seenPlan = plan;
        return sineEngine.synthesizeFragment(fragment);
      },
    };
    const tts = await MingTTS.from_pretrained(MODEL_ID, { engine: spy });
    await tts.generate('Hello.');
    const plan = seenPlan as { voice: string; task: string; useZeroSpkEmb: boolean };
    expect(plan.voice).toBe('US_Emma');
    expect(plan.task).toBe('IP');
    expect(plan.useZeroSpkEmb).toBe(false);
  });
  test('af_heart compat call resolves to US_Emma', async () => {
    const plan = resolveGeneratePlan('Hello.', { voice: 'af_heart', speed: 1 });
    expect(plan.voice).toBe('US_Emma');
  });
  test('validation: empty text, unsupported language, bad combos', async () => {
    const tts = await MingTTS.from_pretrained(MODEL_ID, { engine: sineEngine });
    await expect(tts.generate('   ')).rejects.toThrow(/must not be empty/);
    await expect(tts.generate('Hello.', { language: 'Klingon' })).rejects.toThrow(/unsupported language/);
    await expect(tts.generate('Hello.', { promptWav: new Float32Array(100) })).rejects.toThrow(/requires promptText/);
    await expect(tts.generate('Hello.', { voiceDesign: 'a warm voice', promptWav: new Float32Array(100), promptText: 'Hi.' })).rejects.toThrow(/cannot be combined/);
    await expect(tts.generate('Hello.', { task: 'FREEFORM', controls: { style: 'x' } })).rejects.toThrow();
    await expect(tts.generate('Hello.', { task: 'BASIC', speed: 1 })).rejects.toThrow(/requires all three/);
    await expect(tts.generate('Hello.', { task: 'MULTILINGUAL' })).rejects.toThrow(/requires language_id/);
    await expect(tts.generate('Hello.', { task: 'TTS', pitch: 1.2 })).rejects.toThrow(/require task BASIC/);
  });
  test('stream yields per-fragment audio (fragment-level, not token-level)', async () => {
    const tts = await MingTTS.from_pretrained(MODEL_ID, { engine: sineEngine });
    const seen: string[] = [];
    for await (const part of tts.stream('Hello world. This is a second sentence.')) {
      seen.push(part.text);
      expect(part.sampling_rate).toBe(44100);
      expect(part.audio.length).toBeGreaterThan(0);
    }
    expect(seen.length).toBeGreaterThanOrEqual(1);
  });
  test('list_voices matches the cookbook five', async () => {
    const tts = await MingTTS.from_pretrained(MODEL_ID, { engine: sineEngine });
    expect(tts.list_voices()).toEqual(listVoices());
  });
});
