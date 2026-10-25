import { describe, expect, test } from 'bun:test';
import { decodeWav, encodeWav, resample, kaldiFbank, speakerFeatures } from '../src/audio.ts';
import { prepareText, cutTextBySemanticLength, getSemanticLength, addPeriod } from '../src/segment.ts';
import { normalizeText, integerToWords, decimalToWords, ordinalToWords } from '../src/normalize.ts';

describe('audio', () => {
  test('wav encode/decode round-trips 16-bit PCM', () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1, -1, 0.25]);
    const bytes = encodeWav(samples, 44100);
    const { samples: back, sampleRate } = decodeWav(bytes);
    expect(sampleRate).toBe(44100);
    expect(back.length).toBe(samples.length);
    for (let i = 0; i < samples.length; i++) {
      expect(Math.abs(back[i] - samples[i])).toBeLessThan(1 / 32767 + 1e-6);
    }
  });
  test('decodeWav rejects non-wav bytes', () => {
    expect(() => decodeWav(new Uint8Array([1, 2, 3, 4]))).toThrow();
  });
  test('resample identity at equal rates (float32 tolerance)', () => {
    const x = new Float32Array([0.1, 0.2, 0.3]);
    const y = resample(x, 16000, 16000);
    expect(y.length).toBe(3);
    for (let i = 0; i < 3; i++) expect(Math.abs(y[i] - x[i])).toBeLessThan(1e-7);
  });
  test('resample 16k -> 8k halves the length (approx)', () => {
    const x = new Float32Array(1600).map((_, i) => Math.sin((2 * Math.PI * 440 * i) / 16000));
    const y = resample(x, 16000, 8000);
    expect(y.length).toBe(800);
    expect([...y].every((v) => Number.isFinite(v))).toBe(true);
  });
  test('kaldiFbank on 1s of 440Hz sine at 16k: 98 frames x 80 bins', () => {
    const x = new Float32Array(16000).map((_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000));
    const feats = kaldiFbank(x);
    // frames = 1 + (16000-400)//160 = 98
    expect(feats.length).toBe(98);
    expect(feats[0].length).toBe(80);
    expect(feats.flat().every((v) => Number.isFinite(v))).toBe(true);
  });
  test('speakerFeatures mean-subtracts per bin (column means ~0)', () => {
    const x = new Float32Array(16000).map((_, i) => 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000));
    const feats = speakerFeatures(x, 16000);
    expect(feats.length).toBe(98);
    for (let b = 0; b < 80; b++) {
      const mean = feats.reduce((a, row) => a + row[b], 0) / feats.length;
      expect(Math.abs(mean)).toBeLessThan(1e-4);
    }
  });
  test('speakerFeatures on too-short audio returns empty (snip_edges)', () => {
    expect(speakerFeatures(new Float32Array(100), 16000)).toEqual([]);
  });
});

describe('segment', () => {
  test('addPeriod appends . for English, 。 for Chinese, upgrades weak tails', () => {
    expect(addPeriod('Hello')).toBe('Hello.');
    expect(addPeriod('你好')).toBe('你好。');
    expect(addPeriod('Hello,')).toBe('Hello.');
    expect(addPeriod('Hello.')).toBe('Hello.');
    expect(addPeriod('')).toBe('.');
  });
  test('semantic length counts Han chars and English words', () => {
    expect(getSemanticLength('hello world')).toBe(2);
    expect(getSemanticLength('你好世界')).toBe(4);
    expect(getSemanticLength('hello 你好')).toBe(3);
  });
  test('short text is a single fragment', () => {
    expect(prepareText('Hello, this is a test.')).toEqual(['Hello, this is a test.']);
  });
  test('long English text fragments are each <= 50 semantic units', () => {
    const sentence = 'The quick brown fox jumps over the lazy dog. ';
    const text = sentence.repeat(20);
    const frags = prepareText(text);
    expect(frags.length).toBeGreaterThan(1);
    for (const f of frags) {
      expect(getSemanticLength(f)).toBeLessThanOrEqual(50);
    }
    // No words lost: joining fragments (with spaces) contains all sentences' words
    expect(frags.join(' ').split(/\s+/).length).toBeGreaterThan(100);
  });
  test('decimal points do not split sentences', () => {
    const frags = cutTextBySemanticLength('The value is 3.14 and it matters.');
    expect(frags.join('')).toContain('3.14');
  });
  test('empty / punctuation-only text yields no fragments', () => {
    expect(cutTextBySemanticLength('...')).toEqual([]);
    expect(cutTextBySemanticLength('   ')).toEqual([]);
  });
});

describe('normalize (JS subset, NOT pynini parity)', () => {
  test('integers and decimals expand', () => {
    expect(integerToWords(42)).toBe('forty two');
    expect(integerToWords(1000)).toBe('one thousand');
    expect(decimalToWords('3.14')).toBe('three point one four');
    expect(normalizeText('I have 42 apples.')).toContain('forty two');
  });
  test('ordinals expand', () => {
    expect(ordinalToWords('1', 'st')).toBe('first');
    expect(ordinalToWords('2', 'nd')).toBe('second');
    expect(ordinalToWords('3', 'rd')).toBe('third');
    expect(ordinalToWords('21', 'st')).toBe('twenty first');
    expect(normalizeText('It is my 1st time.')).toContain('first');
  });
  test('money and percent expand', () => {
    expect(normalizeText('It costs $12.99.')).toContain('dollars');
    expect(normalizeText('It costs $12.99.')).toContain('cents');
    expect(normalizeText('Fifty percent off.')).toBe('Fifty percent off.');
    expect(normalizeText('Save 50%.')).toContain('fifty percent');
  });
  test('units expand', () => {
    expect(normalizeText('Wait 10ms.')).toContain('milliseconds');
    expect(normalizeText('It weighs 5 kg.')).toContain('kilograms');
  });
  test('dates expand', () => {
    const out = normalizeText('On 2024-01-05 we met.');
    expect(out).toContain('January');
    expect(out).toContain('fifth');
    expect(out).toContain('twenty twenty four');
  });
  test('paralinguistic tags survive normalization verbatim', () => {
    const out = normalizeText('Hello [Laughter] I have 3 apples.');
    expect(out).toContain('[Laughter]');
    expect(out).toContain('three');
    expect(out).not.toContain('[laughter]');
  });
  test('markdown backticks are stripped', () => {
    expect(normalizeText('Say `hello` now.')).toBe('Say hello now.');
  });
});
