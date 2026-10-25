import { describe, expect, test } from 'bun:test';
import { shardRanges, QuotaExceededError } from '../src/cache.ts';
import { defaultModelPaths, executionProvidersFor, GRAPH_NAMES } from '../src/sessions.ts';

describe('cache / sessions plumbing', () => {
  test('shardRanges covers the whole file without gaps', () => {
    const ranges = shardRanges(1000, 4);
    expect(ranges).toEqual([
      { start: 0, end: 249 },
      { start: 250, end: 499 },
      { start: 500, end: 749 },
      { start: 750, end: 999 },
    ]);
  });
  test('shardRanges with more shards than bytes caps at the file size', () => {
    const ranges = shardRanges(3, 10);
    expect(ranges[ranges.length - 1].end).toBe(2);
    expect(ranges[0].start).toBe(0);
  });
  test('shardRanges rejects non-positive inputs', () => {
    expect(() => shardRanges(0, 4)).toThrow();
    expect(() => shardRanges(100, 0)).toThrow();
  });
  test('QuotaExceededError has a useful message', () => {
    const e = new QuotaExceededError();
    expect(e.message).toContain('quota');
  });
  test('all 7 graphs are named', () => {
    expect([...GRAPH_NAMES].sort()).toEqual(
      ['aggregator', 'campplus', 'dit', 'enc_agg', 'enc_backbone', 'lm', 'voc'].sort() as typeof GRAPH_NAMES[number][],
    );
  });
  test('defaultModelPaths lays out graphs/assets/tokenizer per §9.1', () => {
    const p = defaultModelPaths('https://example.com/models', 'int8');
    expect(p.graph('lm')).toBe('https://example.com/models/int8/lm.onnx');
    expect(p.graph('campplus')).toBe('https://example.com/models/int8/campplus.onnx');
    expect(p.dimsUrl).toBe('https://example.com/models/assets/dims.json');
    expect(p.tokenizerUrl).toBe('https://example.com/models/tokenizer/tokenizer.json');
  });
  test('execution providers: webgpu first with wasm fallback, wasm alone otherwise', () => {
    expect(executionProvidersFor('webgpu')).toEqual(['webgpu', 'wasm']);
    expect(executionProvidersFor('wasm')).toEqual(['wasm']);
  });
});
