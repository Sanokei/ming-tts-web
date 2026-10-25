// Sharded download + cache + progress + quota errors.
// The model is ~1.80 GB (int8) / ~2.91 GB (fp16) — this must be an explicit
// opt-in download with progress, never an on-first-utterance lazy load
// (see RESEARCH §4.3 and the Anascummis voice-model download UX).

export type ProgressCallback = (progress: {
  status: string;
  file?: string;
  progress?: number; // 0-100 for the current file, when known
  loaded?: number;
  total?: number;
}) => void;

export class QuotaExceededError extends Error {
  constructor(message = 'MingTTS: browser storage quota exceeded while caching the model. Free up disk space or use a smaller dtype (int8), then retry.') {
    super(message);
    this.name = 'QuotaExceededError';
  }
}

export class DownloadInterruptedError extends Error {
  constructor(message = 'MingTTS: model download was interrupted. Re-run from_pretrained() to resume.') {
    super(message);
    this.name = 'DownloadInterruptedError';
  }
}

function isQuotaError(e: unknown): boolean {
  if (e instanceof Error) {
    return (
      e.name === 'QuotaExceededError' ||
      /quota/i.test(e.message) ||
      (e as { code?: number }).code === 22
    );
  }
  return false;
}

/**
 * Download one file with progress. Uses the Cache API when available
 * (same pattern kokoro-js/Transformers.js use for ~80 MB, scaled up here
 * with explicit quota handling); falls back to a plain fetch when the
 * Cache API is absent (e.g. Node tests). A server that ignores Range
 * returns 200 with the full file — handled, not treated as an error.
 * Returns the bytes; persistence across reloads is the caller's cache key.
 */
export async function downloadFile(
  url: string,
  onProgress?: ProgressCallback,
  cacheName = 'ming-tts-web',
): Promise<Uint8Array> {
  const report = (loaded: number, total?: number) =>
    onProgress?.({
      status: 'downloading',
      file: url,
      loaded,
      total,
      progress: total ? Math.round((loaded / total) * 100) : undefined,
    });
  try {
    if (typeof caches !== 'undefined') {
      const cache = await caches.open(cacheName);
      const cached = await cache.match(url);
      if (cached) {
        onProgress?.({ status: 'cached', file: url, progress: 100 });
        const buf = await cached.arrayBuffer();
        return new Uint8Array(buf);
      }
    }
  } catch (e) {
    if (isQuotaError(e)) throw new QuotaExceededError();
    // Cache API failure is non-fatal: fall through to a plain fetch.
  }
  let response: Response;
  try {
    response = await fetch(url);
  } catch (e) {
    throw new DownloadInterruptedError(
      `MingTTS: failed to fetch ${url}: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
  if (!response.ok || !response.body) {
    throw new DownloadInterruptedError(
      `MingTTS: failed to fetch ${url}: HTTP ${response.status}`,
    );
  }
  const total = Number(response.headers.get('content-length') ?? 0) || undefined;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        loaded += value.length;
        report(loaded, total);
      }
    }
  } catch {
    throw new DownloadInterruptedError();
  }
  const bytes = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    bytes.set(c, off);
    off += c.length;
  }
  try {
    if (typeof caches !== 'undefined') {
      const cache = await caches.open(cacheName);
      await cache.put(
        url,
        new Response(bytes.slice().buffer, {
          headers: { 'content-type': 'application/octet-stream' },
        }),
      );
    }
  } catch (e) {
    if (isQuotaError(e)) throw new QuotaExceededError();
    // Cache-write failure after a successful download: still return bytes.
  }
  onProgress?.({ status: 'done', file: url, progress: 100, loaded, total });
  return bytes;
}

/**
 * Shard plan for a large graph file: N byte ranges downloaded in parallel
 * (the server must support Range; otherwise downloadFile's 200 fallback
 * applies per shard and the caller must not assume resume). This helper
 * only computes the ranges — the parallel fetch + reassembly lives with
 * the caller so progress accounting stays in one place.
 */
export function shardRanges(
  totalBytes: number,
  shardCount: number,
): { start: number; end: number }[] {
  if (totalBytes <= 0 || shardCount <= 0) {
    throw new Error('shardRanges: totalBytes and shardCount must be positive');
  }
  const size = Math.ceil(totalBytes / shardCount);
  const ranges: { start: number; end: number }[] = [];
  for (let i = 0; i < shardCount; i++) {
    const start = i * size;
    if (start >= totalBytes) break;
    ranges.push({ start, end: Math.min(start + size - 1, totalBytes - 1) });
  }
  return ranges;
}
