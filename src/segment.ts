// Sentence/fragment splitting, ported from ming-tts-mnn
// _frontend/text_segment_cut.py (itself a port of the upstream frontend).
// Semantic length: one Han character = 1, one English word = 1.

export function isChinese(text: string): boolean {
  return /[\u4e00-\u9fff]/.test(text);
}

export function getSemanticLength(text: string): number {
  const han = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
  const words = (text.match(/[a-zA-Z]+/g) || []).length;
  return han + words;
}

export function hasValidContent(text: string): boolean {
  // Any character that is not ASCII punctuation or whitespace.
  for (const ch of text) {
    if (!/[!-\/:-@\[-`{-~\s]/.test(ch)) return true;
    // Chinese punctuation / Han chars fall through as valid (not ASCII punct).
    if (/[\u4e00-\u9fff]/.test(ch)) return true;
  }
  return false;
}

const WEAK_TAIL_PUNCT = '，,、;；';

/** Port of toolkit.add_period. */
export function addPeriod(text: string): string {
  const stripped = text.replace(/\s+$/, '');
  if (!stripped) return '.';
  const last = stripped[stripped.length - 1];
  if (WEAK_TAIL_PUNCT.includes(last)) {
    const hasChinese = /[一-鿿]/.test(stripped);
    return stripped.slice(0, -1) + (hasChinese ? '。' : '.');
  }
  // Unicode-aware alnum check (Python str.isalnum() is Unicode-aware:
  // '好'.isalnum() is True, so a Han-final string gets a period appended).
  if (!/[\p{L}\p{N}]/u.test(last)) return stripped;
  const isChineseChar =
    ('\u4e00' <= last && last <= '\u9fff') || ('\u3400' <= last && last <= '\u4dbf');
  return stripped + (isChineseChar ? '。' : '.');
}

function appendFragment(
  fragments: string[],
  newText: string,
  maxLen: number,
  minTailLength: number,
): string[] {
  const cleaned = newText.replace(/^[，,:;\s]+/, '');
  if (!hasValidContent(cleaned)) return fragments;
  if (fragments.length === 0) {
    fragments.push(cleaned);
    return fragments;
  }
  const last = fragments[fragments.length - 1];
  const lastLen = getSemanticLength(last);
  const newLen = getSemanticLength(cleaned);
  if (lastLen + newLen <= maxLen) {
    if (
      (last.endsWith('。') || last.endsWith('！') || last.endsWith('？')) &&
      newLen < minTailLength
    ) {
      fragments.push(cleaned);
    } else {
      const sep =
        !last.endsWith(' ') && /^[a-zA-Z0-9]/.test(cleaned) ? ' ' : '';
      fragments[fragments.length - 1] = last + sep + cleaned;
    }
  } else {
    fragments.push(cleaned);
  }
  return fragments;
}

function splitByCharWord(text: string, maxLen: number): string[] {
  const fragments: string[] = [];
  let current = '';
  const tokens =
    text.match(/(\s+|[，、；。！？,;!?]|[\u4e00-\u9fa5]|[a-zA-Z]+|[^a-zA-Z\u4e00-\u9fa5]+)/g) || [];
  for (const tok of tokens) {
    const tokLen = getSemanticLength(tok);
    const curLen = getSemanticLength(current);
    if (curLen + tokLen <= maxLen) {
      current += tok;
    } else {
      if (current) fragments.push(current);
      if (tokLen > maxLen) {
        fragments.push(tok);
        current = '';
      } else {
        current = tok;
      }
    }
  }
  if (current) fragments.push(current);
  return fragments;
}

function splitLongFragment(text: string, maxLen: number): string[] {
  if (getSemanticLength(text) <= maxLen) return [text];
  const clausePattern = /([，、；。！？,;!?]|\s+[-•·])/g;
  const parts = text.split(clausePattern).filter((p) => p !== '');
  const clauses: string[] = [];
  for (const part of parts) {
    if (clauses.length > 0 && /^([，、；。！？,;!?]|\s+[-•·])$/.test(part)) {
      clauses[clauses.length - 1] += part;
    } else {
      clauses.push(part);
    }
  }
  const fragments: string[] = [];
  let current = '';
  for (const clause of clauses) {
    const clauseLen = getSemanticLength(clause);
    if (clauseLen > maxLen) {
      if (current) fragments.push(current);
      current = '';
      fragments.push(...splitByCharWord(clause, maxLen));
      continue;
    }
    if (getSemanticLength(current) + clauseLen <= maxLen) {
      current += clause;
    } else {
      if (current) fragments.push(current);
      current = clause;
    }
  }
  if (current) fragments.push(current);
  return fragments;
}

/**
 * Port of cut_text_by_semantic_length, fragments only.
 * Decimal points and acronym dots are protected so "3.14" and "U.S.A."
 * are never treated as sentence boundaries.
 */
export function cutTextBySemanticLength(
  text: string,
  maxSemanticLength = 50,
  minTailLength = 5,
): string[] {
  if (!hasValidContent(text)) return [];
  const DOT = '##DOT##';
  let processed = text.replace(/(\d)\.(\d)/g, `$1${DOT}$2`);
  for (let i = 0; i < 3; i++) {
    processed = processed.replace(/([A-Z])\.([A-Z])/g, `$1${DOT}$2`);
  }
  processed = processed.replace(/\n/g, ' ').replace(/。，/g, '。');
  if (getSemanticLength(processed) <= maxSemanticLength) {
    return [processed.replaceAll(DOT, '.')];
  }
  const normalized = processed
    .replace(/\./g, '。')
    .replace(/!/g, '！')
    .replace(/\?/g, '？')
    .replace(/,/g, '，');
  // Phase 1: sentences on 。！？
  const sentences: string[] = [];
  let cur = '';
  for (const ch of normalized) {
    cur += ch;
    if (ch === '。' || ch === '！' || ch === '？') {
      const s = cur.trim();
      if (s) sentences.push(s);
      cur = '';
    }
  }
  if (cur.trim()) {
    let last = cur.trim();
    if (!/[。！？]$/.test(last)) last += '。';
    sentences.push(last);
  }
  // Phase 2+3: clauses on ，;； then fragment assembly.
  let result: string[] = [];
  for (const sentence of sentences) {
    const clauses: string[] = [];
    let clause = '';
    for (const ch of sentence) {
      clause += ch;
      if (ch === '，' || ch === ';' || ch === '；') {
        const c = clause.trim();
        if (c && hasValidContent(c)) clauses.push(c);
        else if (c && clauses.length > 0) clauses[clauses.length - 1] += c;
        clause = '';
      }
    }
    if (clause.trim()) {
      const c = clause.trim();
      if (c && hasValidContent(c)) clauses.push(c);
      else if (c && clauses.length > 0) clauses[clauses.length - 1] += c;
    }
    let i = 0;
    while (i < clauses.length) {
      const cl = clauses[i];
      const clLen = getSemanticLength(cl);
      if (clLen < minTailLength && i + 1 < clauses.length) {
        const combined = cl + clauses[i + 1];
        if (getSemanticLength(combined) <= maxSemanticLength) {
          result = appendFragment(result, combined, maxSemanticLength, minTailLength);
          i += 2;
          continue;
        }
      }
      if (clLen > maxSemanticLength) {
        for (const sub of splitLongFragment(cl, maxSemanticLength)) {
          result = appendFragment(result, sub, maxSemanticLength, minTailLength);
        }
      } else {
        result = appendFragment(result, cl, maxSemanticLength, minTailLength);
      }
      i += 1;
    }
  }
  return result.map((f) => f.replaceAll(DOT, '.'));
}

/** The MNN pipeline's prepare_text: strip markdown code, add period, split. */
export function prepareText(text: string, maxLength = 50): string[] {
  const stripped = text.replace(/``(.+?)``/g, '$1').replace(/`(.+?)`/g, '$1').replace(/`/g, '');
  return cutTextBySemanticLength(addPeriod(stripped), maxLength).filter((f) =>
    f.trim(),
  );
}
