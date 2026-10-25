// JS text-normalization SUBSET. This is NOT pynini FST parity (upstream
// uses TalkerTN FST grammars + frontend/number_en.py with inflect).
// It covers the task-brief minimum — numbers, dates, units — for English,
// routes around paralinguistic tags verbatim, and NEVER silently skips:
// normalizeText always runs in the generate path unless the caller passes
// rawText: true explicitly. Degradation vs the FST path is documented in
// README.md; unhandled constructs pass through unchanged rather than
// being dropped.

import { PARALINGUISTIC_TAGS } from './instructions.js';

const ONES = [
  'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
  'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
  'sixteen', 'seventeen', 'eighteen', 'nineteen',
];
const TENS = [
  '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
];

export function integerToWords(n: number): string {
  if (!Number.isFinite(n) || n < 0 || Math.floor(n) !== n) {
    throw new Error(`integerToWords: expected a non-negative integer, got ${n}`);
  }
  if (n < 20) return ONES[n];
  if (n < 100) {
    const t = TENS[Math.floor(n / 10)];
    const r = n % 10;
    return r === 0 ? t : `${t} ${ONES[r]}`;
  }
  if (n < 1000) {
    const h = Math.floor(n / 100);
    const r = n % 100;
    return r === 0 ? `${ONES[h]} hundred` : `${ONES[h]} hundred ${integerToWords(r)}`;
  }
  if (n < 1_000_000) {
    const th = Math.floor(n / 1000);
    const r = n % 1000;
    return r === 0
      ? `${integerToWords(th)} thousand`
      : `${integerToWords(th)} thousand ${integerToWords(r)}`;
  }
  if (n < 1_000_000_000) {
    const m = Math.floor(n / 1_000_000);
    const r = n % 1_000_000;
    return r === 0
      ? `${integerToWords(m)} million`
      : `${integerToWords(m)} million ${integerToWords(r)}`;
  }
  // Beyond this, digit-by-digit is safer than a wrong large-number expansion.
  return String(n).split('').map((d) => ONES[Number(d)]).join(' ');
}

const DIGIT_WORDS = ONES.slice(0, 10);

export function decimalToWords(numStr: string): string {
  const negative = numStr.startsWith('-');
  const clean = numStr.replace(/^-/, '').replace(/,/g, '');
  const [intPart, fracPart] = clean.split('.');
  const intWords = integerToWords(intPart === '' ? 0 : parseInt(intPart, 10));
  let out = negative ? `minus ${intWords}` : intWords;
  if (fracPart !== undefined && fracPart.length > 0) {
    out += ` point ${fracPart.split('').map((d) => DIGIT_WORDS[Number(d)]).join(' ')}`;
  }
  return out;
}

const ORDINAL_SUFFIX: Record<string, string> = {
  st: '', nd: '', rd: '', th: '',
};
void ORDINAL_SUFFIX;

export function ordinalToWords(numStr: string, _suffix: string): string {
  const n = parseInt(numStr, 10);
  // General path: expand the cardinal, then replace the last word.
  const cardinal = integerToWords(n);
  const words = cardinal.split(' ');
  const last = words[words.length - 1];
  const irregular: Record<string, string> = {
    one: 'first', two: 'second', three: 'third', five: 'fifth',
    eight: 'eighth', nine: 'ninth', twelve: 'twelfth', twenty: 'twentieth',
    thirty: 'thirtieth', forty: 'fortieth', fifty: 'fiftieth',
    sixty: 'sixtieth', seventy: 'seventieth', eighty: 'eightieth',
    ninety: 'ninetieth',
  };
  if (irregular[last]) {
    words[words.length - 1] = irregular[last];
  } else if (last.endsWith('y')) {
    words[words.length - 1] = `${last.slice(0, -1)}ieth`;
  } else {
    words[words.length - 1] = `${last}th`;
  }
  return words.join(' ');
}

const UNIT_WORDS: Record<string, string> = {
  ms: 'milliseconds', s: 'seconds', hz: 'hertz', khz: 'kilohertz',
  mhz: 'megahertz', ghz: 'gigahertz', kb: 'kilobytes', mb: 'megabytes',
  gb: 'gigabytes', tb: 'terabytes', bps: 'bits per second',
  kbps: 'kilobits per second', mbps: 'megabits per second',
  gbps: 'gigabits per second', cm: 'centimeters', km: 'kilometers',
  kg: 'kilograms', v: 'volts', a: 'amperes', w: 'watts',
  '°c': 'degrees celsius', '°f': 'degrees fahrenheit',
};

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December',
];

function yearToWords(year: number): string {
  if (year >= 2000 && year < 2100) {
    const rest = year - 2000;
    if (rest === 0) return 'two thousand';
    if (rest < 10) return `two thousand ${ONES[rest]}`;
    return `twenty ${integerToWords(rest)}`;
  }
  if (year >= 1900 && year < 2000) {
    return `${integerToWords(Math.floor(year / 100))} ${integerToWords(year % 100)}`;
  }
  return integerToWords(year);
}

function expandDates(text: string): string {
  // YYYY-MM-DD
  text = text.replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (_m, y, mo, d) => {
    const month = MONTHS[Number(mo) - 1];
    if (!month) return _m;
    return `${month} ${ordinalToWords(String(Number(d)), '')}, ${yearToWords(Number(y))}`;
  });
  // MM/DD/YYYY
  text = text.replace(/\b(\d{1,2})\/(\d{1,2})\/(\d{4})\b/g, (_m, mo, d, y) => {
    const month = MONTHS[Number(mo) - 1];
    if (!month) return _m;
    return `${month} ${ordinalToWords(String(Number(d)), '')}, ${yearToWords(Number(y))}`;
  });
  // "January 5, 2024" / "January 5th, 2024"
  text = text.replace(
    /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(st|nd|rd|th)?,\s+(\d{4})\b/g,
    (_m, month, d, _suf, y) =>
      `${month} ${ordinalToWords(String(Number(d)), '')}, ${yearToWords(Number(y))}`,
  );
  return text;
}

function expandNumbersAndUnits(text: string): string {
  // Percentages: -25.5% / 50%
  text = text.replace(/(-?[0-9][0-9,]*(\.[0-9]+)?)%/g, (_m, num) =>
    `${decimalToWords(num)} percent`,
  );
  // Money: $12.99 / £150 (upstream: "minus one hundred fifty pounds" style)
  text = text.replace(/\$(-?[0-9][0-9,]*(?:\.[0-9]+)?)/g, (_m, num) => {
    const [intPart, fracPart] = String(num).replace(/,/g, '').split('.');
    const dollars = decimalToWords(intPart);
    if (fracPart !== undefined) {
      const cents = parseInt(fracPart.padEnd(2, '0').slice(0, 2), 10);
      if (cents > 0) return `${dollars} dollars and ${integerToWords(cents)} cents`;
    }
    return `${dollars} dollars`;
  });
  text = text.replace(/£(-?[0-9][0-9,]*(?:\.[0-9]+)?)/g, (_m, num) =>
    `${decimalToWords(num)} pounds`,
  );
  // Fractions: 3/4 (only when both sides are plain integers)
  text = text.replace(/\b([0-9]+)\/([0-9]+)\b/g, (_m, a, b) =>
    `${integerToWords(Number(a))} over ${integerToWords(Number(b))}`,
  );
  // Ordinals: 1st, 2nd, 3rd, 4th
  text = text.replace(/\b([0-9]+)(st|nd|rd|th)\b/g, (_m, num, suf) =>
    ordinalToWords(num, suf),
  );
  // Units with a number: 10ms, 5.5 kg
  text = text.replace(
    /\b(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*(ms|kHz|MHz|GHz|kbps|Mbps|Gbps|bps|kb|mb|gb|tb|cm|km|kg|Hz|°C|°F|[sVvAaWw])\b/gi,
    (_m, num, unit) => {
      const word = UNIT_WORDS[String(unit).toLowerCase()];
      if (!word) return _m;
      return `${decimalToWords(num)} ${word}`;
    },
  );
  // Remaining plain numbers (last, so units/money/dates already expanded).
  // Thousands separators are stripped first: 1,000 -> 1000.
  text = text.replace(/([0-9][0-9,]+[0-9])/g, (m) => m.replace(/,/g, ''));
  text = text.replace(/\b-?[0-9]+(?:\.[0-9]+)?\b/g, (m) => decimalToWords(m));
  return text;
}

function normalizeSegment(text: string): string {
  let out = text;
  out = out.replace(/``(.+?)``/g, '$1').replace(/`(.+?)`/g, '$1').replace(/`/g, '');
  out = expandDates(out);
  out = expandNumbersAndUnits(out);
  out = out.replace(/\s+/g, ' ').trim();
  return out;
}

const TAG_RE = new RegExp(
  PARALINGUISTIC_TAGS.map((t) => t.replace(/[[\]]/g, (c) => `\\${c}`)).join('|'),
  'g',
);

/** Apply fn to the text around paralinguistic tags only; tags pass verbatim. */
export function protectParalinguisticTags(
  text: string,
  fn: (segment: string) => string,
): string {
  if (!text) return text;
  const tags = text.match(TAG_RE);
  if (!tags) return fn(text);
  const segments = text.split(TAG_RE);
  const pieces: string[] = [];
  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    if (segment.trim()) {
      const lead = segment[0] && /\s/.test(segment[0]) ? ' ' : '';
      const trail = segment[segment.length - 1] && /\s/.test(segment[segment.length - 1]) ? ' ' : '';
      pieces.push(lead + fn(segment).trim() + trail);
    } else {
      pieces.push(segment);
    }
    if (i < tags.length) pieces.push(tags[i]);
  }
  return pieces.join('');
}

/** Full JS-subset normalization for one fragment. Tags survive verbatim. */
export function normalizeText(text: string): string {
  return protectParalinguisticTags(text, normalizeSegment);
}
