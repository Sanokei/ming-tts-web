// Built-in voices + the Kokoro default remap + cloning consent.
// Read this before changing any default: the zero-speaker path is documented
// as weak/variable and must NEVER be the product default.

/**
 * English built-in (IP) voices, from the upstream cookbook
 * (cookbooks/instruction_en.md). NOTE: upstream tts_tasks.py's EN_IP_VOICES
 * constant lists only 4 and omits US_Stella — the cookbook (5) wins here;
 * the omission is flagged, not copied.
 */
export const ENGLISH_VOICES = [
  'US_Emma',
  'US_James',
  'US_Stella',
  'UK_George',
  'UK_Charlotte',
] as const;
export type EnglishVoice = (typeof ENGLISH_VOICES)[number];

/** Pinned product default. Never zero-speaker. */
export const DEFAULT_VOICE: EnglishVoice = 'US_Emma';

/**
 * 'af_heart' does NOT exist in Ming. The Anascummis worker and the
 * kinoinstrument previs export both default to af_heart today; any Ming
 * integration must remap it explicitly (this table) instead of passing it
 * through and getting a silent unknown-voice failure.
 */
export const VOICE_REMAP: Record<string, EnglishVoice> = {
  af_heart: 'US_Emma',
};

export function resolveVoice(voice?: string | null): EnglishVoice {
  if (!voice || !voice.trim()) return DEFAULT_VOICE;
  const v = voice.trim();
  if (v in VOICE_REMAP) return VOICE_REMAP[v];
  if ((ENGLISH_VOICES as readonly string[]).includes(v)) return v as EnglishVoice;
  // Non-English built-ins are IP names too (100+ Chinese IDs in the cookbook);
  // they pass through as IP captions. Unknown names are rejected by the
  // caller when strict mode is on; here we pass through so IP voices are
  // not artificially limited to the English five.
  return v as EnglishVoice;
}

export function listVoices(): string[] {
  return [...ENGLISH_VOICES];
}

/**
 * Cloning consent notice. Carried by the cloning API docs and any
 * user-facing cloning UI (which additionally needs a consent checkbox —
 * that is the product's decision, not the glue's). Wording follows the
 * official model card's Usage Disclaimer.
 */
export const CLONING_CONSENT_NOTICE =
  'Voice cloning is provided for research and legitimate creative use only. ' +
  'Do not use it to impersonate a real person without consent, to deceive, ' +
  'or for any illegal purpose. Label synthetic speech as synthetic where it ' +
  'could be mistaken for a real recording.';
