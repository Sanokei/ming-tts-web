// Token-level input contract, ported from upstream prompt_contract.py.
// The released checkpoint learned ONE exact token layout; rebuilding it by
// hand in every caller is how silent quality regressions happen.

export const SYSTEM_PREFIX = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n';
export const USER_PREFIX = '<|im_start|>user\n';
export const USER_SUFFIX = '<|im_end|>\n';
export const ASSISTANT_PREFIX = '<|im_start|>assistant\n';
export const SPEAKER_START = '<|vision_start|>';
export const SPEAKER_PAD = '<|vision_pad|>';
export const SPEAKER_END = '<|vision_end|>\n';
export const INSTRUCTION_EOS = '<|im_end|>';
export const TEXT_INPUT_PREFIX = ' Text input:\n';
export const AUDIO_START = '<audio>';
export const AUDIO_PATCH = '<audioPatch>';

/** Trained token IDs of the dense 0.5B checkpoint — names alone are not enough. */
export const TOKEN_IDS: Record<string, number> = {
  '<|endoftext|>': 151643,
  '<|im_start|>': 151644,
  '<|im_end|>': 151645,
  '<|vision_start|>': 151652,
  '<|vision_end|>': 151653,
  '<|vision_pad|>': 151654,
  '<audioPatch>': 151705,
  '<audio>': 151706,
};

const BGM_MARKERS = ['Genre: ', 'Mood: ', 'Instrument: ', 'Theme: ', 'Duration: '];

export interface TokenizerLike {
  encode: (text: string) => number[];
  tokenId: (token: string) => number | null;
}

/**
 * Validate a tokenizer against the trained layout. Fails loudly when the
 * special tokens are missing, split, or re-indexed — a tokenizer that
 * silently splits <audio> still produces audio, just conditioned on a
 * sequence the model never saw.
 */
export function validateTokenizer(tok: TokenizerLike): void {
  const broken: Record<string, unknown> = {};
  for (const [token, want] of Object.entries(TOKEN_IDS)) {
    const got = tok.tokenId(token);
    const encoded = tok.encode(token);
    if (got === null || got !== want || encoded.length !== 1 || encoded[0] !== want) {
      broken[token] = { want, got, encoded };
    }
  }
  if (Object.keys(broken).length > 0) {
    throw new Error(
      `Tokenizer does not match the Ming-3.0-tts dense contract (expected, got, encoded): ${JSON.stringify(broken)}`,
    );
  }
}

export interface BuiltInputIds {
  ids: number[];
  speakerPositions: number[];
  latentPosition: number;
}

/**
 * Assemble the input sequence and the rows to overwrite with features.
 * Returns the ids, the index of each speaker_pad row, and the index of the
 * first prompt-latent row (-1 when there is no acoustic prompt).
 * Prompt latents are used ONLY when promptText is present; otherwise the
 * acoustic prompt is dropped entirely (upstream + MNN agree).
 */
export function buildInputIds(
  tok: TokenizerLike,
  prompt: string,
  text: string,
  opts: {
    numSpeakers?: number;
    instruction?: string | null;
    promptText?: string | null;
    promptLatentSteps?: number;
  } = {},
): BuiltInputIds {
  const numSpeakers = opts.numSpeakers ?? 0;
  const enc = (value: string) => tok.encode(value);
  const speakerBlock: number[] = [];
  for (let index = 0; index < numSpeakers; index++) {
    speakerBlock.push(
      ...enc(`  speaker_${index + 1}:`),
      ...enc(SPEAKER_START),
      ...enc(SPEAKER_PAD),
      ...enc(SPEAKER_END),
    );
  }
  const instructionBlock =
    opts.instruction !== null && opts.instruction !== undefined
      ? [...enc(opts.instruction), ...enc(INSTRUCTION_EOS)]
      : [];
  let promptTextBlock: number[] = [];
  let latentBlock: number[] = [];
  if ((opts.promptLatentSteps ?? 0) > 0 && opts.promptText !== null && opts.promptText !== undefined) {
    promptTextBlock = enc(opts.promptText);
    latentBlock = new Array(opts.promptLatentSteps).fill(TOKEN_IDS[AUDIO_PATCH]);
  }
  // A BGM caption carries its own field list and must not get the text prefix.
  const textPrefix = BGM_MARKERS.every((m) => text.includes(m)) ? [] : enc(TEXT_INPUT_PREFIX);
  const ids = [
    ...enc(SYSTEM_PREFIX),
    ...enc(USER_PREFIX),
    ...enc(prompt),
    ...speakerBlock,
    ...textPrefix,
    ...promptTextBlock,
    ...enc(text),
    ...enc(USER_SUFFIX),
    ...enc(ASSISTANT_PREFIX),
    ...instructionBlock,
    ...enc(AUDIO_START),
  ];
  const speakerPadId = tok.tokenId(SPEAKER_PAD);
  const speakerStartId = tok.tokenId(SPEAKER_START);
  const positions = ids
    .map((t, i) => (t === speakerStartId ? i + 1 : -1))
    .filter((i) => i >= 0);
  if (positions.length !== numSpeakers) {
    throw new Error(`expected ${numSpeakers} speaker slots, found ${positions.length}`);
  }
  for (const p of positions) {
    if (ids[p] !== speakerPadId) {
      throw new Error('speaker slot is not followed by its pad token');
    }
  }
  const latentPosition = latentBlock.length > 0 ? ids.length : -1;
  ids.push(...latentBlock);
  return { ids, speakerPositions: positions, latentPosition };
}
