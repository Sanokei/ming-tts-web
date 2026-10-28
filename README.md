# ming-tts-web — Ming-3.0-tts-0.5B in the browser

**Integration status: experimental host library, not a working built-in speech synthesizer.**
`MingTTS.from_pretrained()` now rejects immediately unless a `FragmentEngine`
is supplied. A successful package install, an ONNX session factory, or a model
URL must never be reported as "speech ready". Check `mingReadiness()` before
showing a speech control. Readiness of an injected engine does not certify its
audio quality.

For a working browser TTS option today, use the public
[Kokoro JavaScript package](https://github.com/hexgrad/kokoro/tree/main/kokoro.js):

```ts
import { KokoroTTS } from 'kokoro-js';
const tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-v1.0-ONNX', {
  dtype: 'q8', device: 'wasm',
});
const audio = await tts.generate('Flight display ready.', { voice: 'af_heart' });
// Kokoro returns 24 kHz audio. Keep audio.sampling_rate; do not label it Ming.
```

This is an explicit alternative, with a separate package and model download;
the Ming API does not silently switch providers. Verify real synthesis on the
target device before enabling it for a user. Upsampling is not learned audio
super-resolution.

> **Status (2026-10-25): Step 1 CLOSED, Step 2 GATE NOT RUN, no real
> synthesis verified yet.** This package contains the TypeScript glue
> (host loop, audio frontends, task contracts, kokoro-js-shaped API) with
> tests, plus a real campplus graph check. It does NOT yet synthesize
> speech from real weights: the lm/dit ONNX exports and the WebGPU-vs-WASM
> gate benchmark require a GPU machine (see `notes/STEP2-GATE-PROTOCOL.md`).
> Do not present this as a working Kokoro replacement until that gate
> passes and end-to-end validation (Step 4) is done.

**Scope: `inclusionAI/Ming-3.0-tts-0.5B` ONLY. Never tts-tiny.**
`from_pretrained` refuses any model id containing "tiny".

## Injected-engine API (not a built-in synthesis quick start)

```ts
import { MingTTS, mingReadiness, type FragmentEngine } from 'ming-tts-web';

// This project does not yet provide a real fragment engine. Supply and verify
// your own engine before using this API. Omitting it rejects immediately.
declare const verifiedEngine: FragmentEngine;
if (!mingReadiness({ engine: verifiedEngine }).ready) throw new Error('Engine unavailable');

const tts = await MingTTS.from_pretrained('inclusionAI/Ming-3.0-tts-0.5B', {
  device: 'webgpu',
  dtype: 'fp16',
  engine: verifiedEngine,
  progress_callback: (p) => console.log(p.status, p.progress),
});
const out = await tts.generate('Hello, this is a test.', { voice: 'US_Emma', speed: 1 });
// -> { audio: Float32Array, sampling_rate: 44100 }
```

Full path (new capabilities):

```ts
const out2 = await tts.generate(text, {
  task: 'TTS',
  language: 'English',
  promptWav: wavFloat32Array, // reference clip
  promptText: '…transcript…', // REQUIRED with promptWav
  useSpkEmb: true,
  voiceDesign: '…description…', // VOICE_DESIGN only, no promptWav
  cfg: 2.0,
  seed: 0,
});
tts.list_voices(); // ['US_Emma','US_James','US_Stella','UK_George','UK_Charlotte']
for await (const part of tts.stream(text)) { /* per-fragment audio */ }
```

## Hard rules (enforced in code, not just docs)

- **44.1 kHz output, never 24 kHz.** `sampling_rate` is always 44100.
  Audit: never mix 24 kHz Kokoro tracks with 44.1 kHz Ming tracks in one
  `mixPcmTracks` call (kinoinstrument previs export takes a single rate).
- **`af_heart` does NOT exist in Ming.** It remaps explicitly to `US_Emma`
  (`voices.ts` VOICE_REMAP). The Anascummis worker and kinoinstrument
  previs export both default to `af_heart` today — their Ming migration
  must use this remap, not pass the name through.
- **Built-in voices are IP-task captions, not speaker slots.** There is no
  embedding file for US_Emma etc. in the checkpoint; `voice: 'US_Emma'`
  runs task IP with `ip: 'US_Emma'` in the structured caption.
- **Zero-speaker is NEVER the default.** No-voice calls use US_Emma (IP).
  Zero-speaker English is weak in the original checkpoint and varies with
  the noise draw (0.39 cosine between runs, per the MNN README).
- **No implicit downloads.** `from_pretrained()` requires an injected engine
  and never imports ONNX Runtime or downloads weights. The separate
  `loadOrtWeb()` utility remains lazy for engine implementers.
- **Text normalization is a JS subset, never silently skipped.** Numbers,
  dates, units, money, percent, fractions, ordinals are expanded;
  paralinguistic tags pass verbatim. This is NOT pynini FST parity — see
  `src/normalize.ts` header and `notes/STEP1-PINS-AND-CONTRACTS.md`.
- **Scalar speed is not native.** With task BASIC it maps to the trained
  Chinese speed/pitch/volume caption (BASIC requires all three); otherwise
  a non-1 speed is a post-hoc resample (also shifts pitch — prefer BASIC).
  Speed is never silently dropped.
- **Streaming is fragment-level, not token-level.** The vocoder runs on
  completed latents; `stream()` yields per-fragment audio.
- **No fake audio.** `from_pretrained()` without an engine throws
  `ModelNotLoadedError` immediately. Tests use an injected stub engine and are
  labeled as API/validation tests, not audio-quality tests.

## Voice cloning — consent notice

> Voice cloning is provided for research and legitimate creative use only.
> Do not use it to impersonate a real person without consent, to deceive,
> or for any illegal purpose. Label synthetic speech as synthetic where it
> could be mistaken for a real recording.

Cloning needs a reference clip (MNN documents 5–20 s) + its exact
transcript. The clip is resampled to 16 kHz for CAM++ (80-bin kaldi fbank,
mean-normalised → 192-d embedding) and to 44.1 kHz, front-padded to a
multiple of 3528 samples, for the VAE encoder. A second reference is
rejected by the reference design (one reference only).

## Task modes are contracts, not flags

`TTS`, `FREEFORM`, `VOICE_DESIGN`, `MULTILINGUAL`, `BASIC`, `EMOTION`,
`DIALECT`, `IP` (plus aliases SEED/PLAIN/STYLE). Rules enforced by
`buildTaskGenerationKwargs`: FREEFORM needs an instruction and forbids
controls (and cannot combine with voice design, built-in voice, emotion,
dialect, or basic attributes); MULTILINGUAL requires `language_id`
(ja/ko/de/fr/es); DIALECT takes no `language_id` (ignored with a warning);
BASIC requires all three of speed/pitch/volume (Chinese caption values);
EMOTION takes a reference but no reference transcript; structured tasks
build their instruction from controls and forbid a raw instruction.

The 14 paralinguistic tags (verbatim, case-sensitive): `[Breathing]`,
`[Cough]`, `[Sigh]`, `[Laughter]`, `[Surprise-ah]`, `[Surprise-oh]`,
`[Dissatisfaction-hnn]`, `[Uhm]`, `[Confirmation-en]`, `[Question-ei]`,
`[Question-ah]`, `[Question-en]`, `[Question-oh]`, `[Question-huh]`.

## Layout

```
ming-tts-web/
  RESEARCH.md                 # feasibility research (2026-10-25, research only)
  notes/
    STEP1-PINS-AND-CONTRACTS.md  # Step 1: pins, dims, graph contracts, loop
    STEP2-GATE-PROTOCOL.md       # Step 2 gate protocol for the GPU-machine run
  src/
    ming-tts.ts               # MingTTS class (kokoro-js shape)
    host-loop.ts              # AR loop + CFM loop + stop head + trim (no ORT)
    sessions.ts               # lazy ORT-web sessions, per-graph, device/dtype
    cache.ts                  # sharded download, Cache API, quota errors
    mathops.ts                # RoPE, masks, CFM schedule, ISTFT (MNN port)
    audio.ts                  # WAV I/O, torchaudio resample, kaldi fbank
    segment.ts                # semantic sentence splitting (max 50, min tail 5)
    normalize.ts              # JS normalization subset (tag-aware)
    instructions.ts           # task captions (verbatim upstream strings)
    prompt.ts                 # token-level input contract + tokenizer validation
    voices.ts                 # built-ins, af_heart remap, consent notice
  tests/
    mathops.test.ts  audio-segment-normalize.test.ts  api.test.ts
    cache-sessions.test.ts  bench.ts  (gate runner + campplus smoke)
  models/                     # NOT committed: downloaded ONNX + assets + tokenizer
```

## Tests

```sh
bun test        # 66 tests, all passing (2026-10-25, this VM)
bun tests/bench.ts  # gate runner; without MING_BENCH_CAMPPLUS_ONNX it reports NOT RUN
```

What the tests do and do not prove: they prove the host math, frontends,
contracts, validation, and API shape against the MNN Python reference
values (RoPE, masks, EPSS/sway timesteps, CFM CFG blend, trim, fbank
shapes, tokenizer IDs). They do NOT prove synthesis quality — the
mocked-engine API tests use a sine stub, labeled as such in the test
names. Real synthesis verification (Step 4: log-mel L1, CAM++ cosine ≥
0.93 for int8, Whisper transcript, EN + ZH) is still pending, gated on
Step 2.

## Real-graph check done on this VM (campplus only)

Shipped `campplus.onnx` (27 MB, from the original checkpoint repo):
input `(batch, seq, 80)` float32 → output `(batch, 192)` float32
(correcting the MNN docstring's stale-shape note). Bundled MNN sample
prompt: 9.312 s @ 24 kHz → (1, 929, 80) fbank → embedding L2 12.9053,
deterministic on rerun (max diff 0.0), 0.264 s single ORT CPU run on this
2-vCPU VM (onnxruntime 1.31). The bench smoke (1 s sine, JS fbank, CPU)
gave median 153.69 ms / p90 368.62 ms for campplus — a different input
shape (98 frames), so the two numbers are not directly comparable.
Neither number is the Step 2 gate (lm + dit, WebGPU vs WASM).

## Download / size reality

int8 default ≈ 1.80 GB total (1.52 GB graphs + 261 MB assets + 15 MB
tokenizer); fp16 ≈ 2.91 GB; int4 ≈ 1.05 GB (long-fragment degradation
documented in RESEARCH §4). Requires sharded download, resume, Cache /
IndexedDB persistence, explicit opt-in with progress, and quota-error
handling — all scaffolded in `src/cache.ts`, none of it load-tested at
1.8 GB on this VM. Mobile browsers are effectively excluded at this size.

## Downstream migration notes

- Anascummis `voice.worker.ts`: forwards `rate` already (good); swap
  `af_heart` → `US_Emma` via the remap, keep the worker isolation, and
  make every consumer of `rate` handle 44100.
- Kinoinstrument previs `tts.ts`: forwards `sampleRate` already (good);
  keep the lazy dynamic import; never mix sample rates in `mixPcmTracks`.
