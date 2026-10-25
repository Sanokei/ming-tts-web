# Ming 3.0 TTS (0.5B) in the browser — feasibility research

> **Scope:** `inclusionAI/Ming-3.0-tts-0.5B` ONLY. `tts-tiny` (7.9B MoE, ~18 GB,
> H800 recommended) is explicitly out of scope.
> **Status:** Research only, 2026-10-25 (5 days after the 2026-10-20 release).
> No glue code was written for this report. Every claim below is tagged
> **Confirmed** (official card / conversion repo / PyPI page), **Lineage**
> (inherited from the Ming-omni-tts predecessor — plausible, must be re-verified
> against the Ming-3.0-tts repo before the build relies on it), or **Inferred /
> unverified** (reasoning from the evidence, or a negative search result).
> The build worker's first job is to close the "Verify" items in §9.

## TL;DR

- **Possible, not drop-in, not small, not real-time on average hardware.**
  The model is a 7-graph system (LM + flow-matching DiT + aggregator + VAE
  encoder/decoder + CAM++), not a single ONNX file like Kokoro. There is
  **no browser-ready ONNX version** as of 2026-10-25. The only complete
  non-PyTorch conversion is the MNN port, which is a Python/CPU artifact —
  there is no maintained browser `MNN.js` runtime for it.
- **Recommended path:** convert each of the 7 MNN graphs' PyTorch originals to
  ONNX separately, orchestrate them in JavaScript with **onnxruntime-web**
  (WebGPU first, WASM fallback), ship **fp16 for WebGPU** and **int8 as the
  download-size option**, and keep Kokoro as the fallback until the Ming
  pipeline passes the acceptance tests in §9.
- **Realistic download:** ~1.80 GB total for int8 (1.52 GB graphs + 261 MB
  shared assets + 15 MB tokenizer), ~1.05 GB for int4, ~2.91 GB for fp16.
  Compare Kokoro-82M q8 at ~80 MB. This is a ~20× jump; it is a
  "download once, cache forever" asset, not a casual web asset, and it
  effectively excludes mobile browsers.
- **Realistic speed:** the native MNN port runs at RTF 1.1–2.0 (slower than
  real time) on an Apple M5 Pro with 4 CPU threads. Browser WASM will be
  slower; WebGPU speedup for this particular autoregressive × flow-matching
  loop is **unproven** and is the single most important benchmark the build
  must run first.

---

## 1. Model architecture breakdown

### 1.1 Confirmed facts (official model card)

| Item | Value |
|---|---|
| Model ID | `inclusionAI/Ming-3.0-tts-0.5B` |
| License | MIT |
| Backbone | Dense, Qwen2.5-0.5B |
| Total parameters (HF metadata) | **1,479,454,804, BF16** |
| Total size (BF16) | **2.96 GB** — arithmetic check: 1.479B params × 2 bytes ≈ 2.96 GB ✓ |
| Output | Mono, **44.1 kHz** |
| Languages | Chinese, English, German, Spanish, French, Korean, Japanese |
| Dialects | Sichuanese, Cantonese |
| Built-in voices | 100+ Chinese, **5 English: US_Emma, US_James, US_Stella, UK_George, UK_Charlotte** |
| Paralinguistic control | 14 inline tags (breath, cough, sigh, laughter, hesitation, surprise, question/confirmation — in Chinese or English) |
| Explicit controls | Emotion, speed, pitch, volume (as separate task modes, not free flags — see §6.4) |
| Python API | `ming_tts.MingTTS` / `synthesize(task=..., language_id=..., text=..., prompt_wav_path=..., prompt_text=..., use_spk_emb=..., output_path=...)` |
| CLI | `infer.py --task TTS --language_id ... --text ... --prompt_wav_path ... --prompt_text ... --use_spk_emb` |
| Tested hardware (official) | NVIDIA H800-80GB and H20-96G, CUDA 12.4; `flash-attn` optional, PyTorch SDPA fallback |
| Quality signal (independent) | Artificial Analysis: 1,188 Provider Voice Elo; 1,069 Controlled Voice Elo — 1st among open-weights models on cloning at release |

Sources: https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B ,
https://aireiter.com/blog/ming-3-0-tts-review-local-deployment ,
https://kie.ai/blog/what-is-ming-3-0

### 1.2 Components (confirmed as a set; per-component sizes NOT published)

The card itself says only: "The checkpoint includes the backbone, DiT head,
audio tokenizer, speaker encoder and text tokenizer." The component list below
is confirmed by the union of the official card's acknowledgements
(Qwen2.5 backbone, CAM++ speaker embedding, Ming-omni-tts audio tokenizer)
and the MNN conversion, which had to split the checkpoint into graphs
(§2). **The official card does not publish per-component parameter counts
or file sizes — do not invent them.** The build worker must measure them
from the safetensors key prefixes / the MNN graph files.

| Component | Role | What is known |
|---|---|---|
| **Qwen2.5-0.5B backbone (`lm`)** | Autoregressive LM. Predicts the next latent-patch hidden state from text + speaker + reference-audio conditioning. | "0.5B" is the *backbone label*. The remaining ~0.98B params (1.48B − 0.5B, arithmetic only) are everything else in the package: DiT, aggregator, VAE, CAM++, embeddings, stop head, projections. |
| **DiT / CFM flow-matching head (`dit`)** | Diffusion transformer that refines each LM hidden state into a high-quality audio latent patch via conditional flow matching. | Dominates compute: 10 DiT evaluations per latent step vs 1 LM decode (MNN page). CFG scale default 2.0 (MNN CLI `--cfg`); `sigma` 0.25 and `flow_steps` 10 are the documented defaults for the Tiny sibling — verify for 0.5B from the 0.5B repo, do not copy blindly. |
| **Aggregator (`aggregator`)** | Patch-by-patch compression / projection between latent space and LM embedding space. | **Lineage:** in Ming-omni-tts, `Aggregator` compresses 4 latent frames into one LLM step and projects latents to/from LLM space (`fm/dit.py`). Verify the 0.5B patch size from `dims.json` / the MNN host code. |
| **Audio VAE encoder (`enc_backbone`, `enc_agg`)** | Neural audio tokenizer: encodes a reference waveform into latents (cloning path only; not needed for built-in / voice-design synthesis except when a prompt is supplied). | Split into two graphs in MNN. The "audio tokenizer" is inherited from Ming-omni-tts per the card's acknowledgements. |
| **Audio VAE decoder / vocoder (`voc`)** | Decodes generated latents to a 44.1 kHz waveform. | Single-shot graph (run once per utterance after all latents are generated, or per completed fragment — verify streaming behaviour; the official server path returns a completed waveform, not progressive audio). |
| **CAM++ speaker encoder (`campplus`)** | Extracts a **192-dimensional** speaker embedding from the reference clip, projected through `spk_head` (Linear 192 → LLM hidden dim) into the LM's input. | **Lineage + confirmed dim:** Ming-omni-tts used `campplus.onnx` → 192-dim → `spk_head`. CAM++ in the 3D-Speaker lineage takes 80-dim Fbank features (16 kHz). The exact input sample rate / Fbank parameters for Ming-3.0-tts are a **Verify** item (§9) — the card does not state them. |
| **Stop head** | Learned binary classifier; generation stops when its output crosses 0.5. | **Confirmed behaviour, sensitive:** the MNN page warns this decision is discrete and rounding-sensitive — even the original's own fp32 vs bf16 runs differ in length (45 vs 50 decode steps on the same input). Any quantized browser build will produce slightly different utterance lengths; that is expected, not a bug. |
| **Text tokenizer + task instruction builder** | Qwen tokenizer plus per-task instruction strings (`create_instruction` in the predecessor). | Text is raw text, **not phonemes** (unlike Kokoro). Numbers/dates/units/symbols are normalized with FST grammars via `pynini` in the official Python path (or `--no_text_normalization` to skip). **Pynini has no browser equivalent** — the glue must ship a JS normalization subset or accept degraded numbers/dates. |

### 1.3 Inference pipeline (as the glue must implement it)

```
text (+ task, language_id)
  → task instruction string + Qwen tokenization        [JS: tokenizer.json]
  → [optional] reference wav
        → CAM++ → 192-dim speaker embedding → spk_head projection
        → AudioVAE encoder → reference latents → aggregator projection
  → concatenated conditioning sequence (speaker proj + ref latents + text)
  → AUTOREGRESSIVE LOOP (host-side, in JS):
        LM step → hidden state → aggregator
        → FLOW-MATCHING LOOP: 10 × DiT evaluations (noise → latent patch, CFG 2.0)
        → stop head > 0.5 ? stop : append latent, repeat
  → AudioVAE decoder (voc) → Float32 PCM, 44.1 kHz mono
```

Key structural fact for the build: **the loops are host-side, not inside any
single graph.** A single `torch.onnx.export` of the whole model is not
possible. The glue is a runtime, not a model file. This is the same shape
as the MNN port (§2), which is why that port is the best spec available.

---

## 2. MNN conversion prior art — the Rosetta Stone for this project

### 2.1 What exists

| Artifact | Location |
|---|---|
| Converted weights | `yunfengwang/Ming-3.0-tts-0.5B-MNN` — https://huggingface.co/yunfengwang/Ming-3.0-tts-0.5B-MNN |
| Host code (the pipeline logic) | PyPI package `ming-tts-mnn` v0.1.3 — https://pypi.org/project/ming-tts-mnn/ (code only, ~419 KB; weights downloaded from the HF repo above, ModelScope mirror `yunfeng/Ming-3.0-tts-0.5B-MNN`) |
| Usage | `uvx --from ming-tts-mnn ming-tts "…" -o hello.wav` or `MingTTSMNN.from_pretrained("int8")` → `synthesize(text)` returns float32, 44.1 kHz |

### 2.2 The 7 graphs (per precision tier)

Confirmed from the MNN repo README (identical set in `fp16/`, `int8/`, `int4/`):

| # | Graph | Component |
|---|---|---|
| 1 | `lm` | Qwen2.5-0.5B backbone |
| 2 | `dit` | CFM/DiT flow-matching head |
| 3 | `aggregator` | Latent ↔ LM-space projection / patch compression |
| 4 | `enc_backbone` | Audio-VAE encoder, backbone part |
| 5 | `enc_agg` | Audio-VAE encoder, aggregator part |
| 6 | `voc` | Audio-VAE decoder / vocoder |
| 7 | `campplus` | CAM++ speaker encoder |

Plus shared, tier-independent files:

| File | Size | Notes |
|---|---|---|
| `assets/` (`dims.json`, `host.npz`, `embed.npy`) | **261 MB** | Host-side tensors (shapes, built-in/default embeddings — exact contents to be inspected from `dims.json`; this is a large hidden cost that must be included in every download budget) |
| `tokenizer/` (`tokenizer.json` + configs) | **15 MB** | Qwen tokenizer, reusable in the browser via Transformers.js `AutoTokenizer` |

**The task brief's "7 graphs (lm, dit, aggregator, enc, voc, campplus, +1)"
resolves as: the "+1" is the split of the VAE encoder into `enc_backbone`
+ `enc_agg`. There is no eighth mystery graph.**

### 2.3 How the MNN pipeline works (and what it proves for ONNX)

- Text → tokens (tokenizer) → LM autoregressive decode, one latent patch per
  step, stop head at 0.5, sentence splitting at ≤50 characters per fragment
  by default (`--max-length 50` in the MNN CLI; punctuation-aware).
- Per latent step: **10 DiT evaluations**; per-step cost is identical across
  fp16/int8/int4 — quantization only shrinks the one-off prefill / encoder /
  vocoder costs. Quantization is therefore primarily a **download/memory
  saving, not a speed saving**, for this architecture.
- Reference path: prompt wav + transcript → `campplus` (speaker embedding)
  + VAE encoder (reference latents); zero-speaker path uses a zero vector
  *after* projection.
- Fidelity (same noise, same inputs, vs the fp32 original): MNN fp16 is
  12–23× *closer* to fp32 than the shipped bf16 model is to itself
  (step-level relative peak error 9.0e-4 → 2.8e-2 over 8 flow steps, vs
  bf16's 2.1e-2 → 4.1e-1). In other words, **the decomposition itself is
  numerically sound** — fp16 MNN is inside the original's own rounding
  envelope. This de-risks the "split into graphs" approach for ONNX too.
- Speed (Apple M5 Pro, 4 MNN threads, native CPU): a 3–6 s sentence takes
  1–2× its duration (**RTF 1.1–2.0**). At a fixed 40 decode steps (3.20 s of
  audio): fp16 10.1 s, int8 8.9 s, int4 8.3 s; PyTorch bf16 21.8 s.
  **A browser build should be benchmarked against RTF ≤ 2.0 on desktop as
  the "acceptable" bar, and must not promise real-time on WASM.**

### 2.4 Can this decomposition guide an ONNX conversion? — Yes, directly

The MNN port proves the correct graph boundaries and the host-side loop
structure. The ONNX build should mirror it 1:1:

1. Export the **same 7 PyTorch submodules** to ONNX (not the fused model).
2. Port the **host logic** from `ming-tts-mnn` (Python) to TypeScript:
   sentence splitting, instruction construction, CAM++ Fbank frontend,
   autoregressive loop with KV cache, 10-step flow loop with CFG/noise,
   stop-head check, latent accumulation, vocoder call.
3. Validate numerically **graph-by-graph against the MNN graphs**
   (same inputs → compare outputs), then end-to-end against MNN fp16
   (log-mel L1 / CAM++ cosine, the MNN author's own metrics).
4. Ship the same shared assets (`dims.json` semantics, `host.npz`,
   `embed.npy`, `tokenizer.json`) — do not re-derive them by hand.

**Do not attempt to convert the MNN `.mnn` files themselves to ONNX.**
MNN's converter ingests ONNX; MNN→ONNX export is not a supported,
documented path found in this research. Convert from the PyTorch originals,
using MNN only as the reference implementation and test oracle.

---

## 3. ONNX availability and conversion difficulty

### 3.1 Does a browser-ready ONNX version exist? — No (as of 2026-10-25)

- Searched: `onnx-community` listings, HF search, and the web for
  "Ming 3.0 ONNX". **No `onnx-community/Ming-*` repo, no third-party ONNX
  export, and no Transformers.js model class for Ming was found.**
  (Negative result, 5 days post-release — re-check before building; a
  community export could appear, but the build must not assume one.)
- The only non-PyTorch runtimes found are:
  - **MNN** (`yunfengwang/...-MNN`, §2) — CPU, Python host, no browser build.
  - **GGUF / audio.cpp** (`audio-cpp/Ming-3.0-TTS-0.5B-GGUF`):
    `ming-3.0-tts-0.5b-bf16.gguf` and `-q8_0.gguf` (Q8_0, BF16 stop head
    and F32 speaker encoder retained). Contains Qwen backbone,
    flow-matching head, audio VAE, CAM++ encoder, config, tokenizer.
    Measured on RTX 5090: warm RTF 0.17–0.18 (3.8× the Python bf16
    baseline), peak VRAM 2.77 GB bf16 / 1.92 GB q8. **This proves GPU
    acceleration transforms the economics — but there is no browser/WASM
    port of audio.cpp's `ming3_tts` path, so it is a fallback research
    direction, not the plan.** Source: https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF

### 3.2 Conversion difficulty by component

| Component | Difficulty | Notes |
|---|---|---|
| `campplus` (CAM++) | **Easy** | Single-shot encoder. Precedent: Ming-omni-tts already shipped `campplus.onnx`. Main work is the JS Fbank frontend (resample → 80 mel bins → tensor), not the graph. |
| `enc_backbone` + `enc_agg` (VAE encoder) | **Easy–medium** | Single-shot, prompt-only. Only runs when a reference clip is supplied; can be lazy-loaded. Verify ops (convolutions / resampling) against ORT-web WebGPU support. |
| `voc` (VAE decoder) | **Medium** | Single-shot, but it is the largest single latency spike (full-utterance upsampling). Verify upsampling / conv-transpose ops on WebGPU; fall back to WASM for this graph alone if a WebGPU op is missing. |
| `aggregator` | **Medium** | Small projection, but its exact input/output contract is undocumented outside the MNN host code. Read it from `ming-tts-mnn` source + `dims.json`; do not guess shapes. |
| `lm` (Qwen2.5-0.5B) | **Medium–hard** | Standard decoder architecture (ORT/Transformers.js already run Qwen2), BUT the export is custom: KV-cache decode loop, plus non-standard conditioning inputs (projected speaker embedding + reference latents concatenated with text embeddings). A stock `Qwen2ForCausalLM` ONNX export will not have these inputs. The glue must manage `past_key_values` across steps in JS. |
| `dit` (DiT flow head) | **Hard (loop, not graph)** | The graph itself exports, but inference is a **host-side numerical loop**: 10 flow steps × every autoregressive step, each with timestep conditioning, CFG blending, noise injection (`sigma`), and latent-history feedback. The CFM integrator (Euler step schedule etc. in `fm/CFM.py` lineage) must be re-implemented exactly in JS — a small error here compounds across steps and changes prosody/length. This is the highest-risk porting surface, after the LM loop. |
| Tokenizer + text normalization | **Medium** | `tokenizer.json` is directly reusable (Transformers.js). The FST/`pynini` normalizer is not: ship a JS subset (numbers, dates, common units/symbols) and document the degradation, or run normalization through a small WASM build — do not silently skip it, because un-normalized input changes pronunciation. |
| Task instruction templates | **Medium** | `TTS` / `FREEFORM` / `VOICE_DESIGN` / `MULTILINGUAL` / `BASIC` / `EMOTION` / `DIALECT` / `IP` each build a different instruction string. These strings live in the Ming-3.0-tts repo's cookbook/code; the glue must port them verbatim. Freeform cannot be combined with voice design, built-in voice, emotion, dialect, or basic attributes (official constraint). |

### 3.3 What "conversion" actually means here (for the build plan)

It is **not** `torch.onnx.export(model)`. It is:

1. 7 separate exports (opset 17–18, dynamic axes for sequence length and
   latent length, fp16 weights where ORT-web WebGPU handles them).
2. A TypeScript orchestrator (~the `ming-tts-mnn` host logic, ported).
3. A JS Fbank frontend + JS text-normalization subset + JS WAV decode /
   resample (Web Audio `decodeAudioData` + offline resampling).
4. Per-graph ORT-web sessions with explicit input/output tensor contracts
   taken from `dims.json`, validated against MNN outputs.
5. Quantization as a **separate, per-graph** step (§4) — not a single
   `quantize_dynamic` call over a fused model that does not exist.

---

## 4. Quantization — sizes, quality, and the browser reality

### 4.1 Confirmed numbers (MNN tiers; the only quantized Ming-3.0-tts-0.5B data)

| Tier | Graphs | + shared assets (276 MB) | **Total download** | Quality vs fp16 |
|---|---|---|---|---|
| Original BF16 (HF) | — | — | **2.96 GB** | Reference |
| MNN fp16 | 2.63 GB | 261 MB assets + 15 MB tokenizer | **~2.91 GB** (PyPI: "~2.9 GB including shared assets") | Inside original's own bf16 rounding envelope |
| MNN int8 | 1.52 GB | + 276 MB | **~1.80 GB** | Recommended. Speaker cosine 0.932–0.996 (mean 0.979) over 16 extra texts; Whisper transcribed 16/16; long-form ASR on par with original |
| MNN int4 | 0.77 GB | + 276 MB | **~1.05 GB** | Audibly coarser timbre/articulation. Cosine 0.828–0.927 (mean 0.879). One forced single-fragment long paragraph: Whisper CER 80% (vs 27–31% for fp16/int8) and speaker cosine collapse to 0.468–0.505; can repeat/cut off on long out-of-domain text. Sentence-split short texts still transcribed 16/16 — int4's failure mode is long fragments, not all text. |

Sources: https://huggingface.co/yunfengwang/Ming-3.0-tts-0.5B-MNN ,
https://pypi.org/project/ming-tts-mnn/
Note: an earlier `int4mix` tier was removed in `ming-tts-mnn` 0.1.3 after
it proved the long-text regression comes from the LM's 4-bit error floor,
not the flow head — **do not revive mixed int4-LM ideas without re-proving
them; the evidence says the LM is the int4-sensitive part.**

### 4.2 What quantization does NOT do here

- **int8/int4 barely speed up this model.** Per §2.3, per-step cost is
  identical across tiers because the DiT loop dominates; int8/int4 save
  only ~12% / ~18% end-to-end at fixed step count. Buy quantization for
  download size and RAM, not for speed.
- **ONNX int8 ≠ MNN int8.** ORT dynamic quantization typically emits
  `DequantizeLinear` nodes that dequantize weights back to fp32 before
  each op — one measured ORT-web case found int8 *slower* than fp16 on
  WebGPU (359 ms vs 209 ms) while only saving download size, because no
  `ConvInteger`/`QLinearConv` kernels were used. **For WebGPU, fp16 is
  likely faster than int8; for WASM/CPU, int8 saves memory bandwidth.
  The build must benchmark fp16 vs int8 per backend and must not assume
  int8 is faster.**
- **GGUF q8_0 is a separate data point:** audio.cpp's q8_0 keeps the stop
  head in BF16 and the speaker encoder in F32 (the two most
  rounding-sensitive parts) — a useful precedent if the ONNX build does
  per-graph mixed precision: keep `campplus` and the stop head at higher
  precision, quantize `lm`/`dit`/`voc` first.

### 4.3 Realistic browser download size (recommendation)

| Ship | Total | When |
|---|---|---|
| **int8 (default)** | **~1.80 GB** | Default for desktop. Best size/quality trade per MNN author. |
| int4 | ~1.05 GB | Only if disk/RAM is the hard limit, and only with sentence splitting enforced + a warning that long text may degrade. |
| fp16 | ~2.91 GB | WebGPU performance path if int8 proves slower on WebGPU (§4.2), or as the quality reference. |

All three require: sharded files (parallel download + resume), Cache API /
IndexedDB persistence (kokoro-js already caches ~80 MB this way; 1.8 GB
needs explicit quota handling and a progress UI — browsers do not
guarantee multi-GB storage), `Cross-Origin-Opener-Policy` /
`Cross-Origin-Embedder-Policy` headers for multithreaded WASM
(SharedArrayBuffer), and a "this is a large one-time download" UX.
**Do not ship this as an on-first-utterance lazy load; it must be an
explicit opt-in download with progress, like the Anascummis voice-model
download UX.**

---

## 5. kokoro-js API shape — the interface the Ming replacement must mirror

### 5.1 What Anascummis actually uses today (verified in-repo)

File: `~/workspace/anascummis/apps/studio/components/voice.worker.ts`
(worker thread; the voice path is already worker-isolated — the Ming
worker must stay in a worker too).

```ts
import { KokoroTTS } from 'kokoro-js';

// Load (singleton, lazy):
synthesizer ||= await KokoroTTS.from_pretrained(
  'onnx-community/Kokoro-82M-v1.0-ONNX',
  { device: 'wasm', dtype: 'q8', progress_callback: progress }
);

// Warm-up:
await synthesizer.generate('Hello.', { voice: 'af_heart', speed: 1 });

// Real call:
const result = await synthesizer.generate(text, { voice: 'af_heart', speed: 1 });
// → posts back: { audio: result.audio, rate: result.sampling_rate }
//    transferred as [result.audio.buffer] (zero-copy)
```

The worker contract is therefore: **`{ audio: Float32Array, sampling_rate: number }`**,
with `audio` transferable. Any replacement must return exactly this shape
or the worker message contract must be updated in the same change.

### 5.2 Full kokoro-js public surface (npm docs)

```ts
import { KokoroTTS, TextSplitterStream } from 'kokoro-js';

const tts = await KokoroTTS.from_pretrained(model_id, {
  dtype: 'q8',        // 'fp32' | 'fp16' | 'q8' | 'q4' | 'q4f16'
  device: 'wasm',     // 'wasm' | 'webgpu' (web) | 'cpu' (node)
  progress_callback,  // (progress) => void
});

const audio = await tts.generate(text, { voice: 'af_heart' });
// audio: RawAudio { audio: Float32Array, sampling_rate: 24000 }
// audio.save('audio.wav');            // RawAudio helper
tts.list_voices();                    // list available voice IDs

// Streaming variant:
const splitter = new TextSplitterStream();
const stream = tts.stream(splitter);
for await (const { text, phonemes, audio } of stream) { /* … */ }
splitter.push(token); splitter.close();
```

Sources: `kokoro-js` npm docs (via socket.dev mirror),
https://www.javiertsalas.com/blog/kokoro-tts.html

### 5.3 Kokoro internals the Ming glue must NOT copy

Kokoro's ONNX graph is a **single** model: phoneme IDs (max 512 tokens,
padded with 0) + a 256-dim style vector sliced by token count + scalar
speed → waveform at **24 kHz**. Ming differs on every axis:

| | Kokoro | Ming 3.0 0.5B |
|---|---|---|
| Input text | Phonemes (espeak-style G2P) | Raw text + task instruction |
| Voice | Fixed style-vector ID (`af_heart`, …) | Built-in ID **or** reference wav + transcript **or** voice-design text |
| Graphs | 1 ONNX graph | 7 graphs + host loop |
| Output rate | 24,000 Hz | **44,100 Hz** |
| Speed | Scalar `speed` | `BASIC` task with speed + pitch + volume together; otherwise speed is not a standalone scalar |

### 5.4 Proposed mirror interface (spec for the build worker)

Keep the kokoro-js shape so the Anascummis worker diff is small, but
extend it — do not pretend Ming is a drop-in voice-ID swap:

```ts
import { MingTTS } from 'ming-tts-web'; // new package, name TBD

const tts = await MingTTS.from_pretrained('inclusionAI/Ming-3.0-tts-0.5B', {
  device: 'webgpu',      // 'webgpu' preferred, 'wasm' fallback
  dtype: 'fp16',         // WebGPU default; 'q8' for WASM/download-size path
  progress_callback,
});

// Compatibility path (what the Anascummis worker calls today):
const out = await tts.generate(text, { voice: 'US_Emma', speed: 1 });
// → { audio: Float32Array, sampling_rate: 44100 }
// NOTE: 'af_heart' does NOT exist in Ming. The worker must remap its
// default voice (af_heart → US_Emma or a bundled cloned voice) explicitly.
// `speed` maps to the BASIC task only when pitch/volume are also given;
// otherwise it is ignored or applied as a post-hoc resample — decide in
// the build and document it. Do NOT silently drop it.

// Full path (new capabilities — the reason to switch):
const out2 = await tts.generate(text, {
  task: 'TTS',                 // TTS | FREEFORM | VOICE_DESIGN | MULTILINGUAL | BASIC | EMOTION | DIALECT | IP
  language: 'English',         // language_id for MULTILINGUAL; DIALECT takes none
  promptWav: wavFloat32Array,  // reference clip (see §6)
  promptText: '…transcript…',  // REQUIRED with promptWav
  useSpkEmb: true,
  voiceDesign: '…description…',// VOICE_DESIGN only, no promptWav
  emotion: 'happy', speed: 1.0, pitch: 1.0, volume: 1.0, // task-gated
  cfg: 2.0, seed: 0,
});
tts.list_voices(); // → ['US_Emma','US_James','US_Stella','UK_George','UK_Charlotte', …Chinese IDs…]
```

**Downstream audit required (same change):** anything that assumes
24 kHz — audio playback rate, duration math, and especially the
kinoinstrument previs export timing — must be updated to read
`sampling_rate` (44,100) instead of hard-coding 24,000.
Verified 2026-10-25 in
`~/workspace/kinoinstrument/apps/web/src/views/previs/export/tts.ts`:
it uses the same model/voice/speed (`onnx-community/Kokoro-82M-v1.0-ONNX`,
`af_heart`, speed 1, `dtype: 'q8'`, `device: 'wasm'`), caps input at
**4000 chars per request**, and its `synthesizeLine()` already returns
`{ samples: out.audio, sampleRate: out.sampling_rate }` — so the rate is
forwarded, not hard-coded, which is good. Two remaining audit points:
(1) its `mixPcmTracks(tracks, sampleRate)` takes a single sample rate
for all tracks, so a caller must never mix 24 kHz Kokoro tracks with
44.1 kHz Ming tracks in one mix; (2) it lazy-loads kokoro-js via dynamic
`import()` because the module throws at import time in some environments
— the Ming loader must preserve that lazy-load pattern. The Anascummis
worker already forwards `rate`, which is good; check every other consumer
of that field.

**Streaming:** kokoro-js `stream()` yields per-sentence audio. Ming
cannot stream mid-utterance the same way (latents are generated
autoregressively, but the vocoder runs on completed latents). The honest
streaming equivalent is **per-fragment streaming**: split text into
sentences/fragments (≤50 chars in the MNN CLI default is very aggressive;
tune for English), synthesize fragment-by-fragment, yield each fragment's
audio as it completes. Do not promise token-level streaming.

---

## 6. Voice cloning input format + Voice Design

### 6.1 Zero-shot cloning — what the caller must supply

| Field | Value |
|---|---|
| Reference audio | **WAV file** (`--prompt_wav_path` / `--prompt-wav me.wav`). Browser equivalent: a decoded `Float32Array` (via `AudioContext.decodeAudioData`, which also accepts MP3/OGG — but the build should accept WAV first and document any transcoding). |
| Reference transcript | **Exact transcript of the reference clip, required** (`--prompt_text` / `prompt_text`). This is not optional metadata — the reference latents are conditioned on it. |
| `use_spk_emb` | `true` to use the CAM++ embedding path (in addition to the VAE latent path). |
| Clip length | **Conflicting sources — resolve by testing:** the model card / press describe "a short reference clip" / "3- to 5-second" clips; the MNN CLI documents **5–20 s** wav + transcript. The build must test 3 s, 5 s, 10 s, and 20 s clips and pick the documented range from measurements, not from marketing. Longer is likely better for CAM++ (more frames to pool) up to the encoder's context; do not hard-code "3–5 s" as a validation rule until tested. |
| Input sample rate | **Not stated on the model card.** CAM++ lineage uses 16 kHz / 80-dim Fbank; the VAE encoder may expect a different rate. **Verify from `ming-tts-mnn` source / `dims.json` before writing the resampler.** The glue must resample the decoded clip to whatever the graphs expect — never feed the browser's native 48 kHz decode output directly. |
| Count | **One reference only** — a second reference is rejected (per setup-guide reporting). |
| Output of the two paths | (a) 192-dim CAM++ embedding → `spk_head` projection; (b) VAE-encoded reference latents → aggregator projection. Both are concatenated with text embeddings before the LM loop. A cached speaker embedding alone (192 floats) is enough to *reuse* a voice without re-running CAM++ — but reusing a voice without its reference latents is a different conditioning mix; test whether embedding-only reuse sounds identical before offering "save voice" as a feature. (audio.cpp accepts a saved `speaker_embedding_file`, which suggests embedding reuse is a real path — verify against Ming-3.0, not just audio.cpp.) |

### 6.2 Voice Design (no reference audio)

- Task `VOICE_DESIGN`, **no `prompt_wav`**.
- Input: a natural-language description of the target speaker
  (timbre, apparent age, physical resonance, etc.). In the predecessor
  this becomes an instruction string (`create_instruction` →
  `BASE_CAPTION_TEMPLATE`) fed to the LLM as text tokens — i.e. Voice
  Design is *text conditioning*, not a separate voice-encoder model.
- Use it for: generating a new consistent persona without a recording,
  and for the Anascummis default voice if no built-in fits.
- Caveat: open-ended Voice Design does not yet match the strongest
  evaluated system (official limitations), and the same description with
  a different seed/noise draw will not give the identical voice — for a
  stable product voice, prefer a built-in ID or a bundled cloned
  embedding over a fresh Voice Design call per session.

### 6.3 Built-in voices and the default-voice trap

- English built-ins: `US_Emma`, `US_James`, `US_Stella`, `UK_George`,
  `UK_Charlotte`. Chinese: 100+ IDs (enumerate from the repo in the build).
- **Do not ship the zero-speaker default as the product voice.**
  `use_zero_spk_emb` (zero vector after projection) produces "the model's
  average voice", but the MNN page documents that it **varies with the
  noise draw** (two runs score only 0.39 cosine against each other).
  Worse, the MNN author reports the original checkpoint's English
  zero-speaker output as weak — "unintelligible syllables for the full
  26 s budget" on their test sentence — and the conversion faithfully
  reproduces that behaviour. **Product default must be a pinned built-in
  (e.g. `US_Emma`) or a bundled prompt (wav + transcript shipped with the
  app), never zero-speaker.**
- Consent/disclaimer: the official card requires legitimate creative use,
  no impersonation without consent, and labelling synthetic speech where
  it could be mistaken for a recording. The glue's cloning API should
  carry that notice in its docs and, for any user-facing cloning UI, a
  consent checkbox is the product's decision, not the glue's — flag it
  for the product owner.

### 6.4 Task modes are contracts, not flags

`TTS`, `FREEFORM`, `VOICE_DESIGN`, `MULTILINGUAL`, `BASIC`, `EMOTION`,
`DIALECT`, `IP`. Rules that constrain the API design:

- `FREEFORM` (natural-language delivery direction) cannot be combined
  with voice design, built-in voice, emotion, dialect, or basic attributes.
- `MULTILINGUAL` requires `language_id` (German/Spanish/French/Korean/
  Japanese); `DIALECT` takes no `language_id`.
- `BASIC` requires all three of speed, pitch, volume.
- `EMOTION` takes a reference but no reference transcript.
- Multiple paralinguistic tags in one sentence are supported but less
  stable than single tags (official limitation).
- The exact 14 tag strings must be copied from the repo's cookbook —
  they were not fully enumerated on the model card, and inventing tag
  spellings will silently fail (unknown tags are just text).

---

## 7. Runtime choice

### 7.1 Comparison

| Runtime | Browser-ready? | WebGPU? | WASM? | Verdict for Ming |
|---|---|---|---|---|
| **onnxruntime-web (ORT-web)** | **Yes** — npm `onnxruntime-web`, maintained by Microsoft, the runtime under kokoro-js and Transformers.js | Yes (WebGPU execution provider) | Yes (SIMD + threaded, needs COOP/COEP for threads) | **Recommended.** Only production browser runtime with per-graph sessions, dynamic shapes, and a WebGPU path. Cost: the glue must implement the loop, KV cache, and CFG in JS — ORT-web gives sessions, not a TTS pipeline. |
| **Transformers.js (`@huggingface/transformers`)** | Yes | Via ORT-web | Via ORT-web | **Not sufficient alone.** It has no Ming model class and no task for this 7-graph loop. Use it only for what it is good at: `AutoTokenizer` from `tokenizer.json` and the HF cache conventions. Do not try to shoehorn Ming into a `pipeline()`. |
| **MNN.js / MNN WASM** | **No maintained browser package found** (searched 2026-10-25) | No WebGPU backend in the JS story (MNN backends are CPU/OpenCL/Vulkan/Metal for native) | Would require a custom Emscripten build of `alibaba/MNN` + hand-written JS bindings for 7 graphs, KV handling, and the host loop | **Not recommended as the browser runtime**, despite the MNN graphs being the best conversion reference. Revisit only if the build proves an MNN WASM build beats ORT-web WebGPU on this loop — that proof does not exist today. MNN's value to this project is as **spec + test oracle**, not as the ship runtime. |
| **GGUF / audio.cpp / llama.cpp-style WASM** | No port found for `ming3_tts` | N/A | Would require porting audio.cpp's Ming path to WASM + a JS sampler | **Fallback research only.** Strongest native numbers (RTF 0.17 on RTX 5090, q8), but zero browser evidence. If ORT-web fails on speed, this is the second investigation, not a parallel build. |
| **Server-side (ming-tts-mnn / audio.cpp behind an API)** | N/A (not in-browser) | N/A | N/A | **Pragmatic fallback if the browser proof fails.** Keep Kokoro in-browser; run Ming server-side. This preserves the voice quality win without the 1.8 GB download. Decide this fallback *before* promising users an in-browser Ming, not after. |

### 7.2 Recommendation (with reasoning)

**onnxruntime-web, WebGPU-first, WASM fallback, fp16-first for WebGPU,
int8 as the download option.**

1. It is the only runtime where every piece (sessions, WebGPU, WASM,
   quantization tooling, HF-style caching) already exists and is used in
   production by the exact library being replaced (kokoro-js).
2. WebGPU is the only plausible route to acceptable speed: native CPU is
   already RTF 1.1–2.0 on a strong laptop, so WASM-only would be a
   regression users feel. But WebGPU speedup is **unproven for this loop**
   (many small session calls, KV-cache tensor passing, DiT loop) — the
   first benchmark must be WebGPU vs WASM on a single LM step + a single
   DiT step, before building the full loop.
3. Per-graph mixed precision (GGUF precedent): keep `campplus` and the
   stop head at fp32/fp16, quantize `lm`/`dit`/`voc` — and benchmark
   fp16 vs int8 on WebGPU separately, because ORT-web int8 can be slower
   on WebGPU (§4.2).
4. Run everything in a **Web Worker** (as Anascummis already does for
   Kokoro), with `InferenceSession` creation off the main thread, model
   files in Cache API / IndexedDB, and progress callbacks plumbed through
   `postMessage` exactly like the current worker.

---

## 8. Honest feasibility verdict

### Verdict: **Feasible as a desktop, opt-in, cached, WebGPU-first feature.
Not feasible as a drop-in Kokoro replacement, not feasible on mobile,
not feasible as a small download, and not real-time on WASM.**

| Dimension | Assessment |
|---|---|
| Quality | **Yes, if the port is faithful.** MNN fp16 is inside the original's own bf16 rounding envelope; int8 holds speaker cosine 0.93–1.00 and full intelligibility on sentence-split text. Ming's cloning (1,069 Controlled Elo, 1st open-weights) is a genuine upgrade over Kokoro for cloning and multilingual/expressive speech. |
| Speed | **Marginal.** Native CPU RTF 1.1–2.0 (M5 Pro) → a 10 s line takes 11–20 s to render on CPU. WebGPU *should* help (audio.cpp GPU RTF 0.17 shows the headroom), but no browser WebGPU number exists for this loop. Ship only with per-fragment streaming so the first sentence plays while later ones render; never block the UI on a full utterance. |
| Size / distribution | **The hardest product problem.** 1.80 GB (int8) vs 80 MB (Kokoro q8). Requires sharded download, resume, quota handling, and explicit user opt-in. First-load UX dominates the feature. |
| Engineering effort | **Large.** Not a conversion — a runtime: 7 ONNX exports + JS host loop + Fbank frontend + text normalization + task templates + numerical validation + benchmark matrix. Estimate in "weeks of focused work with GPU hardware for validation", not days. The MNN host code is the spec that makes this tractable at all. |
| Compatibility | **Breaking in small, manageable ways:** 24 kHz → 44.1 kHz (audit all consumers), `af_heart` → built-in/cloned Ming voice (explicit remap), scalar `speed` → task-gated controls, no token-level streaming (fragment-level instead). |

### Top 3 blockers (in order)

1. **Download + memory (1.80 GB int8, 2.91 GB fp16, + runtime RAM).**
   No browser TTS precedent at this size. Cache/IndexedDB quotas,
   tab memory pressure, and mobile exclusion are product decisions, not
   just engineering. If the product cannot accept an explicit ~2 GB
   opt-in download, stop here and use the server-side fallback (§7.1).
2. **The host loop is unwritten JavaScript, and its numerics are fragile.**
   No ONNX, no Transformers.js class, no JS host code exists. The
   autoregressive loop × 10-step flow loop × stop head × CFG must be
   ported exactly from `ming-tts-mnn` and validated graph-by-graph;
   the stop head's rounding sensitivity means even a correct port will
   produce different utterance lengths than the original. The DiT/CFM
   integrator is the single most error-prone piece.
3. **Browser speed is unproven, and int4/int8 do not fix it.**
   Quantization saves download, not loop time (§4.2); int4 additionally
   degrades timbre and long-form stability. If WebGPU does not deliver
   a large speedup on the LM+DiT loop, the feature is a slow, huge
   download — worse than Kokoro for the Anascummis conversational use
   case (where latency matters more than voice quality) and only
   justifiable for previs export (where quality matters more than latency).
   **Run the WebGPU-vs-WASM single-step benchmark before committing to
   the full build.**

### Secondary risks (do not ignore, but not blockers)

- English zero-speaker default is weak in the original itself — always
  ship a pinned voice (§6.3).
- Text normalization (FST/`pynini`) has no browser equivalent (§3.2).
- 14 paralinguistic tag spellings must come from the cookbook, not memory.
- Task-mode combination rules (§6.4) will surprise API consumers used to
  kokoro-js's free-form `{voice, speed}` options.
- The model is 5 days old: expect upstream fixes, a possible community
  ONNX export, and MNN package updates. Re-check HF/PyPI before building
  and pin exact revisions (`inclusionAI/Ming-3.0-tts-0.5B` revision,
  `ming-tts-mnn` version, MNN repo revision) in the build notes.

---

## 9. Recommended architecture + build sequence for the glue worker

### 9.1 File / module layout (proposed)

```
ming-tts-web/
  RESEARCH.md                 # this file
  src/
    ming-tts.ts               # MingTTS class: from_pretrained / generate / list_voices / stream (fragment)
    sessions.ts               # 7 ORT-web InferenceSessions, lazy-loaded per path
                              # (campplus+enc_* only when a prompt is supplied)
    host-loop.ts              # AR loop + flow loop + stop head + CFG + noise/seed
    fbank.ts                  # resample + 80-dim Fbank frontend for campplus
    normalize.ts              # JS text-normalization subset (numbers/dates/units)
    instructions.ts           # task instruction templates (verbatim from repo cookbook)
    cache.ts                  # sharded download, Cache API/IndexedDB, progress, resume
    voices.ts                 # built-in IDs, bundled prompt (wav+transcript), embedding cache
    wav.ts                    # Float32 → WAV encode; decode via AudioContext
  models/                     # NOT committed: downloaded, sharded ONNX + assets + tokenizer
    <dtype>/lm.onnx  dit.onnx  aggregator.onnx  enc_backbone.onnx
             enc_agg.onnx  voc.onnx  campplus.onnx
    assets/dims.json  host.npz  embed.npy
    tokenizer/tokenizer.json
  tests/
    graphs.test.ts            # per-graph vs MNN outputs (same inputs)
    e2e.test.ts               # vs MNN fp16: log-mel L1, CAM++ cosine, Whisper transcript
    bench.ts                  # WebGPU vs WASM, fp16 vs int8, RTF by sentence length
```

### 9.2 Build sequence (in order — do not skip 1–3)

1. **Pin + read.** Pin `inclusionAI/Ming-3.0-tts-0.5B` revision and
   `ming-tts-mnn` 0.1.3. Read the MNN host source end-to-end and the
   Ming-3.0-tts repo's cookbook. Extract: `dims.json` tensor contracts,
   instruction templates, tag list, CAM++/VAE input sample rates and
   Fbank parameters, flow-step count / CFG / sigma / patch size for 0.5B,
   sentence-splitting rules, stop-head handling.
2. **Single-step benchmark (go/no-go gate).** Export `lm` and `dit` to
   ONNX, run one LM step + one DiT step in ORT-web on WebGPU and WASM.
   If WebGPU is not clearly faster than WASM here, escalate before
   building the loop — this decides the whole feature's viability.
3. **Per-graph validation.** Export all 7 graphs; compare each against
   the MNN graph outputs on identical inputs. Fix op mismatches here,
   not in the full loop.
4. **Host loop.** Port the AR + flow loops, stop head, CFG, seeding.
   Validate end-to-end vs MNN fp16 (same seed): log-mel L1, speaker
   cosine, transcript check.
5. **Quantization + backends.** Per-graph fp16/int8; benchmark WebGPU vs
   WASM × fp16 vs int8. Keep `campplus` + stop head at high precision.
   Enforce sentence splitting for int4 if int4 ships at all.
6. **Packaging + UX.** Sharded download, Cache/IndexedDB, progress,
   quota errors, explicit opt-in, worker isolation, `sampling_rate`
   audit (24 kHz → 44.1 kHz) across Anascummis + kinoinstrument previs.
7. **Voice pinning.** Ship a pinned default (built-in or bundled prompt);
   never zero-speaker. Add cloning UI only with the consent notice (§6.3).

### 9.3 Acceptance criteria (suggested — tighten before building)

- Per-graph outputs within MNN fp16 tolerance on the same inputs.
- End-to-end (same seed, sentence-split text): speaker cosine ≥ 0.93
  (int8 bar from MNN's own 16-text test) and Whisper-transcribable output
  on a fixed test set (EN + ZH sentences, one paragraph, one cloned voice).
- Desktop WebGPU RTF ≤ 2.0 on the benchmark machine for a 10 s utterance
  (fragment-streamed: first fragment audio available well before the full
  utterance finishes); WASM RTF reported, not promised.
- Download resumes after interruption; second load is fully offline
  from cache; quota failure produces a clear error, not a silent stall.
- `generate(text, {voice:'US_Emma'})` returns `{audio, sampling_rate:
  44100}` through the existing Anascummis worker message contract.

### 9.4 Open questions the build worker must close (not answered by sources)

1. Exact per-component parameter counts / file sizes (measure from
   safetensors + MNN graphs).
2. CAM++ / VAE-encoder input sample rate and Fbank parameters for 0.5B.
3. 0.5B flow steps / CFG / sigma / patch size / max decode steps
   (Tiny's 10 / 2.0 / 0.25 are not automatically 0.5B's).
4. Exact 14 paralinguistic tag strings + instruction templates per task.
5. Contents of `host.npz` / `embed.npy` (which built-in/default embeddings
   they carry, and whether they can be trimmed for an English-only ship).
6. Whether embedding-only voice reuse (no reference latents) matches
   full cloning quality.
7. WebGPU vs WASM single-step timings (§9.2 step 2) — the go/no-go gate.
8. ~~Kinoinstrument previs export's exact kokoro-js usage~~ — **CLOSED
   2026-10-25:** `~/workspace/kinoinstrument/apps/web/src/views/previs/export/tts.ts`
   uses the same Kokoro model/voice/speed/q8/wasm setup, 4000-char cap,
   and already forwards `out.sampling_rate` as `sampleRate` (see §5.4).
   Remaining audit: single-rate `mixPcmTracks` (never mix 24 kHz and
   44.1 kHz tracks) and the lazy dynamic-import pattern.
9. Whether an English-only subset (tokenizer + built-in voices +
   dropping Chinese task paths) can materially shrink the download, or
   whether the weights are inseparably multilingual (likely the latter
   — do not promise a small English-only build without measuring).

---

## Sources

- Official model card: https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B
- Official repo (inference code, cookbook): linked from the card as
  `github.com/inclusionAI/Ming-3.0-tts` (read this before building; the
  card states the HF repo is checkpoint-only and inference lives in the
  GitHub repo)
- MNN weights: https://huggingface.co/yunfengwang/Ming-3.0-tts-0.5B-MNN
- MNN host package: https://pypi.org/project/ming-tts-mnn/
- GGUF port: https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF
- Tiny sibling (for contrast only, out of scope):
  https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny
- Architecture / quality background:
  https://kie.ai/blog/what-is-ming-3-0 ,
  https://aireiter.com/blog/ming-3-0-tts-review-local-deployment ,
  https://aireiter.com/blog/ming-3-0-tts-local-setup-guide ,
  https://nomorecoding.github.io/Ming-3.0-tts/
- Predecessor architecture (lineage only — Ming-omni-tts DeepWiki):
  https://deepwiki.com/inclusionAI/Ming-omni-tts/3-architecture ,
  https://deepwiki.com/inclusionAI/Ming-omni-tts/4.1-speech-synthesis-and-attribute-control ,
  https://deepwiki.com/inclusionAI/Ming-omni-tts/4.2-voice-design ,
  https://deepwiki.com/inclusionAI/Ming-omni-tts/3.3-flow-matching-diffusion-head
- CAM++ background: https://deepwiki.com/modelscope/3D-Speaker/7-deployment
  (CAM++ ONNX export, 80-dim Fbank, 192-dim embedding)
- kokoro-js API: `kokoro-js` npm docs (via https://socket.dev/npm/package/kokoro-js),
  https://www.javiertsalas.com/blog/kokoro-tts.html
- In-repo current usage (verified 2026-10-25):
  `~/workspace/anascummis/apps/studio/components/voice.worker.ts`
- ORT-web WebGPU/int8 caveat (single-case measurement, not a universal law):
  https://dev.to/yue_shu_c621a4a637f22396f/webgpu-vs-wasm-in-onnxruntime-web-14x-to-94x-faster-on-the-same-mac-depending-on-the-model-li5
- Browser model-size practicalities:
  https://www.sitepoint.com/webgpu-browser-ai-javascript-inference/
