# Step 2 go/no-go gate — protocol for the GPU-machine run (NOT run on this VM)

Status: NOT RUN. This file is the exact protocol so the gate can be executed
without re-deriving anything. Do not mark the gate passed until every line
below has a measured number attached.

## Why this gate exists

RESEARCH §8 blocker 3: native CPU is RTF 1.1–2.0 on an Apple M5 Pro, and
WebGPU speedup for this autoregressive × flow-matching loop is unproven.
Many small ORT-web session calls + KV-cache tensor passing may erase the
GPU advantage. If WebGPU is not clearly faster than WASM on a single step,
the full loop must not be built on the assumption that it will be.

## Prerequisites (GPU machine only)

- A machine with a real GPU + a Chromium/Chrome build with WebGPU enabled
  (`chrome://gpu` shows WebGPU available). This VM has no `/dev/dri` and no
  Chromium binary — the gate cannot run here, full stop.
- The 2.96 GB checkpoint `inclusionAI/Ming-3.0-tts-0.5B` @
  `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0` (or the MNN fp16 graphs as a
  cross-check oracle — MNN graphs are NOT convertible to ONNX; export from
  the PyTorch originals).
- PyTorch + the upstream repo @ `028c1eace193e7acdb256453da89b6e9207e052b`,
  enough RAM/disk for a 0.5B export (the 7.7 GB / ~0.6 GB-free VM is not it).

## Export (step 2 minimal scope — lm + dit ONLY)

1. Export `lm` as ONE graph with inputs: x (1,S,896), mask (1,1,S,KV),
   cos/sin (1,S,64), past_k/past_v (24,1,2,KV,64); outputs hidden (1,S,896),
   out_k/out_v (concatenated cache). Opset 17–18, dynamic axes on S and KV.
   This is the MNN `lm` contract (engine.py `MNNBackend.lm`); the stock
   Qwen2 ONNX export does NOT have the speaker/latent conditioning rows —
   the export must take `inputs_embeds`, not `input_ids`.
2. Export `dit` with inputs x (2,1,64), t (2,), c (2,1,896),
   history (2,1,64), cos/sin (1,3,64); output velocity (2,1,64).
   The CFG doubling (batch 2) is a host-side concat, not part of the graph.
3. Record: opset, per-graph file sizes (fp16), any op that ORT-web WebGPU
   refuses (fall back to WASM for that graph alone and record it).

## Benchmark (ORT-web, same inputs, same shapes, both EPs)

- Prefill shape: S = a real prefill length for "Hello, this is a test."
  with the US_Emma IP caption (measure and record S; do not use S=1).
- Decode shape: S = 1, KV = prefill length (record it).
- DiT shape: exactly the contract above (batch 2, patch 1, latent 64).
- For each (graph, EP in {webgpu, wasm}, dtype in {fp16, int8-if-exported}):
  warm up 5 runs, then time 20 timed runs, report median + p90 ms/run.
  WASM runs need COOP/COEP + SharedArrayBuffer threads enabled, or the
  comparison is meaningless — record the headers used.
- Also time the host-side overhead separately: tensor creation + session.run
  dispatch for a 1×1×64 dummy, so a "WebGPU is slower" result can be
  attributed (kernel time vs dispatch overhead).

## Pass / fail bar (decide BEFORE running, then apply honestly)

- PASS (build the full loop): WebGPU median single-step time (lm decode +
  1 dit step) is at least 2× faster than WASM on the same machine, for both
  graphs, with no unsupported-op fallback that forces the hot loop to WASM.
- FAIL / STOP: WebGPU is slower than, or within ~20% of, WASM on either
  hot graph, or the lm decode step cannot run on WebGPU at all. In that
  case: stop, report the numbers, and let the coordinator decide between
  (a) WASM-only with RTF disclosed, (b) server-side Ming with Kokoro
  in-browser, (c) GGUF/audio.cpp WASM research. Do NOT build the full loop
  on a failed gate and do NOT re-run the gate with different shapes until
  it passes.

## What counts as evidence

A bench log file committed alongside the code: machine (GPU, browser
version, ORT-web version), shapes, per-run timings (not just averages),
EP + dtype matrix, and the pass/fail call. A sentence like "WebGPU felt
faster" is not evidence.

## After a PASS only (steps 3–7, in order)

3. Per-graph validation: export all 7 graphs; compare each against the MNN
   graph outputs on identical inputs (tolerances: fp16 vs MNN fp16 —
   record max abs + relative peak; the MNN author's own envelope is
   9.0e-4 → 2.8e-2 step-level relative peak vs fp32).
4. Host loop: port ar_generate + sample_latent + stop head + CFG + seeding
   (see notes/STEP1-PINS-AND-CONTRACTS.md); validate end-to-end vs MNN
   fp16, same seed: log-mel L1, CAM++ speaker cosine (bar ≥ 0.93 for int8
   per MNN's 16-text test), Whisper transcript on EN + ZH sentences.
5. Quantization + backends: per-graph fp16/int8; benchmark WebGPU vs WASM
   × fp16 vs int8; keep campplus + stop head at high precision; default
   ship int8 (~1.80 GB total) unless WebGPU int8 proves slower (§4.2).
6. Packaging + UX: sharded download, Cache/IndexedDB, progress, quota
   errors, explicit opt-in, worker isolation, 24 kHz → 44.1 kHz audit.
7. Voice pinning: default US_Emma (IP caption, NOT zero-speaker, NOT a
   speaker-embedding slot); cloning UI only with the consent notice.
