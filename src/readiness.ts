/** Safe to import in a browser before loading ONNX Runtime or any weights. */
export const MING_RUNTIME_STATUS = Object.freeze({
  model: 'inclusionAI/Ming-3.0-tts-0.5B',
  bundledEngine: false,
  realSynthesisVerified: false,
  estimatedInt8DownloadBytes: 1_800_000_000,
  workingAlternative: 'https://github.com/hexgrad/kokoro/tree/main/kokoro.js',
  reason: 'The real Ming fragment engine and end-to-end synthesis gate are not complete. ONNX installation, a model URL, or a session factory alone does not enable synthesis.',
});

export function mingReadiness(options: {engine?: unknown} = {}) {
  const engine = options.engine as {synthesizeFragment?: unknown} | null | undefined;
  const ready = typeof engine?.synthesizeFragment === 'function';
  return {ready, provider: 'ming' as const, engineSource: ready ? 'caller-injected' as const : null, realSynthesisVerified: false, reason: ready ? 'A fragment engine was supplied by the caller; its quality must be verified separately.' : MING_RUNTIME_STATUS.reason};
}
