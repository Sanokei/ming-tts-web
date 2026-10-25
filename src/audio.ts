// Host-side audio: WAV I/O, torchaudio-compatible resampling, kaldi fbank.
// Ported from ming-tts-mnn 0.1.3 audioops.py. None of this belongs in a graph.

const EPSILON = 1.1920928955078125e-7; // float32 eps, as in the Python port

// ---------------------------------------------------------------- WAV I/O
export function decodeWav(bytes: Uint8Array): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ascii = (off: number, len: number) => {
    let s = '';
    for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(off + i));
    return s;
  };
  if (ascii(0, 4) !== 'RIFF' || ascii(8, 4) !== 'WAVE') {
    throw new Error('decodeWav: not a RIFF/WAVE file');
  }
  let pos = 12;
  let fmt: DataView | null = null;
  let fmtOff = 0;
  let dataOff = -1;
  let dataLen = 0;
  while (pos + 8 <= bytes.length) {
    const id = ascii(pos, 4);
    const size = view.getUint32(pos + 4, true);
    if (id === 'fmt ') {
      fmt = view;
      fmtOff = pos + 8;
    } else if (id === 'data') {
      dataOff = pos + 8;
      dataLen = size;
    }
    pos += 8 + size + (size & 1);
  }
  if (!fmt || dataOff < 0) throw new Error('decodeWav: wav file without fmt/data chunk');
  let tag = fmt.getUint16(fmtOff, true);
  const channels = fmt.getUint16(fmtOff + 2, true);
  const rate = fmt.getUint32(fmtOff + 4, true);
  const bits = fmt.getUint16(fmtOff + 14, true);
  if (tag === 0xfffe && fmtOff + 40 <= bytes.length) {
    tag = fmt.getUint16(fmtOff + 24, true); // WAVE_FORMAT_EXTENSIBLE
  }
  const frames = bytes.subarray(dataOff, dataOff + dataLen);
  const fview = new DataView(frames.buffer, frames.byteOffset, frames.byteLength);
  let data: Float32Array;
  if (tag === 1) {
    if (bits === 8) {
      data = new Float32Array(frames.length);
      for (let i = 0; i < frames.length; i++) data[i] = (frames[i] - 128) / 128;
    } else if (bits === 16) {
      const n = Math.floor(frames.length / 2);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = fview.getInt16(i * 2, true) / 32768;
    } else if (bits === 24) {
      const n = Math.floor(frames.length / 3);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let v = frames[i * 3] | (frames[i * 3 + 1] << 8) | (frames[i * 3 + 2] << 16);
        if (v & 0x800000) v -= 0x1000000;
        data[i] = v / 8388608;
      }
    } else if (bits === 32) {
      const n = Math.floor(frames.length / 4);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = fview.getInt32(i * 4, true) / 2147483648;
    } else {
      throw new Error(`decodeWav: unsupported PCM sample width: ${bits} bits`);
    }
  } else if (tag === 3) {
    if (bits === 32) {
      const n = Math.floor(frames.length / 4);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = fview.getFloat32(i * 4, true);
    } else if (bits === 64) {
      const n = Math.floor(frames.length / 8);
      data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = fview.getFloat64(i * 8, true);
    } else {
      throw new Error(`decodeWav: unsupported float sample width: ${bits} bits`);
    }
  } else {
    throw new Error(`decodeWav: unsupported wav format tag: ${tag}`);
  }
  if (channels > 1) {
    // Multi-channel files are averaged, as the reference does.
    const mono = new Float32Array(Math.floor(data.length / channels));
    for (let i = 0; i < mono.length; i++) {
      let s = 0;
      for (let ch = 0; ch < channels; ch++) s += data[i * channels + ch];
      mono[i] = s / channels;
    }
    return { samples: mono, sampleRate: rate };
  }
  return { samples: data, sampleRate: rate };
}

/** Encode mono float32 audio as 16-bit PCM WAV (the format the reference saves). */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const dataSize = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buf);
  const ascii = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, 'data');
  view.setUint32(40, dataSize, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, Math.round(v * 32767), true);
  }
  return new Uint8Array(buf);
}

// ---------------------------------------------------------------- resample
function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/**
 * torchaudio.transforms.Resample (bandlimited sinc, Hann window, rolloff 0.99).
 * Ported from MNN audioops.resample — a different resampler would change the
 * prompt audio and therefore the speaker embedding and prompt latent.
 * NOTE: this is O(n * taps) and is not a fast polyphase implementation; it is
 * bit-faithful to the reference kernel construction. For long prompts prefer
 * a native resampler ONLY after verifying it matches torchaudio's output.
 */
export function resample(
  waveform: Float32Array,
  origFreq: number,
  newFreq: number,
): Float32Array {
  const x = waveform;
  if (origFreq === newFreq) return new Float32Array(x);
  const g = gcd(origFreq, newFreq);
  const o = Math.floor(origFreq / g);
  const n = Math.floor(newFreq / g);
  const lowpassWidth = 6;
  const rolloff = 0.99;
  const baseFreq = Math.min(o, n) * rolloff;
  const width = Math.ceil((lowpassWidth * o) / baseFreq);
  // Kernel phases: (nPhases, K) where K = 2*width + o.
  const kLen = 2 * width + o;
  const kernels: number[][] = [];
  for (let phase = 0; phase < n; phase++) {
    const row: number[] = [];
    for (let j = 0; j < kLen; j++) {
      const idx = j - width;
      let t = phase / n - 0 + idx / o; // polyphase phase offset, see below
      // Recompute exactly as torchaudio: t = arange(0,-n,-1)/n + idx/o, * baseFreq
      t = (-phase / n + idx / o) * baseFreq;
      t = Math.min(Math.max(t, -lowpassWidth), lowpassWidth);
      const window = Math.cos((t * Math.PI) / lowpassWidth / 2) ** 2;
      const tt = t * Math.PI;
      const sinc = tt === 0 ? 1 : Math.sin(tt) / tt;
      row.push(sinc * (window * (baseFreq / o)));
    }
    kernels.push(row);
  }
  const padded = new Float32Array(x.length + width + (width + o));
  padded.set(x, width);
  const outLen = Math.ceil((n * x.length) / o);
  const out = new Float32Array(outLen);
  // conv with stride o, phases interleaved back into one signal.
  for (let outIdx = 0; outIdx < outLen; outIdx++) {
    const frame = Math.floor(outIdx / n);
    const phase = outIdx % n;
    const start = frame * o;
    let s = 0;
    const kernel = kernels[phase];
    for (let k = 0; k < kLen; k++) {
      s += padded[start + k] * kernel[k];
    }
    out[outIdx] = s;
  }
  return out;
}

// ---------------------------------------------------------------- kaldi fbank
function melScale(freq: number): number {
  return 1127 * Math.log(1 + freq / 700);
}

function melBanks(
  numBins: number,
  paddedWindow: number,
  sampleFreq: number,
  lowFreq: number,
  highFreq: number,
): number[][] {
  const numFftBins = Math.floor(paddedWindow / 2);
  const nyquist = 0.5 * sampleFreq;
  if (highFreq <= 0) highFreq += nyquist;
  const fftBinWidth = sampleFreq / paddedWindow;
  const melLow = melScale(lowFreq);
  const melHigh = melScale(highFreq);
  const delta = (melHigh - melLow) / (numBins + 1);
  const bins: number[][] = [];
  for (let b = 0; b < numBins; b++) {
    const left = melLow + b * delta;
    const center = melLow + (b + 1) * delta;
    const right = melLow + (b + 2) * delta;
    const row: number[] = [];
    for (let k = 0; k < numFftBins; k++) {
      const freq = fftBinWidth * k;
      const mel = 1127 * Math.log(1 + freq / 700);
      const up = (mel - left) / (center - left);
      const down = (right - mel) / (right - center);
      row.push(Math.max(0, Math.min(up, down)));
    }
    row.push(0); // DC bin contributes nothing (matches the Python pad)
    bins.push(row);
  }
  return bins;
}

function poveyWindow(size: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < size; i++) {
    const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
    out.push(hann ** 0.85);
  }
  return out;
}

/** Magnitude spectrum via naive DFT (correct for test-scale frames). */
function rfftPower(frame: number[], padded: number): number[] {
  const out: number[] = [];
  const half = Math.floor(padded / 2) + 1;
  for (let k = 0; k < half; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < frame.length; n++) {
      const ang = (-2 * Math.PI * k * n) / padded;
      re += frame[n] * Math.cos(ang);
      im += frame[n] * Math.sin(ang);
    }
    out.push(re * re + im * im);
  }
  return out;
}

/**
 * torchaudio.compliance.kaldi.fbank with the release's parameters:
 * 80 bins, 16 kHz, 25 ms frame, 10 ms shift, low 20 Hz, pre-emphasis 0.97,
 * dither 0, snip_edges, DC removal, power + log. Returns (T, 80).
 */
export function kaldiFbank(
  waveform: Float32Array,
  numMelBins = 80,
  sampleFrequency = 16000,
): number[][] {
  const frameLength = 25;
  const frameShift = 10;
  const windowShift = Math.floor(sampleFrequency * frameShift * 0.001);
  const windowSize = Math.floor(sampleFrequency * frameLength * 0.001);
  const paddedWindowSize =
    windowSize === 0 ? 1 : 2 ** Math.ceil(Math.log2(windowSize));
  const x = waveform;
  if (x.length < windowSize) return [];
  const m = 1 + Math.floor((x.length - windowSize) / windowShift);
  const win = poveyWindow(windowSize);
  const bins = melBanks(numMelBins, paddedWindowSize, sampleFrequency, 20, 0);
  const out: number[][] = [];
  for (let f = 0; f < m; f++) {
    let frame = Array.from(x.slice(f * windowShift, f * windowShift + windowSize));
    const mean = frame.reduce((a, b) => a + b, 0) / frame.length;
    frame = frame.map((v) => v - mean);
    const pre: number[] = frame.map((v, i) =>
      i === 0 ? v : v - 0.97 * frame[i - 1],
    );
    const windowed = pre.map((v, i) => v * win[i]);
    const padded = [...windowed, ...new Array(paddedWindowSize - windowSize).fill(0)];
    const spectrum = rfftPower(padded, paddedWindowSize);
    const row: number[] = [];
    for (let b = 0; b < numMelBins; b++) {
      let e = 0;
      for (let k = 0; k < spectrum.length; k++) e += spectrum[k] * bins[b][k];
      row.push(Math.log(Math.max(e, EPSILON)));
    }
    out.push(row);
  }
  return out;
}

/**
 * The features CAM++ expects: 80-bin kaldi fbank, mean-normalised per
 * utterance (mean subtracted across time). Input may be at any rate; it is
 * resampled to 16 kHz first. Returns (T, 80) — callers add the batch dim.
 */
export function speakerFeatures(
  waveform: Float32Array,
  sampleRate: number,
): number[][] {
  let audio = waveform;
  if (sampleRate !== 16000) audio = resample(audio, sampleRate, 16000);
  const feat = kaldiFbank(audio, 80, 16000);
  if (feat.length === 0) return feat;
  const means = new Array(80).fill(0);
  for (const row of feat) for (let b = 0; b < 80; b++) means[b] += row[b];
  for (let b = 0; b < 80; b++) means[b] /= feat.length;
  return feat.map((row) => row.map((v, b) => v - means[b]));
}
