export async function decodeAudioFile(file: File): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioContextClass =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const audioCtx = new AudioContextClass();
  try {
    return await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    void audioCtx.close();
  }
}

export function sliceToWav(
  buffer: AudioBuffer,
  startSec: number,
  endSec: number,
  targetSampleRate = 16000
): Blob {
  const originalSampleRate = buffer.sampleRate;
  const startSample = Math.max(0, Math.floor(startSec * originalSampleRate));
  const endSample = Math.min(buffer.length, Math.floor(endSec * originalSampleRate));
  const sliceLen = Math.max(0, endSample - startSample);

  const channelCount = buffer.numberOfChannels;
  const mono = new Float32Array(sliceLen);
  for (let ch = 0; ch < channelCount; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < sliceLen; i++) {
      mono[i] += data[startSample + i] / channelCount;
    }
  }

  const ratio = originalSampleRate / targetSampleRate;
  const newLength = Math.max(1, Math.floor(sliceLen / ratio));
  const resampled = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const srcIdx = i * ratio;
    const idx0 = Math.floor(srcIdx);
    const idx1 = Math.min(sliceLen - 1, idx0 + 1);
    const frac = srcIdx - idx0;
    resampled[i] = mono[idx0] * (1 - frac) + mono[idx1] * frac;
  }

  return encodeWav(resampled, targetSampleRate);
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const dataBytes = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export type RepetitionFinding = {
  isLoop: boolean;
  phrase?: string;
  count?: number;
};

export function detectRepetitionLoop(text: string, threshold = 5): RepetitionFinding {
  const tokens = text.split(/\s+/).filter(Boolean);
  for (let phraseLen = 1; phraseLen <= 5; phraseLen++) {
    let runCount = 1;
    for (let i = phraseLen; i + phraseLen <= tokens.length; i += phraseLen) {
      const prev = tokens.slice(i - phraseLen, i).join(' ');
      const curr = tokens.slice(i, i + phraseLen).join(' ');
      if (prev === curr) {
        runCount++;
        if (runCount > threshold) {
          return { isLoop: true, phrase: curr, count: runCount };
        }
      } else {
        runCount = 1;
      }
    }
  }
  return { isLoop: false };
}

export function isSuspiciouslyShort(text: string, durationSec: number): boolean {
  const cleaned = text.replace(/\[.*?\]/g, '').trim();
  const expectedMinChars = durationSec * 2;
  return cleaned.length < expectedMinChars;
}
