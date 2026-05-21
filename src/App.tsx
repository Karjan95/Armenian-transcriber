import { useState, useRef } from 'react';
import { GoogleGenAI } from '@google/genai';
import { motion, AnimatePresence } from 'motion/react';
import {
  AudioLines,
  Upload,
  Settings,
  CheckCircle2,
  Loader2,
  Copy,
  Download,
  Languages,
  Sparkles,
  Clock,
  ShieldCheck,
  AlertCircle,
  FileAudio,
  ChevronRight,
  ArrowRight,
  Bug,
  ChevronDown
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { decodeAudioFile, sliceToWav, detectRepetitionLoop, isSuspiciouslyShort } from './audio';

type DebugEntry = {
  kind: 'speaker-map' | 'chunk' | 'insights' | 'info' | 'error';
  label: string;
  chunkIndex?: number;
  startSec?: number;
  endSec?: number;
  durationSec?: number;
  promptPreview?: string;
  rawText?: string;
  charCount?: number;
  repetitionLoop?: { phrase?: string; count?: number };
  suspiciouslyShort?: boolean;
  errorMessage?: string;
  finishReason?: string;
  blockReason?: string;
  safetyRatings?: unknown;
  usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number; totalTokenCount?: number };
  retried?: boolean;
  timestamp: string;
};

type StreamCallResult = {
  text: string;
  finishReason?: string;
  blockReason?: string;
  safetyRatings?: unknown;
  usageMetadata?: DebugEntry['usageMetadata'];
  errorMessage?: string;
};

// Helper for Tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

function formatTime(sec: number): string {
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Rewrites local chunk timestamps (e.g. "00:05", "[00:05]", "[00:05.123]") to absolute by adding offsetSec.
function rewriteTimestamps(text: string, offsetSec: number): string {
  if (!offsetSec) return text;
  return text.replace(/\[?(\d{1,2}):([0-5]\d)(?:[.,](\d{1,3}))?\]?/g, (match, mm, ss) => {
    const localSec = parseInt(mm, 10) * 60 + parseInt(ss, 10);
    if (localSec > 60 * 60) return match; // already absolute (>1h), leave alone
    const abs = localSec + offsetSec;
    const absMin = Math.floor(abs / 60);
    const absSec = abs % 60;
    const wrapped = match.startsWith('[');
    const stamp = `${absMin.toString().padStart(2, '0')}:${absSec.toString().padStart(2, '0')}`;
    return wrapped ? `[${stamp}]` : stamp;
  });
}

// Model Fleet Definitions
interface ModelSpec {
  id: string;
  name: string;
  tag: string;
  desc: string;
  rate: number; // per 1M tokens
}

const MODELS: ModelSpec[] = [
  { id: 'gemini-2.5-pro',        name: '2.5 Pro',        tag: 'Expert',    desc: 'Highest accuracy',         rate: 1.25 },
  { id: 'gemini-2.5-flash',      name: '2.5 Flash',      tag: 'Balanced',  desc: 'Best price/performance',   rate: 0.30 },
  { id: 'gemini-2.5-flash-lite', name: '2.5 Flash Lite', tag: 'Instant',   desc: 'Cheapest & fastest',       rate: 0.10 },
  { id: 'gemini-1.5-pro',        name: '1.5 Pro',        tag: 'Long-form', desc: '2M context, audio-friendly', rate: 1.25 },
  { id: 'gemini-1.5-flash',      name: '1.5 Flash',      tag: 'Long-form', desc: '1M context, audio-friendly', rate: 0.10 },
];

export default function App() {
  // AI Studio injects GEMINI_API_KEY at runtime (via vite.config.ts define).
  // When present, use it — that key is the one authorized for AI Studio's
  // app sandbox. Otherwise fall back to the UI input for local dev.
  const injectedKey = (typeof process !== 'undefined' && process.env?.GEMINI_API_KEY) || '';
  const isInjectedKeyValid = injectedKey && injectedKey !== 'MY_GEMINI_API_KEY';
  const [apiKey, setApiKey] = useState(isInjectedKeyValid ? injectedKey : '');
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [file, setFile] = useState<File | null>(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'analyzing' | 'done' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const [transcript, setTranscript] = useState('');
  const [insights, setInsights] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [debugLog, setDebugLog] = useState<DebugEntry[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const [expandedDebugIdx, setExpandedDebugIdx] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const pushDebug = (entry: Omit<DebugEntry, 'timestamp'>) => {
    setDebugLog((prev) => [...prev, { ...entry, timestamp: new Date().toISOString() }]);
  };

  type StreamArgs = Parameters<GoogleGenAI['models']['generateContentStream']>[0];
  const streamCall = async (
    ai: GoogleGenAI,
    args: StreamArgs,
    onDelta?: (text: string, accumulated: string) => void
  ): Promise<StreamCallResult> => {
    const result: StreamCallResult = { text: '' };
    try {
      const stream = await ai.models.generateContentStream(args);
      for await (const chunk of stream) {
        const anyChunk = chunk as unknown as {
          text?: string;
          candidates?: { finishReason?: string; safetyRatings?: unknown }[];
          promptFeedback?: { blockReason?: string };
          usageMetadata?: DebugEntry['usageMetadata'];
        };
        if (anyChunk.text) {
          result.text += anyChunk.text;
          onDelta?.(anyChunk.text, result.text);
        }
        const cand = anyChunk.candidates?.[0];
        if (cand?.finishReason) result.finishReason = cand.finishReason;
        if (cand?.safetyRatings) result.safetyRatings = cand.safetyRatings;
        if (anyChunk.promptFeedback?.blockReason) result.blockReason = anyChunk.promptFeedback.blockReason;
        if (anyChunk.usageMetadata) result.usageMetadata = anyChunk.usageMetadata;
      }
    } catch (err: unknown) {
      result.errorMessage = err instanceof Error ? err.message : String(err);
    }
    return result;
  };

  // Cost Estimation
  const selectedModelSpec = MODELS.find(m => m.id === selectedModel);
  const modelRate = selectedModelSpec?.rate || 0;
  // Gemini 1.5/2.0/3.1 uses ~50 tokens/sec
  const estTokens = audioDuration * 50; 
  // 3-pass process (Speaker ID + Transcript + Insights)
  const estCost = ((estTokens * 3) / 1000000) * modelRate;

  const addLog = (msg: string) => setLog(prev => [...prev.slice(-3), msg]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      const audio = new Audio(url);
      audio.onloadedmetadata = () => {
        setAudioDuration(audio.duration);
        URL.revokeObjectURL(url);
      };
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const downloadFile = (content: string, filename: string) => {
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const runTranscription = async () => {
    if (!file || !apiKey) return;

    setStatus('uploading');
    setProgress(2);
    setLog(["Initializing secure data link..."]);
    setErrorMessage('');
    setDebugLog([]);

    const CHUNK_SECONDS = 600; // 10-minute chunks (smaller = safer vs hallucination loops)

    try {
      if (!audioDuration) {
        throw new Error("Audio duration is still being calculated or failed to load. Please re-select the file or wait a moment.");
      }

      const ai = new GoogleGenAI({ apiKey });

      // Step A: decode + slice the audio locally
      addLog("Decoding audio locally...");
      pushDebug({ kind: 'info', label: 'Decoding audio in browser', durationSec: audioDuration });
      const decoded = await decodeAudioFile(file);
      pushDebug({
        kind: 'info',
        label: `Decoded: ${decoded.sampleRate}Hz, ${decoded.numberOfChannels}ch, ${decoded.duration.toFixed(1)}s`,
      });

      const sliceBoundaries: { start: number; end: number }[] = [];
      for (let s = 0; s < decoded.duration; s += CHUNK_SECONDS) {
        sliceBoundaries.push({ start: s, end: Math.min(decoded.duration, s + CHUNK_SECONDS) });
      }

      addLog(`Audio split into ${sliceBoundaries.length} chunk(s).`);

      // Step B: encode each chunk to WAV + upload to File API
      const uploadedChunks: { uri: string; name: string; mimeType: string; start: number; end: number; durationSec: number }[] = [];
      for (let i = 0; i < sliceBoundaries.length; i++) {
        const { start, end } = sliceBoundaries[i];
        const durationSec = end - start;

        addLog(`Encoding & uploading chunk ${i + 1}/${sliceBoundaries.length}...`);
        const wavBlob = sliceToWav(decoded, start, end, 16000);
        const wavFile = new File([wavBlob], `chunk-${i + 1}.wav`, { type: 'audio/wav' });

        const uploaded = await ai.files.upload({
          file: wavFile,
          config: { mimeType: 'audio/wav', displayName: wavFile.name },
        });

        let isReady = false;
        let attempt = 0;
        while (!isReady) {
          const pollFile = await ai.files.get({ name: uploaded.name! });
          if (pollFile.state === 'ACTIVE') {
            isReady = true;
          } else if (pollFile.state === 'FAILED') {
            throw new Error(`Cloud indexing failed for chunk ${i + 1}.`);
          } else {
            attempt++;
            if (attempt > 60) throw new Error(`Cloud indexing timeout for chunk ${i + 1}.`);
            await new Promise((r) => setTimeout(r, 2000));
          }
        }

        uploadedChunks.push({
          uri: uploaded.uri!,
          name: uploaded.name!,
          mimeType: 'audio/wav',
          start,
          end,
          durationSec,
        });

        pushDebug({
          kind: 'info',
          label: `Chunk ${i + 1} uploaded`,
          chunkIndex: i + 1,
          startSec: start,
          endSec: end,
          durationSec,
        });

        setProgress(5 + Math.round(((i + 1) / sliceBoundaries.length) * 15));
      }

      // Step C: Speaker map — run on the first chunk (most representative + cheaper than full file)
      setStatus('processing');
      setProgress(25);
      addLog("Mapping speaker signatures...");
      setInsights("Generating Speaker Map...\n");

      const speakerPrompt = `You are an expert audio analyst conducting speaker diarization. Listen to the audio and identify EVERY distinct speaker.
1. There may be 4, 5, or more different voices. Pay close attention to subtle voice changes.
2. If a speaker introduces themselves, or if others refer to them by name, you MUST use their actual name as their label.
3. If their name is never mentioned, label them as "Speaker 1", "Speaker 2", etc.
4. For each identified speaker, briefly describe tone, pitch, gender, and speaking style.
Provide a complete, exhaustive mapping of every voice heard.`;

      const firstChunk = uploadedChunks[0];
      const speakerResult = await streamCall(
        ai,
        {
          model: selectedModel,
          contents: [{
            parts: [
              { fileData: { mimeType: firstChunk.mimeType, fileUri: firstChunk.uri } },
              { text: speakerPrompt },
            ],
          }],
        },
        (_d, acc) => setInsights(acc)
      );

      const speakerMapText = speakerResult.text;
      const speakerMap = speakerMapText || "Standard Speaker Map";

      pushDebug({
        kind: 'speaker-map',
        label: 'Speaker map response',
        promptPreview: speakerPrompt,
        rawText: speakerMapText,
        charCount: speakerMapText.length,
        finishReason: speakerResult.finishReason,
        blockReason: speakerResult.blockReason,
        safetyRatings: speakerResult.safetyRatings,
        usageMetadata: speakerResult.usageMetadata,
        errorMessage: speakerResult.errorMessage,
      });
      addLog("Speaker signatures mapped.");

      // Step D: Transcribe each chunk INDEPENDENTLY (no time window guesswork — each upload IS the segment)
      let combinedTranscript = "";

      for (let i = 0; i < uploadedChunks.length; i++) {
        const c = uploadedChunks[i];
        setProgress(30 + Math.round((i / uploadedChunks.length) * 55));
        addLog(`Verbatim transcription ${i + 1}/${uploadedChunks.length}...`);

        const chunkOffsetLabel = `${formatTime(c.start)}–${formatTime(c.end)}`;
        const chunkPrompt = `You are an expert transcriber. Transcribe THIS audio file verbatim in Armenian.

Use these speaker labels: ${speakerMap}

Rules:
1. Transcribe ALL speech in this audio file from start to end. Do not skip portions.
2. Use timestamps RELATIVE TO THIS AUDIO FILE (start at 00:00). Format each line as "MM:SS Speaker N: <text>".
3. Transcribe verbatim — including repeated words actually spoken by the speaker. Only stop if you start fabricating words that are NOT in the audio.
4. If a portion is truly silent, output "[Silence]" once and continue with the next utterance.
5. Output ONLY the transcript — no preamble, no commentary, no markdown.`;

        const runCall = () =>
          streamCall(
            ai,
            {
              model: selectedModel,
              contents: [{
                parts: [
                  { fileData: { mimeType: c.mimeType, fileUri: c.uri } },
                  { text: chunkPrompt },
                ],
              }],
              config: { temperature: 0.1 },
            },
            (_d, acc) => setTranscript(combinedTranscript + (i === 0 ? "" : "\n\n") + acc)
          );

        let result = await runCall();
        let retried = false;
        if (!result.text.trim() && !result.errorMessage) {
          addLog(`Chunk ${i + 1} returned empty — retrying once...`);
          await new Promise((r) => setTimeout(r, 1500));
          retried = true;
          result = await runCall();
        }

        const rawChunkText = result.text;
        const repetition = detectRepetitionLoop(rawChunkText);
        const shortFlag = isSuspiciouslyShort(rawChunkText, c.durationSec);

        pushDebug({
          kind: result.errorMessage || !rawChunkText ? 'error' : 'chunk',
          label: `Chunk ${i + 1} (${chunkOffsetLabel})${retried ? ' [retried]' : ''}`,
          chunkIndex: i + 1,
          startSec: c.start,
          endSec: c.end,
          durationSec: c.durationSec,
          promptPreview: chunkPrompt,
          rawText: rawChunkText,
          charCount: rawChunkText.length,
          repetitionLoop: repetition.isLoop ? { phrase: repetition.phrase, count: repetition.count } : undefined,
          suspiciouslyShort: shortFlag,
          finishReason: result.finishReason,
          blockReason: result.blockReason,
          safetyRatings: result.safetyRatings,
          usageMetadata: result.usageMetadata,
          errorMessage: result.errorMessage,
          retried,
        });

        const offsetSec = Math.round(c.start);
        const rewritten = rewriteTimestamps(rawChunkText, offsetSec);
        const finalized = rewritten.trim();
        if (finalized) {
          combinedTranscript += (combinedTranscript ? "\n\n" : "") + `--- [${chunkOffsetLabel}] ---\n` + finalized;
        } else {
          combinedTranscript += (combinedTranscript ? "\n\n" : "") + `--- [${chunkOffsetLabel}] ---\n[NO TRANSCRIPT RETURNED — see Debug Console]`;
        }
      }

      setTranscript(combinedTranscript);

      // Step E: Insights — text-based, from the transcript we just produced
      setStatus('analyzing');
      setProgress(90);
      addLog("Synthesizing context from transcript...");
      setInsights((prev) => prev + "\n\n---\n\nSynthesizing context...\n");

      const insightPrompt = `Below is a verbatim Armenian transcript. Produce a Critical Bilingual Analysis in BOTH Armenian [AM] and English [EN].
For each major point, output one paragraph prefixed "[AM]" then one prefixed "[EN]".

Cover: (1) executive summary, (2) key decisions and action items, (3) strategic risks or open questions.

IMPORTANT: Base your analysis ONLY on what is in the transcript below. Do NOT invent details.

--- TRANSCRIPT START ---
${combinedTranscript}
--- TRANSCRIPT END ---`;

      const insightPreambleLen = (speakerMapText + "\n\n---\n\n").length;
      const insightResult = await streamCall(
        ai,
        {
          model: selectedModel,
          contents: [{ parts: [{ text: insightPrompt }] }],
        },
        (_d, acc) => setInsights(speakerMapText + "\n\n---\n\n" + acc)
      );

      const finalInsights = speakerMapText + "\n\n---\n\n" + insightResult.text;
      setInsights(finalInsights);
      void insightPreambleLen;

      pushDebug({
        kind: insightResult.errorMessage ? 'error' : 'insights',
        label: 'Insights response (transcript-based)',
        promptPreview: insightPrompt.slice(0, 500) + (insightPrompt.length > 500 ? '… [truncated]' : ''),
        rawText: insightResult.text,
        charCount: insightResult.text.length,
        finishReason: insightResult.finishReason,
        blockReason: insightResult.blockReason,
        safetyRatings: insightResult.safetyRatings,
        usageMetadata: insightResult.usageMetadata,
        errorMessage: insightResult.errorMessage,
      });

      setProgress(100);
      setStatus('done');
      addLog("Multi-pass process complete.");

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("Transcription Error:", err);
      pushDebug({ kind: 'error', label: 'Fatal error', errorMessage: msg });
      setStatus('error');
      setErrorMessage(msg || "Engine failure detected. Check browser console for details.");
    }
  };

  const downloadDebug = () => {
    const payload = {
      file: file ? { name: file.name, sizeBytes: file.size, type: file.type, durationSec: audioDuration } : null,
      model: selectedModel,
      generatedAt: new Date().toISOString(),
      entries: debugLog,
      finalTranscript: transcript,
      finalInsights: insights,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'debug-trace.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-bg text-tx font-sans selection:bg-ac/20">
      <div className="max-w-3xl mx-auto px-6 py-16 md:py-24 space-y-12">
        
        {/* Simplified Modern Header */}
        <header className="flex flex-col items-center text-center space-y-4">
          <motion.div 
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="w-12 h-12 bg-ac rounded-2xl flex items-center justify-center text-white shadow-xl shadow-ac/30 mb-2"
          >
            <Sparkles className="w-6 h-6" />
          </motion.div>
          <div className="space-y-2">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-tx">
              Armenian Transcriber <span className="text-ac">Pro</span>
            </h1>
            <p className="text-td font-medium max-w-sm mx-auto">
              Precision audio decoding and bilingual intelligence for the modern workspace.
            </p>
          </div>
        </header>

        <main className="space-y-6">
          
          {/* Main Controls Card */}
          <section className="glass-card p-8 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            
            {/* API Auth Sub-section */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-[11px] font-bold text-tm uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck className="w-3.5 h-3.5 text-ac" /> Secure Identity
                </label>
                {!apiKey && status === 'idle' && (
                  <span className="text-[10px] text-rd animate-pulse font-bold uppercase tracking-tighter">Key Required</span>
                )}
              </div>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Paste Gemini API Key from aistudio.google.com/apikey..."
                className="input-field shadow-sm"
              />
            </div>

            {/* Model Selector Sub-section */}
            <div className="space-y-4">
              <label className="text-[11px] font-bold text-tm uppercase tracking-widest flex items-center gap-2">
                <Settings className="w-3.5 h-3.5 text-ac" /> Intelligence Selection
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {MODELS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => status === 'idle' && setSelectedModel(m.id)}
                    className={cn(
                      "flex flex-col items-start p-4 rounded-2xl border transition-all text-left",
                      selectedModel === m.id 
                        ? "bg-ac text-white border-ac shadow-lg shadow-ac/20" 
                        : "bg-sf hover:bg-sf2 border-bd"
                    )}
                  >
                    <span className="text-xs font-bold mb-0.5">{m.name}</span>
                    <span className={cn(
                      "text-[10px] font-medium opacity-80",
                      selectedModel === m.id ? "text-white" : "text-tm"
                    )}>{m.tag}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* File Zone Sub-section */}
            <div className="space-y-4">
              <label className="text-[11px] font-bold text-tm uppercase tracking-widest flex items-center gap-2">
                <FileAudio className="w-3.5 h-3.5 text-ac" /> Source Media
              </label>
              
              <div 
                onClick={() => status === 'idle' && fileInputRef.current?.click()}
                className={cn(
                  "group relative overflow-hidden flex flex-col items-center justify-center rounded-[24px] p-12 cursor-pointer transition-all border-2 border-dashed",
                  file ? "border-gn/40 bg-gn/5" : "border-bd hover:border-ac/30 hover:bg-sf2",
                  status !== 'idle' && "opacity-50 cursor-not-allowed"
                )}
              >
                <input 
                  type="file" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="audio/*"
                  className="hidden" 
                />
                
                {file ? (
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-gn/10 rounded-full flex items-center justify-center mx-auto text-gn">
                      <CheckCircle2 className="w-8 h-8" />
                    </div>
                    <div className="space-y-1">
                      <p className="font-bold text-tx">{file.name}</p>
                      <p className="text-xs text-tm">{(file.size / (1024 * 1024)).toFixed(1)} MB · Ready for processing</p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center space-y-4">
                    <div className="w-16 h-16 bg-sf2 rounded-full flex items-center justify-center mx-auto text-tm group-hover:bg-ac/10 group-hover:text-ac transition-colors">
                      <Upload className="w-6 h-6" />
                    </div>
                    <div className="space-y-1">
                      <p className="font-bold text-tx whitespace-nowrap">Drop recording or click to browse</p>
                      <p className="text-xs text-tm">Secure cloud transfer supports up to 2GB</p>
                    </div>
                  </div>
                )}
              </div>

              <AnimatePresence>
                {file && status === 'idle' && (
                  <motion.div 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="space-y-4 pt-4 border-t border-bd overflow-hidden"
                  >
                    <div className="flex items-center justify-between px-6 py-4 bg-sf2/50 rounded-2xl border border-bd">
                      <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                          <Clock className="w-3.5 h-3.5 text-tm" />
                          <span className="text-xs font-bold text-td">{Math.round(audioDuration/60)}m {Math.round(audioDuration%60)}s</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Sparkles className="w-3.5 h-3.5 text-tm" />
                          <span className="text-xs font-bold text-td">${estCost.toFixed(3)} est.</span>
                        </div>
                      </div>
                      <button 
                        onClick={runTranscription}
                        className="bg-ac text-white px-5 py-2 rounded-xl text-xs font-bold hover:bg-ac-hover transition-colors flex items-center gap-2"
                      >
                        Start Analysis <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 pb-2">
                      <div className="p-3 bg-sf2/30 rounded-xl border border-bd/50 flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gn/10 flex items-center justify-center text-gn">
                          <ShieldCheck className="w-4 h-4" />
                        </div>
                        <div>
                          <p className="text-[10px] font-bold text-tx uppercase tracking-tighter leading-none mb-1">Cloud Storage: $0.00</p>
                          <p className="text-[9px] text-tm leading-tight">File API upload & 48h indexing is free.</p>
                        </div>
                      </div>
                      <div className="p-3 bg-sf2/30 rounded-xl border border-bd/50 flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-ac/10 flex items-center justify-center text-ac">
                          <Settings className="w-4 h-4" />
                        </div>
                        <div>
                          <p className="text-[10px] font-bold text-tx uppercase tracking-tighter leading-none mb-1">Verbatim Guard</p>
                          <p className="text-[9px] text-tm leading-tight">Logical segments prevent text truncation.</p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </section>

          {/* Progress / Logs Card */}
          <AnimatePresence>
            {status !== 'idle' && (
              <motion.section 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="glass-card p-6"
              >
                <div className="space-y-5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {status === 'error' ? (
                        <AlertCircle className="w-4 h-4 text-rd" />
                      ) : status === 'done' ? (
                        <CheckCircle2 className="w-4 h-4 text-gn" />
                      ) : (
                        <Loader2 className="w-4 h-4 text-ac animate-spin" />
                      )}
                      <span className="text-[11px] font-bold text-tm uppercase tracking-widest">
                        {status === 'uploading' && "Data Stream Active"}
                        {status === 'processing' && "Linguistic Engine"}
                        {status === 'analyzing' && "Context Analysis"}
                        {status === 'done' && "Analysis Ready"}
                        {status === 'error' && "System Interrupt"}
                      </span>
                    </div>
                    <span className="text-xs font-bold text-ac">{progress}%</span>
                  </div>

                  <div className="h-1.5 bg-sf2 rounded-full overflow-hidden">
                    <motion.div 
                      key={status}
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      className={cn("h-full transition-all duration-500", status === 'error' ? 'bg-rd' : 'bg-ac')}
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    {log.map((msg, i) => (
                      <div key={i} className="flex items-center gap-2 opacity-60">
                        <ChevronRight className="w-3 h-3 text-ac" />
                        <span className="text-[11px] font-medium text-td">{msg}</span>
                      </div>
                    ))}
                  </div>

                  {status === 'error' && (
                    <p className="text-xs text-rd font-bold pt-2 border-t border-rd/10">{errorMessage}</p>
                  )}
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Output Section */}
          <AnimatePresence>
            {(transcript || insights || debugLog.length > 0 || status === 'done') && (
              <motion.section 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-6"
              >
                <div className="flex items-center gap-4 text-tm">
                  <div className="h-px flex-1 bg-bd" />
                  <span className="text-[10px] font-bold uppercase tracking-widest">Analysis Results</span>
                  <div className="h-px flex-1 bg-bd" />
                </div>

                {/* Insight Engine */}
                {insights && (
                  <div className="glass-card p-10 space-y-6 overflow-hidden">
                    <div className="flex items-center justify-between border-b border-bd pb-6">
                      <label className="text-[11px] font-bold text-tm uppercase tracking-widest flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-ac" /> Bilingual Intelligence
                      </label>
                      <button 
                        onClick={() => downloadFile(insights, 'intelligence.md')}
                        className="flex items-center gap-2 px-4 py-2 hover:bg-sf2 rounded-xl text-tm hover:text-tx transition-all text-[11px] font-bold border border-bd"
                      >
                        <Download className="w-3.5 h-3.5" /> Download Insights
                      </button>
                    </div>
                    <div className="markdown-body">
                      <ReactMarkdown
                        components={{
                          p: ({ children }) => {
                            const content = String(children);
                            if (content.includes('[AM]')) {
                              return (
                                <div className="p-4 bg-sf2/30 rounded-2xl border-l-4 border-ac mb-4">
                                  <span className="block text-[8px] font-black text-ac uppercase mb-2 tracking-tighter opacity-50">Armenian</span>
                                  <p className="m-0 text-tx font-armenian leading-relaxed">{content.replace('[AM]', '').trim()}</p>
                                </div>
                              );
                            }
                            if (content.includes('[EN]')) {
                              return (
                                <div className="p-4 bg-sf2/20 rounded-2xl border-l-4 border-tm mb-4">
                                  <span className="block text-[8px] font-black text-tm uppercase mb-2 tracking-tighter opacity-50">English</span>
                                  <p className="m-0 text-td leading-relaxed">{content.replace('[EN]', '').trim()}</p>
                                </div>
                              );
                            }
                            return <p>{children}</p>;
                          }
                        }}
                      >
                        {insights}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}

                {/* Full Transcript */}
                {transcript && (
                  <div className="glass-card p-10 space-y-6">
                    <div className="flex items-center justify-between border-b border-bd pb-6">
                      <label className="text-[11px] font-bold text-tm uppercase tracking-widest flex items-center gap-2">
                        <AudioLines className="w-4 h-4 text-ac" /> Full Transcript
                      </label>
                      <div className="flex items-center justify-end gap-2">
                        <button 
                          onClick={() => downloadFile(transcript, 'transcript.md')}
                          className="flex items-center gap-2 px-4 py-2 hover:bg-sf2 rounded-xl text-tm hover:text-tx transition-all text-[11px] font-bold border border-bd"
                        >
                          <Download className="w-3.5 h-3.5" /> Download Transcript
                        </button>
                        <button 
                          onClick={() => copyToClipboard(transcript)}
                          className="flex items-center gap-2 px-4 py-2 hover:bg-sf2 rounded-xl text-tm hover:text-tx transition-all text-[11px] font-bold border border-bd"
                        >
                          <Copy className="w-3.5 h-3.5" /> Copy Text
                        </button>
                      </div>
                    </div>
                    <div className="text-td font-armenian text-base leading-relaxed whitespace-pre-wrap max-h-[500px] overflow-y-auto pr-4 scrollbar-thin scrollbar-thumb-bd">
                      {transcript}
                    </div>
                  </div>
                )}

                {/* Debug Console */}
                {debugLog.length > 0 && (
                  <div className="glass-card p-6 space-y-4">
                    <div className="flex items-center justify-between border-b border-bd pb-4">
                      <button
                        onClick={() => setShowDebug((v) => !v)}
                        className="flex items-center gap-2 text-[11px] font-bold text-tm uppercase tracking-widest"
                      >
                        <Bug className="w-4 h-4 text-ac" />
                        Debug Console ({debugLog.length})
                        <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", showDebug && "rotate-180")} />
                      </button>
                      <button
                        onClick={downloadDebug}
                        className="flex items-center gap-2 px-4 py-2 hover:bg-sf2 rounded-xl text-tm hover:text-tx transition-all text-[11px] font-bold border border-bd"
                      >
                        <Download className="w-3.5 h-3.5" /> Download debug.json
                      </button>
                    </div>

                    {showDebug && (
                      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
                        {debugLog.map((entry, idx) => {
                          const expanded = expandedDebugIdx === idx;
                          const flagged = entry.kind === 'error' || entry.repetitionLoop || entry.suspiciouslyShort;
                          return (
                            <div
                              key={idx}
                              className={cn(
                                "rounded-xl border text-[12px]",
                                flagged ? "border-rd/30 bg-rd/5" : "border-bd bg-sf2/30"
                              )}
                            >
                              <button
                                onClick={() => setExpandedDebugIdx(expanded ? null : idx)}
                                className="w-full flex items-center justify-between px-4 py-3 text-left"
                              >
                                <div className="flex items-center gap-3 min-w-0">
                                  <span
                                    className={cn(
                                      "text-[9px] font-bold uppercase px-2 py-1 rounded-md tracking-wider shrink-0",
                                      entry.kind === 'error' && "bg-rd/10 text-rd",
                                      entry.kind === 'chunk' && "bg-ac/10 text-ac",
                                      entry.kind === 'speaker-map' && "bg-gn/10 text-gn",
                                      entry.kind === 'insights' && "bg-gn/10 text-gn",
                                      entry.kind === 'info' && "bg-sf3 text-tm"
                                    )}
                                  >
                                    {entry.kind}
                                  </span>
                                  <span className="font-bold text-tx truncate">{entry.label}</span>
                                  {entry.charCount !== undefined && (
                                    <span className="text-tm text-[10px] shrink-0">{entry.charCount} chars</span>
                                  )}
                                  {entry.repetitionLoop && (
                                    <span className="text-rd text-[10px] font-bold shrink-0">
                                      ⚠ loop ×{entry.repetitionLoop.count}
                                    </span>
                                  )}
                                  {entry.suspiciouslyShort && (
                                    <span className="text-rd text-[10px] font-bold shrink-0">⚠ short</span>
                                  )}
                                </div>
                                <ChevronDown className={cn("w-3.5 h-3.5 text-tm transition-transform shrink-0", expanded && "rotate-180")} />
                              </button>

                              {expanded && (
                                <div className="px-4 pb-4 space-y-3 border-t border-bd/50 pt-3">
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px] text-tm">
                                    {entry.chunkIndex !== undefined && <div><b className="text-tx">Index:</b> {entry.chunkIndex}</div>}
                                    {entry.startSec !== undefined && <div><b className="text-tx">Start:</b> {formatTime(entry.startSec)}</div>}
                                    {entry.endSec !== undefined && <div><b className="text-tx">End:</b> {formatTime(entry.endSec)}</div>}
                                    {entry.durationSec !== undefined && <div><b className="text-tx">Duration:</b> {entry.durationSec.toFixed(1)}s</div>}
                                    {entry.finishReason && (
                                      <div><b className="text-tx">finishReason:</b> <span className={entry.finishReason !== 'STOP' ? 'text-rd font-bold' : ''}>{entry.finishReason}</span></div>
                                    )}
                                    {entry.blockReason && <div className="text-rd"><b>blockReason:</b> {entry.blockReason}</div>}
                                    {entry.usageMetadata?.promptTokenCount !== undefined && (
                                      <div><b className="text-tx">In tokens:</b> {entry.usageMetadata.promptTokenCount}</div>
                                    )}
                                    {entry.usageMetadata?.candidatesTokenCount !== undefined && (
                                      <div><b className="text-tx">Out tokens:</b> {entry.usageMetadata.candidatesTokenCount}</div>
                                    )}
                                    {entry.retried && <div className="text-ac font-bold">↻ retried</div>}
                                    <div className="col-span-2"><b className="text-tx">At:</b> {entry.timestamp}</div>
                                  </div>

                                  {entry.safetyRatings ? (
                                    <details className="text-[11px]">
                                      <summary className="cursor-pointer text-tm font-bold mb-1">Safety ratings</summary>
                                      <pre className="bg-sf p-3 rounded-lg border border-bd whitespace-pre-wrap font-mono text-[10px] text-td max-h-32 overflow-y-auto">
{JSON.stringify(entry.safetyRatings, null, 2)}
                                      </pre>
                                    </details>
                                  ) : null}

                                  {entry.repetitionLoop && (
                                    <div className="p-3 bg-rd/10 rounded-lg text-[11px] text-rd font-bold">
                                      Repetition detected: phrase "{entry.repetitionLoop.phrase}" repeats {entry.repetitionLoop.count} times.
                                    </div>
                                  )}
                                  {entry.suspiciouslyShort && (
                                    <div className="p-3 bg-rd/10 rounded-lg text-[11px] text-rd font-bold">
                                      Output is suspiciously short for {entry.durationSec?.toFixed(0)}s of audio — likely truncated by the API.
                                    </div>
                                  )}

                                  {entry.errorMessage && (
                                    <div className="p-3 bg-rd/10 rounded-lg text-[11px] text-rd whitespace-pre-wrap">
                                      {entry.errorMessage}
                                    </div>
                                  )}

                                  {entry.promptPreview && (
                                    <details className="text-[11px]">
                                      <summary className="cursor-pointer text-tm font-bold mb-1">Prompt sent</summary>
                                      <pre className="bg-sf p-3 rounded-lg border border-bd whitespace-pre-wrap font-mono text-[10px] text-td max-h-48 overflow-y-auto">
{entry.promptPreview}
                                      </pre>
                                    </details>
                                  )}

                                  {entry.rawText !== undefined && (
                                    <details className="text-[11px]" open>
                                      <summary className="cursor-pointer text-tm font-bold mb-1">Raw API response</summary>
                                      <pre className="bg-sf p-3 rounded-lg border border-bd whitespace-pre-wrap font-armenian text-[11px] text-tx max-h-64 overflow-y-auto">
{entry.rawText || '[empty response]'}
                                      </pre>
                                    </details>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </motion.section>
            )}
          </AnimatePresence>

        </main>

        <footer className="text-center pt-8 border-t border-bd pb-24">
          <p className="text-[10px] uppercase font-bold text-tm tracking-[4px]">
            Powered by Google Cloud Inference
          </p>
        </footer>
      </div>
    </div>
  );
}
