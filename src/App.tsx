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
  ArrowRight
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper for Tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
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
  { id: 'gemini-1.5-pro', name: '1.5 Pro', tag: 'Expert', desc: 'Highest Accuracy', rate: 1.25 },
  { id: 'gemini-1.5-flash', name: '1.5 Flash', tag: 'Balanced', desc: 'Fast Audio Engine', rate: 0.10 },
  { id: 'gemini-2.0-flash', name: '2.0 Flash', tag: 'Fast', desc: 'Capable Engine', rate: 0.10 },
  { id: 'gemini-2.5-flash', name: '2.5 Flash', tag: 'Fast', desc: 'Capable Engine', rate: 0.10 },
  { id: 'gemini-3.5-flash', name: '3.5 Flash', tag: 'Balanced', desc: 'Fast Audio Engine', rate: 0.10 },
  { id: 'gemini-3.1-pro-preview', name: '3.1 Pro', tag: 'Expert', desc: 'Highest Accuracy', rate: 1.25 },
  { id: 'gemini-3.1-flash-lite', name: '3.1 Flash Lite', tag: 'Instant', desc: 'Fast & Reliable', rate: 0.08 },
];

export default function App() {
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const [file, setFile] = useState<File | null>(null);
  const [audioDuration, setAudioDuration] = useState(0);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'analyzing' | 'done' | 'error'>('idle');
  const [progress, setProgress] = useState(0);
  const [log, setLog] = useState<string[]>([]);
  const [transcript, setTranscript] = useState('');
  const [insights, setInsights] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);

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
    setProgress(5);
    setLog(["Initializing secure data link..."]);
    setErrorMessage('');

    try {
      if (!audioDuration) {
         throw new Error("Audio duration is still being calculated or failed to load. Please re-select the file or wait a moment.");
      }

      const ai = new GoogleGenAI({ apiKey });
      
      addLog("Transmitting audio stream to cloud...");
      const uploadedFile = await ai.files.upload({
        file: file,
        config: {
          mimeType: file.type || 'audio/mpeg',
          displayName: file.name
        }
      });
      
      const fileUri = uploadedFile.uri;
      const fileName = uploadedFile.name;

      setProgress(20);
      addLog("Preparing linguistic engine...");
      let isReady = false;
      let attempt = 0;
      while (!isReady) {
        const pollFile = await ai.files.get({ name: fileName });
        if (pollFile.state === 'ACTIVE') {
            isReady = true;
        } else if (pollFile.state === 'FAILED') {
            throw new Error("Cloud indexing failed.");
        } else {
            attempt++;
            if (attempt > 60) throw new Error("Cloud indexing timeout (2 minutes).");
            await new Promise(r => setTimeout(r, 2000));
        }
      }

      // Pass 1: Speaker Profiling
      setStatus('processing');
      setProgress(25);
      addLog("Mapping speaker signatures... (Please wait, analyzing full audio context)");
      setInsights("Generating Speaker Map...\n");
      
      const speakerStream = await ai.models.generateContentStream({
        model: selectedModel,
        contents: [{
          parts: [
            { fileData: { mimeType: file.type || 'audio/mpeg', fileUri: fileUri } },
            { text: `You are an expert audio analyst conducting speaker diarization. Carefully listen to the ENTIRE audio file and identify EVERY distinct speaker.
1. There may be 4, 5, or more different voices. Pay very close attention to subtle voice changes.
2. If a speaker introduces themselves, or if others refer to them by name, you MUST use their actual name as their label.
3. If their name is never mentioned, label them as "Speaker 1", "Speaker 2", etc.
4. For each identified speaker, provide a brief description of their tone, pitch, gender, and speaking style to establish a robust signature.
Provide a complete, exhaustive mapping of every voice heard.` }
          ]
        }]
      });
      
      let speakerMapText = "";
      for await (const chunk of speakerStream) {
        if (chunk.text) {
          speakerMapText += chunk.text;
          setInsights(speakerMapText);
        }
      }
      const speakerMap = speakerMapText || "Standard Speaker Map";
      addLog("Speaker signatures mapped.");

      // Pass 2: Managed Chunking
      const CHUNK_SIZE = 900; // 15 mins chunks for better verbatim safety
      const OVERLAP = 60; // 1 min overlap
      const chunks = [];
      let currentStart = 0;
      while (currentStart < audioDuration) {
        const currentEnd = Math.min(currentStart + CHUNK_SIZE, audioDuration);
        chunks.push({ start: currentStart, end: currentEnd });
        if (currentEnd >= audioDuration) break;
        currentStart = currentEnd - OVERLAP;
      }

      let combinedTranscript = "";
      let previousTail = "";

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        setProgress(30 + Math.round((i / chunks.length) * 50));
        addLog(`Verbatim transcription part ${i + 1}/${chunks.length}...`);

        const chunkStream = await ai.models.generateContentStream({
          model: selectedModel,
          contents: [{
            parts: [
              { fileData: { mimeType: file.type || 'audio/mpeg', fileUri: fileUri } },
              { text: `You are an expert transcriber. Transcribe the audio segment verbatim in Armenian.

Time Segment to focus on: ${Math.round(chunk.start)}s to ${Math.round(chunk.end)}s.
Speaker Labels to use: ${speakerMap}

${previousTail ? `Context (The previous segment ended with these words): "${previousTail}"\nCONTINUE EXACTLY FROM WHERE THIS LEFT OFF WITHOUT REPEATING.` : ""}

CRITICAL RULES TO PREVENT REPETITION ERRORS (HALLUCINATION LOOPS):
1. ONLY transcribe from ${Math.round(chunk.start)}s to ${Math.round(chunk.end)}s.
2. DO NOT REPEAT YOURSELF. If you find yourself caught in a loop and repeating the exact same dialogue lines multiple times, YOU MUST IMMEDIATELY STOP GENERATING.
3. If there is silence or no clear speech at the end of the segment, DO NOT invent words or repeat previous sentences. Simply output "[Silence]" and stop.
4. Include exact timestamps on each line.` }
            ]
          }],
          config: {
             temperature: 0.1
          }
        });

        let currentChunkText = "";
        for await (const tsChunk of chunkStream) {
          if (tsChunk.text) {
             currentChunkText += tsChunk.text;
             setTranscript(combinedTranscript + (i === 0 ? "" : "\n\n") + currentChunkText);
          }
        }
        
        const finalizedChunkText = currentChunkText.trim();
        if (finalizedChunkText) {
            combinedTranscript += (i === 0 ? "" : "\n\n") + finalizedChunkText;
            previousTail = finalizedChunkText.slice(-200).replace(/\n/g, ' ').trim();
        }
      }

      setTranscript(combinedTranscript);
      
      setStatus('analyzing');
      setProgress(90);
      addLog("Synthesizing context...");
      setInsights((prev) => prev + "\n\n---\n\nSynthesizing context...\n");

      const insightStream = await ai.models.generateContentStream({
        model: selectedModel,
        contents: [
          {
            parts: [
              { fileData: { mimeType: file.type || 'audio/mpeg', fileUri: fileUri } },
              { text: "Critical Bilingual Analysis [AM]/[EN]: summary, key actions, strategic risks." }
            ]
          }
        ],
      });

      let finalInsights = speakerMapText + "\n\n---\n\n";
      for await (const chunk of insightStream) {
         if (chunk.text) {
            finalInsights += chunk.text;
            setInsights(finalInsights);
         }
      }

      setProgress(100);
      setStatus('done');
      addLog("Multi-pass process complete.");

    } catch (err: any) {
      console.error("Transcription Error:", err);
      setStatus('error');
      setErrorMessage(err.message || "Engine failure detected. Check browser console for details.");
    }
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
                placeholder="Paste Gemini Cloud API Key..."
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
            {(transcript || insights || status === 'done') && (
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
