import { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Zap,
  Play,
  Square,
  Trash2,
  Copy,
  Check,
  Eye,
  EyeOff,
  Terminal,
  Cpu,
  Layers,
  Sliders,
  ChevronDown,
  Info,
  Clock,
  Gauge,
  FileText,
  AlertTriangle,
  Flame,
  CheckCircle2,
  ExternalLink,
  Settings2
} from 'lucide-react';
import {
  GROQ_MODELS,
  analyzePrompt,
  generateCurlCode,
  generatePythonCode,
  generateNodeCode
} from './types';

export default function App() {
  // Config state
  const [apiKey, setApiKey] = useState<string>(() => {
    return localStorage.getItem('groq_api_key') || '';
  });
  const [showKey, setShowKey] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('llama-3.3-70b-versatile');
  const [systemPrompt, setSystemPrompt] = useState<string>('You are an expert, concise, and helpful AI assistant.');
  const [prompt, setPrompt] = useState<string>('Explain how quantum entanglement works in 3 clear bullet points, then provide a simple thought experiment.');
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(2048);
  const [topP] = useState<number>(1.0);

  // Execution state
  const [response, setResponse] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // Real-time telemetry
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [tokensGenerated, setTokensGenerated] = useState<number>(0);
  const [promptTokens, setPromptTokens] = useState<number>(0);
  const [totalTokens, setTotalTokens] = useState<number>(0);
  const [speedTps, setSpeedTps] = useState<number>(0);

  // UI tabs & view options
  const [activeTab, setActiveTab] = useState<'output' | 'analysis' | 'code'>('output');
  const [codeLang, setCodeLang] = useState<'curl' | 'python' | 'node'>('python');
  const [copiedKey, setCopiedKey] = useState(false);
  const [copiedOutput, setCopiedOutput] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);
  const [showSystemModal, setShowSystemModal] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const outputBottomRef = useRef<HTMLDivElement | null>(null);
  const timerRef = useRef<number | null>(null);

  // Persist API key
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('groq_api_key', apiKey);
    }
  }, [apiKey]);

  // Prompt analytics
  const promptAnalysis = analyzePrompt(prompt);
  const currentModelConfig = GROQ_MODELS.find(m => m.id === selectedModel) || GROQ_MODELS[0];

  // Auto-scroll response when streaming
  useEffect(() => {
    if (isStreaming && outputBottomRef.current) {
      outputBottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [response, isStreaming]);

  // Clean timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleRun = async () => {
    if (!prompt.trim()) {
      setError('Please provide a prompt to run.');
      return;
    }
    if (!apiKey.trim()) {
      setError('Please enter a Groq API Key. You can get a free key at console.groq.com');
      return;
    }

    // Reset runtime states
    setError(null);
    setResponse('');
    setIsStreaming(true);
    setTokensGenerated(0);
    setElapsedMs(0);
    setSpeedTps(0);

    const start = performance.now();

    // Track elapsed time
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      setElapsedMs(performance.now() - start);
    }, 50);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const messages = [];
      if (systemPrompt.trim()) {
        messages.push({ role: 'system', content: systemPrompt.trim() });
      }
      messages.push({ role: 'user', content: prompt.trim() });

      const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey.trim()}`
        },
        body: JSON.stringify({
          model: selectedModel,
          messages,
          temperature,
          max_tokens: maxTokens,
          top_p: topP,
          stream: true
        }),
        signal: controller.signal
      });

      if (!res.ok) {
        let errMessage = `Error ${res.status}: ${res.statusText}`;
        try {
          const errData = await res.json();
          if (errData?.error?.message) {
            errMessage = errData.error.message;
          }
        } catch {
          // ignore parsing error
        }
        throw new Error(errMessage);
      }

      if (!res.body) {
        throw new Error('Response body is null.');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumulatedText = '';
      let chunkTokens = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const cleanLine = line.trim();
          if (!cleanLine || !cleanLine.startsWith('data: ')) continue;
          const dataStr = cleanLine.slice(6);
          if (dataStr === '[DONE]') continue;

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta?.content || '';
            if (delta) {
              accumulatedText += delta;
              chunkTokens += 1;
              setResponse(accumulatedText);
              setTokensGenerated(chunkTokens);

              // Update real-time TPS
              const currentElapsedSec = (performance.now() - start) / 1000;
              if (currentElapsedSec > 0.1) {
                setSpeedTps(Math.round(chunkTokens / currentElapsedSec));
              }
            }

            // Check if usage info is available in chunk
            if (parsed.usage) {
              setPromptTokens(parsed.usage.prompt_tokens || 0);
              setTokensGenerated(parsed.usage.completion_tokens || chunkTokens);
              setTotalTokens(parsed.usage.total_tokens || 0);
            }
          } catch {
            // continue reading chunk
          }
        }
      }

      // Finish metrics calculation
      const finalElapsed = performance.now() - start;
      setElapsedMs(finalElapsed);
      const estCompletionTokens = Math.max(chunkTokens, Math.ceil(accumulatedText.length / 4));
      setTokensGenerated(estCompletionTokens);
      const estPromptTokens = promptAnalysis.estimatedPromptTokens + (systemPrompt.length ? Math.ceil(systemPrompt.length / 4) : 0);
      setPromptTokens(estPromptTokens);
      setTotalTokens(estPromptTokens + estCompletionTokens);
      
      const seconds = Math.max(finalElapsed / 1000, 0.05);
      setSpeedTps(Math.round(estCompletionTokens / seconds));

    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        // User aborted intentionally
      } else {
        setError(err instanceof Error ? err.message : 'An unexpected error occurred while communicating with Groq API.');
      }
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (timerRef.current) clearInterval(timerRef.current);
    setIsStreaming(false);
  };

  const handleClear = () => {
    setPrompt('');
    setResponse('');
    setError(null);
    setTokensGenerated(0);
    setPromptTokens(0);
    setTotalTokens(0);
    setSpeedTps(0);
    setElapsedMs(0);
  };

  const copyToClipboard = (text: string, type: 'key' | 'output' | 'code') => {
    navigator.clipboard.writeText(text);
    if (type === 'key') {
      setCopiedKey(true);
      setTimeout(() => setCopiedKey(false), 2000);
    } else if (type === 'output') {
      setCopiedOutput(true);
      setTimeout(() => setCopiedOutput(false), 2000);
    } else if (type === 'code') {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

  const samplePrompts = [
    { label: 'Quantum Physics', text: 'Explain how quantum entanglement works in 3 clear bullet points, then provide a simple thought experiment.' },
    { label: 'System Architecture', text: 'Design a high-throughput real-time notification engine using WebSockets, Redis, and Kafka. Provide an architectural overview and latency trade-offs.' },
    { label: 'Rust vs Go', text: 'Compare Rust and Go for building high-concurrency microservices. Highlight memory management, developer ergonomics, and performance.' },
    { label: 'Creative Fiction', text: 'Write the opening paragraph of a neo-noir cyberpunk mystery set in underwater Neo-Tokyo in the year 2088.' }
  ];

  return (
    <div className="min-h-screen bg-[#090b10] text-slate-100 flex flex-col font-sans selection:bg-purple-600/30 selection:text-purple-200">
      
      {/* Top Header */}
      <header className="border-b border-white/[0.07] bg-[#0c0f17]/80 backdrop-blur-md sticky top-0 z-40 px-5 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 via-indigo-600 to-cyan-500 p-[1px] shadow-lg shadow-purple-500/20">
            <div className="w-full h-full bg-[#0d1019] rounded-[11px] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-extrabold tracking-tight text-base sm:text-lg bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                INSIDE THE LLM
              </h1>
              <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 font-medium">
                Playground
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">Ultra-low latency Groq inference sandbox</p>
          </div>
        </div>

        {/* Global Key & Status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-300 font-mono text-[11px]">Groq LPUs Live</span>
          </div>

          <a
            href="https://console.groq.com/keys"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 transition-colors px-2 py-1 rounded hover:bg-cyan-500/10"
          >
            Get Groq Key <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        
        {/* Left Column: Controls & Prompt Input (7 cols on lg) */}
        <section className="lg:col-span-7 flex flex-col gap-5">

          {/* Model Selector & API Key Card */}
          <div className="bg-[#101420]/90 border border-white/[0.08] rounded-2xl p-5 shadow-xl shadow-black/40 backdrop-blur-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/5 rounded-full blur-3xl pointer-events-none"></div>
            
            <div className="flex flex-col gap-4">
              
              {/* API Key Input */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-purple-400" />
                    Groq API Key
                  </label>
                  {apiKey && (
                    <span className="text-[11px] text-emerald-400 font-medium flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Stored securely in browser
                    </span>
                  )}
                </div>
                
                <div className="relative flex items-center">
                  <input
                    type={showKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="gsk_..."
                    className="w-full bg-[#0a0d14] border border-white/[0.08] focus:border-purple-500/50 focus:ring-1 focus:ring-purple-500/50 rounded-xl px-4 py-2.5 text-sm font-mono text-slate-200 placeholder:text-slate-600 outline-none transition-smooth"
                  />
                  <div className="absolute right-2 flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] rounded-lg transition-colors"
                      title={showKey ? 'Hide Key' : 'Show Key'}
                    >
                      {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                    {apiKey && (
                      <button
                        type="button"
                        onClick={() => copyToClipboard(apiKey, 'key')}
                        className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] rounded-lg transition-colors"
                        title="Copy Key"
                      >
                        {copiedKey ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Model Selector Bar */}
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 mb-1.5">
                  <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                  Model Architecture
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {GROQ_MODELS.map((model) => {
                    const isSelected = selectedModel === model.id;
                    return (
                      <button
                        key={model.id}
                        type="button"
                        onClick={() => setSelectedModel(model.id)}
                        className={`text-left p-3 rounded-xl border transition-smooth relative flex flex-col justify-between ${
                          isSelected
                            ? 'bg-purple-950/20 border-purple-500/50 shadow-md shadow-purple-900/20'
                            : 'bg-[#0b0e16]/60 border-white/[0.06] hover:bg-white/[0.03] hover:border-white/[0.12]'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-1 mb-1">
                          <span className={`text-xs font-bold leading-tight ${isSelected ? 'text-purple-300' : 'text-slate-200'}`}>
                            {model.name}
                          </span>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/[0.05] text-cyan-400 border border-cyan-500/20">
                            {model.speed}
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-400 line-clamp-1 mb-2">
                          {model.description}
                        </p>
                        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
                          <span>Ctx: {(model.contextWindow / 1024).toFixed(0)}k</span>
                          <span>•</span>
                          <span>Max: {(model.maxOutputTokens / 1024).toFixed(0)}k</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Sliders: Temperature & Max Tokens */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-white/[0.06]">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium flex items-center gap-1">
                      <Sliders className="w-3 h-3 text-purple-400" /> Temperature
                    </span>
                    <span className="font-mono text-purple-300 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/20">
                      {temperature.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="2.0"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>Precise (0.0)</span>
                    <span>Creative (2.0)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium flex items-center gap-1">
                      <Layers className="w-3 h-3 text-cyan-400" /> Max Tokens
                    </span>
                    <span className="font-mono text-cyan-300 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                      {maxTokens}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max={currentModelConfig.maxOutputTokens}
                    step="64"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>64</span>
                    <span>{currentModelConfig.maxOutputTokens} tokens</span>
                  </div>
                </div>
              </div>

              {/* Collapsible System Prompt Toggle */}
              <div className="pt-2 border-t border-white/[0.06]">
                <button
                  type="button"
                  onClick={() => setShowSystemModal(!showSystemModal)}
                  className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 transition-colors py-1 cursor-pointer"
                >
                  <span className="flex items-center gap-1.5 font-medium">
                    <Settings2 className="w-3.5 h-3.5 text-purple-400" />
                    System Instructions {systemPrompt.trim() && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>}
                  </span>
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${showSystemModal ? 'rotate-180' : ''}`} />
                </button>

                {showSystemModal && (
                  <div className="mt-2.5">
                    <textarea
                      value={systemPrompt}
                      onChange={(e) => setSystemPrompt(e.target.value)}
                      placeholder="You are an expert AI..."
                      rows={2}
                      className="w-full bg-[#0a0d14] border border-white/[0.08] focus:border-purple-500/50 rounded-xl p-3 text-xs font-mono text-slate-300 placeholder:text-slate-600 outline-none resize-none transition-smooth"
                    />
                  </div>
                )}
              </div>

            </div>
          </div>

          {/* Prompt Input Card */}
          <div className="bg-[#101420]/90 border border-white/[0.08] rounded-2xl p-5 shadow-xl shadow-black/40 backdrop-blur-sm flex flex-col gap-4">
            
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-cyan-400" />
                Prompt
              </label>
              
              {/* Quick Preset Buttons */}
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-slate-500 mr-1 hidden sm:inline">Quick presets:</span>
                <div className="flex gap-1 overflow-x-auto max-w-[280px] sm:max-w-none">
                  {samplePrompts.slice(0, 3).map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setPrompt(sample.text)}
                      className="text-[10px] bg-white/[0.04] hover:bg-purple-500/20 hover:text-purple-300 border border-white/[0.06] hover:border-purple-500/30 px-2 py-1 rounded-md transition-smooth whitespace-nowrap text-slate-300 cursor-pointer"
                    >
                      {sample.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Prompt Textarea */}
            <div className="relative">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter instructions, questions, or code to run through the LLM..."
                rows={6}
                className="w-full bg-[#0a0d14] border border-white/[0.08] focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 rounded-xl p-4 text-sm text-slate-100 placeholder:text-slate-600 outline-none resize-y transition-smooth leading-relaxed"
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault();
                    if (!isStreaming) handleRun();
                  }
                }}
              />
            </div>

            {/* Prompt Quick Info Footer */}
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 px-1">
              <div className="flex items-center gap-3 font-mono text-[11px]">
                <span>{prompt.length} chars</span>
                <span>•</span>
                <span>{promptAnalysis.wordCount} words</span>
                <span>•</span>
                <span className="text-cyan-400">~{promptAnalysis.estimatedPromptTokens} prompt tokens</span>
              </div>
              <span className="text-[10px] text-slate-500 font-mono hidden sm:inline">
                Press <kbd className="px-1.5 py-0.5 bg-white/[0.08] rounded border border-white/[0.1] text-slate-300">⌘+Enter</kbd> to run
              </span>
            </div>

            {/* Action Bar (Run, Stop, Clear) */}
            <div className="flex items-center gap-3 pt-2">
              {!isStreaming ? (
                <button
                  type="button"
                  onClick={handleRun}
                  className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white font-semibold py-3 px-5 rounded-xl shadow-lg shadow-purple-600/25 hover:shadow-cyan-500/25 active:scale-[0.99] transition-smooth text-sm cursor-pointer"
                >
                  <Play className="w-4 h-4 fill-current" />
                  Run Inference
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleStop}
                  className="flex-1 flex items-center justify-center gap-2 bg-rose-600 hover:bg-rose-500 text-white font-semibold py-3 px-5 rounded-xl shadow-lg shadow-rose-600/30 active:scale-[0.99] transition-smooth text-sm cursor-pointer"
                >
                  <Square className="w-4 h-4 fill-current" />
                  Stop Generation
                </button>
              )}

              <button
                type="button"
                onClick={handleClear}
                className="flex items-center gap-1.5 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] text-slate-300 hover:text-white px-4 py-3 rounded-xl transition-smooth text-sm font-medium cursor-pointer"
                title="Clear Prompt & Output"
              >
                <Trash2 className="w-4 h-4 text-slate-400" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            </div>

            {/* Error Notification */}
            {error && (
              <div className="bg-rose-950/40 border border-rose-500/40 rounded-xl p-3.5 flex items-start gap-3 text-rose-200 text-xs">
                <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold text-rose-300 mb-0.5">Execution Failed</p>
                  <p className="text-rose-200/90 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

          </div>

        </section>

        {/* Right Column: Output & Inspector (5 cols on lg) */}
        <section className="lg:col-span-5 flex flex-col gap-5">
          
          {/* Output Card */}
          <div className="bg-[#101420]/90 border border-white/[0.08] rounded-2xl p-5 shadow-xl shadow-black/40 backdrop-blur-sm flex flex-col min-h-[580px] relative overflow-hidden">
            
            {/* Tab navigation & Output actions */}
            <div className="flex items-center justify-between pb-3.5 border-b border-white/[0.08]">
              <div className="flex items-center gap-1 bg-black/40 p-1 rounded-xl border border-white/[0.05]">
                <button
                  type="button"
                  onClick={() => setActiveTab('output')}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-smooth cursor-pointer ${
                    activeTab === 'output'
                      ? 'bg-purple-600/30 text-purple-200 border border-purple-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Output
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('analysis')}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-smooth cursor-pointer ${
                    activeTab === 'analysis'
                      ? 'bg-cyan-600/30 text-cyan-200 border border-cyan-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Prompt Analysis
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('code')}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-smooth cursor-pointer ${
                    activeTab === 'code'
                      ? 'bg-purple-600/30 text-purple-200 border border-purple-500/40 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Code Preview
                </button>
              </div>

              {activeTab === 'output' && response && (
                <button
                  type="button"
                  onClick={() => copyToClipboard(response, 'output')}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 bg-white/[0.04] hover:bg-white/[0.08] px-2.5 py-1.5 rounded-lg border border-white/[0.06] transition-smooth font-mono cursor-pointer"
                  title="Copy Output"
                >
                  {copiedOutput ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedOutput ? 'Copied' : 'Copy'}</span>
                </button>
              )}
            </div>

            {/* Live Metrics Telemetry Strip */}
            <div className="grid grid-cols-3 gap-2 py-3 border-b border-white/[0.06]">
              <div className="bg-[#0b0e16]/80 rounded-xl p-2.5 border border-white/[0.05] flex flex-col">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1">
                  <Gauge className="w-3 h-3 text-cyan-400" /> Speed
                </span>
                <span className="text-sm font-bold font-mono text-cyan-300 mt-0.5">
                  {speedTps > 0 ? `${speedTps} t/s` : '—'}
                </span>
              </div>
              <div className="bg-[#0b0e16]/80 rounded-xl p-2.5 border border-white/[0.05] flex flex-col">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1">
                  <Flame className="w-3 h-3 text-purple-400" /> Tokens
                </span>
                <span className="text-sm font-bold font-mono text-purple-300 mt-0.5">
                  {tokensGenerated > 0 ? tokensGenerated : '—'}
                </span>
              </div>
              <div className="bg-[#0b0e16]/80 rounded-xl p-2.5 border border-white/[0.05] flex flex-col">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1">
                  <Clock className="w-3 h-3 text-emerald-400" /> Latency
                </span>
                <span className="text-sm font-bold font-mono text-emerald-300 mt-0.5">
                  {elapsedMs > 0 ? `${(elapsedMs / 1000).toFixed(2)}s` : '—'}
                </span>
              </div>
            </div>

            {/* Tab 1: Live Output Screen */}
            {activeTab === 'output' && (
              <div className="flex-1 overflow-y-auto py-4 font-mono text-xs sm:text-[13px] leading-relaxed text-slate-200">
                {response ? (
                  <div className="whitespace-pre-wrap font-sans leading-relaxed text-slate-200">
                    {response}
                    {isStreaming && <span className="cursor-blink"></span>}
                  </div>
                ) : isStreaming ? (
                  <div className="flex items-center gap-2 text-cyan-400">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 pulse-streaming"></span>
                    <span className="text-xs">Streaming from Groq LPU engine...</span>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500">
                    <div className="w-12 h-12 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-3">
                      <Terminal className="w-6 h-6 text-slate-600" />
                    </div>
                    <p className="text-sm font-medium text-slate-400">Ready to infer</p>
                    <p className="text-xs text-slate-600 max-w-[260px] mt-1">
                      Configure your prompt on the left and click Run Inference to stream responses with token telemetry.
                    </p>
                  </div>
                )}
                <div ref={outputBottomRef} />
              </div>
            )}

            {/* Tab 2: Prompt Analysis */}
            {activeTab === 'analysis' && (
              <div className="flex-1 overflow-y-auto py-4 space-y-4">
                <div className="bg-[#0b0e16] border border-white/[0.06] rounded-xl p-4">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Info className="w-3.5 h-3.5 text-cyan-400" />
                    Structural Breakdown
                  </h3>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-slate-500 block text-[10px] uppercase">Character Count</span>
                      <span className="font-mono text-slate-200 text-sm font-bold">{promptAnalysis.charCount}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-slate-500 block text-[10px] uppercase">Word Count</span>
                      <span className="font-mono text-slate-200 text-sm font-bold">{promptAnalysis.wordCount}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-slate-500 block text-[10px] uppercase">Estimated Tokens</span>
                      <span className="font-mono text-cyan-300 text-sm font-bold">~{promptAnalysis.estimatedPromptTokens}</span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-slate-500 block text-[10px] uppercase">Reading Duration</span>
                      <span className="font-mono text-slate-200 text-sm font-bold">{promptAnalysis.readingTimeSec}s</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#0b0e16] border border-white/[0.06] rounded-xl p-4">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-purple-400" />
                    Heuristic Diagnostics
                  </h3>
                  <div className="space-y-2.5 text-xs">
                    <div className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]">
                      <span className="text-slate-400">Prompt Complexity</span>
                      <span className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium ${
                        promptAnalysis.complexityScore === 'Complex'
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          : promptAnalysis.complexityScore === 'Moderate'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                          : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      }`}>
                        {promptAnalysis.complexityScore}
                      </span>
                    </div>

                    <div className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]">
                      <span className="text-slate-400">Code Snippets / Syntax</span>
                      <span className="font-mono text-xs text-slate-300">
                        {promptAnalysis.hasCode ? 'Detected' : 'None'}
                      </span>
                    </div>

                    <div className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]">
                      <span className="text-slate-400">Role & System Directives</span>
                      <span className="font-mono text-xs text-slate-300">
                        {promptAnalysis.hasSystemDirectives ? 'Present' : 'Standard Query'}
                      </span>
                    </div>

                    <div className="flex items-center justify-between p-2 rounded-lg bg-white/[0.02]">
                      <span className="text-slate-400">Model Context Budget</span>
                      <span className="font-mono text-xs text-cyan-400">
                        {((promptAnalysis.estimatedPromptTokens / currentModelConfig.contextWindow) * 100).toFixed(2)}% of {(currentModelConfig.contextWindow / 1024).toFixed(0)}k
                      </span>
                    </div>
                  </div>
                </div>

                {/* Additional Token Details if run */}
                {totalTokens > 0 && (
                  <div className="bg-[#0b0e16] border border-white/[0.06] rounded-xl p-4">
                    <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-cyan-400" />
                      Last Run Token Consumption
                    </h3>
                    <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                      <div className="p-2 rounded bg-white/[0.02]">
                        <span className="text-[10px] text-slate-500 block">Prompt</span>
                        <span className="text-slate-300 font-bold">{promptTokens}</span>
                      </div>
                      <div className="p-2 rounded bg-white/[0.02]">
                        <span className="text-[10px] text-slate-500 block">Completion</span>
                        <span className="text-purple-400 font-bold">{tokensGenerated}</span>
                      </div>
                      <div className="p-2 rounded bg-white/[0.02]">
                        <span className="text-[10px] text-slate-500 block">Total</span>
                        <span className="text-cyan-400 font-bold">{totalTokens}</span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="p-3 bg-purple-950/20 border border-purple-500/20 rounded-xl text-xs text-purple-300 leading-relaxed">
                  💡 <strong>Optimization Tip:</strong> For faster generation and lower token usage, specify the desired format explicitly (e.g. "Answer in 2 sentences" or "Return JSON").
                </div>
              </div>
            )}

            {/* Tab 3: API Code Preview (Python / cURL / Node) */}
            {activeTab === 'code' && (
              <div className="flex-1 flex flex-col overflow-hidden py-3">
                <div className="flex items-center justify-between pb-3">
                  <div className="flex gap-1.5">
                    <button
                      type="button"
                      onClick={() => setCodeLang('python')}
                      className={`text-xs px-2.5 py-1 rounded-md font-mono transition-smooth cursor-pointer ${
                        codeLang === 'python'
                          ? 'bg-purple-600/40 text-purple-200 border border-purple-500/50'
                          : 'bg-white/[0.03] text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Python
                    </button>
                    <button
                      type="button"
                      onClick={() => setCodeLang('curl')}
                      className={`text-xs px-2.5 py-1 rounded-md font-mono transition-smooth cursor-pointer ${
                        codeLang === 'curl'
                          ? 'bg-cyan-600/40 text-cyan-200 border border-cyan-500/50'
                          : 'bg-white/[0.03] text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      cURL
                    </button>
                    <button
                      type="button"
                      onClick={() => setCodeLang('node')}
                      className={`text-xs px-2.5 py-1 rounded-md font-mono transition-smooth cursor-pointer ${
                        codeLang === 'node'
                          ? 'bg-purple-600/40 text-purple-200 border border-purple-500/50'
                          : 'bg-white/[0.03] text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Node.js
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => {
                      const snippet =
                        codeLang === 'python'
                          ? generatePythonCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens)
                          : codeLang === 'curl'
                          ? generateCurlCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens)
                          : generateNodeCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens);
                      copyToClipboard(snippet, 'code');
                    }}
                    className="flex items-center gap-1 text-xs text-slate-300 hover:text-white bg-white/[0.05] hover:bg-white/[0.1] px-2 py-1 rounded font-mono border border-white/[0.06] transition-smooth cursor-pointer"
                  >
                    {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCode ? 'Copied' : 'Copy Code'}</span>
                  </button>
                </div>

                <div className="flex-1 bg-[#07090e] border border-white/[0.06] rounded-xl p-3.5 overflow-auto font-mono text-[11px] text-slate-300 leading-relaxed">
                  <pre>
                    {codeLang === 'python' && generatePythonCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens)}
                    {codeLang === 'curl' && generateCurlCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens)}
                    {codeLang === 'node' && generateNodeCode(apiKey, selectedModel, prompt, systemPrompt, temperature, maxTokens)}
                  </pre>
                </div>
              </div>
            )}

            {/* Response Footer Stats */}
            <div className="pt-3 border-t border-white/[0.06] flex items-center justify-between text-[11px] font-mono text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${isStreaming ? 'bg-cyan-400 animate-ping' : 'bg-slate-600'}`}></span>
                {isStreaming ? 'Receiving chunks' : 'Inference idle'}
              </span>
              <span>Model: {selectedModel}</span>
            </div>

          </div>

        </section>

      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.06] py-4 px-6 text-center text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span>INSIDE THE LLM Playground</span>
          <span>•</span>
          <span className="text-slate-600">Built with Groq LPUs & React</span>
        </div>
        <div className="flex items-center gap-4 text-slate-400 text-[11px]">
          <span>Dark Charcoal & Purple Cyan Palette</span>
        </div>
      </footer>

    </div>
  );
}
