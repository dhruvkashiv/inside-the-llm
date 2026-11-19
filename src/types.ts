export interface ModelConfig {
  id: string;
  name: string;
  description: string;
  contextWindow: number;
  maxOutputTokens: number;
  speed: string;
  category: 'ultra-fast' | 'reasoning' | 'balanced' | 'vision';
}

export const GROQ_MODELS: ModelConfig[] = [
  {
    id: 'llama-3.3-70b-versatile',
    name: 'Llama 3.3 70B Versatile',
    description: 'State of the art open model with exceptional reasoning & coding',
    contextWindow: 128000,
    maxOutputTokens: 32768,
    speed: '~280 t/s',
    category: 'reasoning'
  },
  {
    id: 'llama-3.1-8b-instant',
    name: 'Llama 3.1 8B Instant',
    description: 'Blazing fast throughput for quick tasks, low latency',
    contextWindow: 128000,
    maxOutputTokens: 8192,
    speed: '~850 t/s',
    category: 'ultra-fast'
  },
  {
    id: 'openai/gpt-oss-120b',
    name: 'GPT OSS 120B (Groq)',
    description: 'Flagship open weight model on Groq LPU inference engine',
    contextWindow: 65536,
    maxOutputTokens: 8192,
    speed: '~240 t/s',
    category: 'reasoning'
  },
  {
    id: 'mixtral-8x7b-32768',
    name: 'Mixtral 8x7B (MoE)',
    description: 'High quality Mixture-of-Experts with 32k context',
    contextWindow: 32768,
    maxOutputTokens: 4096,
    speed: '~500 t/s',
    category: 'balanced'
  },
  {
    id: 'gemma2-9b-it',
    name: 'Gemma 2 9B IT',
    description: "Google's powerful lightweight instruction model",
    contextWindow: 8192,
    maxOutputTokens: 4096,
    speed: '~600 t/s',
    category: 'ultra-fast'
  },
  {
    id: 'llama-3.2-11b-vision-preview',
    name: 'Llama 3.2 11B Vision',
    description: 'Multimodal capabilities with high-speed text generation',
    contextWindow: 128000,
    maxOutputTokens: 8192,
    speed: '~400 t/s',
    category: 'vision'
  }
];

export interface PromptMetrics {
  charCount: number;
  wordCount: number;
  estimatedPromptTokens: number;
  readingTimeSec: number;
  complexityScore: 'Simple' | 'Moderate' | 'Complex';
  hasCode: boolean;
  hasSystemDirectives: boolean;
  language: string;
}

export function analyzePrompt(text: string): PromptMetrics {
  const trimmed = text.trim();
  if (!trimmed) {
    return {
      charCount: 0,
      wordCount: 0,
      estimatedPromptTokens: 0,
      readingTimeSec: 0,
      complexityScore: 'Simple',
      hasCode: false,
      hasSystemDirectives: false,
      language: 'English'
    };
  }

  const charCount = text.length;
  const words = trimmed.split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  // Estimate tokens (~ 1 token per 3.8 chars or 0.75 words)
  const estimatedPromptTokens = Math.ceil(Math.max(charCount / 3.8, wordCount * 1.3));
  const readingTimeSec = Math.max(1, Math.round((wordCount / 200) * 60));

  const hasCode = /```|function|def |class |const |let |import |=>|public |SELECT /i.test(text);
  const hasSystemDirectives = /^(You are|Act as|System:|Instruction:|Role:)/im.test(text);

  let complexityScore: 'Simple' | 'Moderate' | 'Complex' = 'Simple';
  if (estimatedPromptTokens > 400 || (hasCode && estimatedPromptTokens > 150)) {
    complexityScore = 'Complex';
  } else if (estimatedPromptTokens > 100 || hasCode || hasSystemDirectives) {
    complexityScore = 'Moderate';
  }

  return {
    charCount,
    wordCount,
    estimatedPromptTokens,
    readingTimeSec,
    complexityScore,
    hasCode,
    hasSystemDirectives,
    language: 'English'
  };
}

export function generateCurlCode(apiKey: string, model: string, prompt: string, systemPrompt: string, temperature: number, maxTokens: number): string {
  const messages = [];
  if (systemPrompt.trim()) {
    messages.push({ role: 'system', content: systemPrompt.trim() });
  }
  messages.push({ role: 'user', content: prompt || 'Hello, how are you?' });

  const payload = {
    model,
    messages,
    temperature,
    max_tokens: maxTokens,
    stream: true
  };

  const keyDisplay = apiKey ? apiKey : 'YOUR_GROQ_API_KEY';

  return `curl -X POST "https://api.groq.com/openai/v1/chat/completions" \\
  -H "Authorization: Bearer ${keyDisplay}" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(payload, null, 2)}'`;
}

export function generatePythonCode(apiKey: string, model: string, prompt: string, systemPrompt: string, temperature: number, maxTokens: number): string {
  const keyDisplay = apiKey ? `"${apiKey}"` : 'os.environ.get("GROQ_API_KEY")';
  const sysLine = systemPrompt.trim() ? `        {"role": "system", "content": ${JSON.stringify(systemPrompt)}},\n` : '';

  return `import os
from groq import Groq

client = Groq(
    api_key=${keyDisplay}
)

stream = client.chat.completions.create(
    messages=[
${sysLine}        {"role": "user", "content": ${JSON.stringify(prompt || 'Hello, how are you?')}}
    ],
    model="${model}",
    temperature=${temperature},
    max_tokens=${maxTokens},
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
`;
}

export function generateNodeCode(apiKey: string, model: string, prompt: string, systemPrompt: string, temperature: number, maxTokens: number): string {
  const keyDisplay = apiKey ? `apiKey: "${apiKey}"` : 'apiKey: process.env.GROQ_API_KEY';
  const sysObj = systemPrompt.trim() ? `{ role: 'system', content: ${JSON.stringify(systemPrompt)} }, ` : '';

  return `import Groq from 'groq-sdk';

const groq = new Groq({ ${keyDisplay} });

async function main() {
  const stream = await groq.chat.completions.create({
    messages: [
      ${sysObj}{ role: 'user', content: ${JSON.stringify(prompt || 'Hello, how are you?')} }
    ],
    model: '${model}',
    temperature: ${temperature},
    max_tokens: ${maxTokens},
    stream: true,
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}

main();
`;
}
