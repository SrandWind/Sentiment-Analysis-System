import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// Types
export interface InferRequest {
  text: string
  model_variant?: 'base' | 'lora_merged' | 'gguf4bit'
  preset?: 'quick' | 'standard' | 'deep'
  max_tokens?: number
  temperature?: number
  top_p?: number
  repeat_penalty?: number
}

export interface InferResponse {
  text: string
  output: string
  scores: Record<string, number>
  raw_intensity_scores?: Record<string, number>
  target_scores?: Record<string, number>
  primary_emotion: string
  confidence: number
  cot: Record<string, string>
  json_parse_ok: boolean
  cot_complete?: boolean
  latency_ms: number
  vad_dimensions?: {
    valence: number
    arousal: number
    dominance: number
  }
  emotion_cause?: string
  uncertainty_level?: 'low' | 'medium' | 'high'
}

export interface StreamChunk {
  delta: string
  done: boolean
  output?: string
  latency_ms?: number
  error?: string
  retry?: boolean
  retry_count?: number
  error_msg?: string
  scores?: Record<string, number>
  raw_intensity_scores?: Record<string, number>
  target_scores?: Record<string, number>
  primary_emotion?: string
  confidence?: number
  cot?: Record<string, string>
  json_parse_ok?: boolean
  cot_complete?: boolean
  vad_dimensions?: {
    valence: number
    arousal: number
    dominance: number
  }
  emotion_cause?: string
  uncertainty_level?: 'low' | 'medium' | 'high'
  risk_warning?: string
}

export interface BatchRequest {
  texts: string[]
  model_variant?: 'base' | 'lora_merged' | 'gguf4bit'
  use_quick_preset?: boolean
}

export interface BatchItem {
  id: string
  text: string
  success: boolean
  result?: InferResponse
  error?: string
}

export interface BatchResponse {
  total: number
  success: number
  failed: number
  results: BatchItem[]
}

export interface BatchProgress {
  status: string
  total: number
  processed: number
  progress_percent: number
  current_text?: string
  message?: string
}

export interface BatchStartResponse {
  batch_id: string
  total: number
}

export interface BatchResultsResponse {
  status: string
  total: number
  success: number
  failed: number
  results: BatchItem[]
}

export interface MetricsResponse {
  model_variant: string
  emotion_macro_mae: number
  emotion_macro_mse: number
  emotion_per_dim_mae: Record<string, number>
  emotion_per_dim_mse: Record<string, number>
  primary_cls_accuracy: number
  primary_cls_macro_f1: number
  primary_cls_per_class_f1: Record<string, number>
  primary_cls_confusion_matrix?: {
    matrix: number[][]
    labels: string[]
  }
  json_parse_rate: number
  cot7_complete_rate: number
}

export interface CompareResponse {
  models: MetricsResponse[]
  comparison_table: string
}

// API methods
export const sentimentApi = {
  async infer(request: InferRequest): Promise<InferResponse> {
    // Non-streaming requests need longer timeout for CoT inference
    const response = await api.post<InferResponse>('/infer', request, {
      timeout: 120000, // 120 seconds for complete inference
    })
    return response.data
  },

  async inferStream(request: InferRequest): Promise<AsyncGenerator<StreamChunk>> {
    const response = await fetch('/api/infer/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || 'Streaming inference failed')
    }

    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('ReadableStream not supported')
    }

    // Use TextDecoder with stream: true to handle multi-byte UTF-8 characters
    const textDecoder = new TextDecoder('utf-8')
    let buffer = ''

    return (async function* () {
      let done = false
      while (!done) {
        const { done: readerDone, value } = await reader.read()
        done = readerDone

        if (value) {
          // Decode chunk as UTF-8, handling multi-byte characters correctly
          const text = textDecoder.decode(value, { stream: !done })
          buffer += text
        }

        // Parse SSE messages
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine.startsWith('data: ')) {
            const data = trimmedLine.slice(6)
            if (data) {
              try {
                const chunk = JSON.parse(data)
                yield chunk
                // Continue processing even after done=true to receive final_chunk from backend
                // Do NOT return here - let the loop finish naturally
              } catch (e) {
                // May be incomplete JSON due to streaming, skip and continue
                console.debug('Waiting for complete JSON...')
              }
            }
          }
        }
      }
      
      // Process any remaining data in buffer after stream ends
      // This handles the final_chunk sent by backend in finally block
      if (buffer.trim()) {
        const lines = buffer.split('\n')
        for (const line of lines) {
          const trimmedLine = line.trim()
          if (trimmedLine.startsWith('data: ')) {
            const data = trimmedLine.slice(6)
            if (data) {
              try {
                const chunk = JSON.parse(data)
                yield chunk
              } catch (e) {
                console.debug('Error parsing final chunk:', e)
              }
            }
          }
        }
      }
    })()
  },

  async batchInfer(request: BatchRequest): Promise<BatchResponse> {
    const response = await api.post<BatchResponse>('/batch', request, {
      timeout: 300000,
    })
    return response.data
  },

  async startBatchInfer(request: BatchRequest): Promise<BatchStartResponse> {
    const response = await api.post<BatchStartResponse>('/batch/start', request)
    return response.data
  },

  async getBatchProgress(batchId: string): Promise<BatchProgress> {
    const response = await api.get<BatchProgress>(`/batch/progress/${batchId}`)
    return response.data
  },

  async getBatchResults(batchId: string): Promise<BatchResultsResponse> {
    const response = await api.get<BatchResultsResponse>(`/batch/results/${batchId}`)
    return response.data
  },

  async getMetrics(modelVariant: string): Promise<MetricsResponse> {
    const response = await api.get<MetricsResponse>(`/metrics/${modelVariant}`)
    return response.data
  },

  async compare(modelVariants?: string[]): Promise<CompareResponse> {
    const response = await api.post<CompareResponse>('/compare', { model_variants: modelVariants })
    return response.data
  },

  async health(): Promise<{ status: string; lmstudio_connected: boolean; database_connected: boolean }> {
    const response = await api.get('/health')
    return response.data
  },
}

export default api
