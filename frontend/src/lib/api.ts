/** Typed API client. Uses the Vite dev proxy; override with VITE_API_URL for other setups. */

const BASE = import.meta.env.VITE_API_URL ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers:
      init?.body && !(init.body instanceof FormData)
        ? { ...init.headers, 'Content-Type': 'application/json' }
        : init?.headers,
  })
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch { /* keep default */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<import('./types').HealthInfo>('/health'),

  uploadResume: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<import('./types').ResumeProfile>('/resume/upload', { method: 'POST', body: fd })
  },

  getResume: (id: string) => request<import('./types').ResumeProfile>(`/resume/${id}`),

  resumeHistory: (id: string) =>
    request<{ resume_id: string; analyses: import('./types').HistoryItem[] }>(`/resume/${id}/history`),

  analyzeJobs: (resumeId: string, jobs: Array<{ title?: string; description: string }>) =>
    request<import('./types').MatchAnalysis[]>('/jobs/analyze', {
      method: 'POST',
      body: JSON.stringify({ resume_id: resumeId, jobs }),
    }),

  /** SSE: emits each analysis the moment it's computed. */
  analyzeJobsStream: (
    resumeId: string,
    jobs: Array<{ title?: string; description: string }>,
    onResult: (a: import('./types').MatchAnalysis, index: number, total: number) => void,
  ): Promise<void> =>
    new Promise((resolve, reject) => {
      void (async () => {
        try {
          const res = await fetch(`${BASE}/jobs/analyze/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_id: resumeId, jobs }),
          })
          if (!res.ok || !res.body) {
            let detail = `Analysis failed (${res.status})`
            try { detail = (await res.json()).detail ?? detail } catch { /* keep */ }
            throw new Error(detail)
          }
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buf = ''
          for (;;) {
            const { done, value } = await reader.read()
            if (done) break
            buf += decoder.decode(value, { stream: true })
            const events = buf.split('\n\n')
            buf = events.pop() ?? ''
            for (const evt of events) {
              const line = evt.replace(/^data:\s*/, '')
              if (!line || line === '[DONE]') continue
              try {
                const parsed = JSON.parse(line)
                if (parsed.type === 'result') onResult(parsed.analysis, parsed.index, parsed.total)
              } catch { /* ignore malformed */ }
            }
          }
          resolve()
        } catch (e) {
          reject(e)
        }
      })()
    }),

  getAnalysis: (id: string) => request<import('./types').MatchAnalysis>(`/analysis/${id}`),

  recommendations: (resumeId: string) =>
    request<import('./types').RankedJob[]>(`/recommendations?resume_id=${encodeURIComponent(resumeId)}`),

  chat: (question: string, analysisId?: string | null) =>
    request<import('./types').ChatResponse>('/chat', {
      method: 'POST',
      body: JSON.stringify({ question, analysis_id: analysisId ?? undefined }),
    }),

  /** SSE chat: onMeta (sources) first, then streamed answer tokens, then done(method). */
  chatStream: (
    question: string,
    analysisId: string | null,
    handlers: {
      onMeta: (sources: import('./types').ChatSource[]) => void
      onToken: (text: string) => void
      onDone: (method: string) => void
      onError?: (msg: string) => void
    },
  ): void => {
    void (async () => {
      try {
        const res = await fetch(`${BASE}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, analysis_id: analysisId ?? undefined }),
        })
        if (!res.ok || !res.body) {
          let detail = `Chat failed (${res.status})`
          try { detail = (await res.json()).detail ?? detail } catch { /* keep */ }
          handlers.onError?.(detail)
          return
        }
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })
          const events = buf.split('\n\n')
          buf = events.pop() ?? ''
          for (const evt of events) {
            const line = evt.replace(/^data:\s*/, '')
            if (!line || line === '[DONE]') continue
            try {
              const parsed = JSON.parse(line)
              if (parsed.type === 'meta') handlers.onMeta(parsed.sources ?? [])
              else if (parsed.type === 'answer') handlers.onToken(parsed.text)
              else if (parsed.type === 'done') handlers.onDone(parsed.method)
              else if (parsed.type === 'error') handlers.onError?.(parsed.detail)
            } catch { /* ignore malformed */ }
          }
        }
      } catch (e) {
        handlers.onError?.(e instanceof Error ? e.message : 'Chat stream failed')
      }
    })()
  },

  demoData: () =>
    request<{ resume_text: string; jobs: Array<{ title: string; description: string }> }>('/demo/sample'),

  seedDemo: () => request<import('./types').ResumeProfile>('/demo/seed', { method: 'POST' }),

  fixResume: (resumeId: string, jobId?: string | null) =>
    request<{ resume_id: string; fixed_text: string; changes: Array<{ section: string; change: string }>; llm_used: boolean }>(
      `/fix/${resumeId}${jobId ? `?job_id=${jobId}` : ''}`,
      { method: 'POST' },
    ),

  fixDownloadUrl: (resumeId: string, format: 'txt' | 'docx', jobId?: string | null) =>
    `${BASE}/fix/${resumeId}/download?format=${format}${jobId ? `&job_id=${jobId}` : ''}`,
}

export function fmtPct(v: number | undefined | null): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${Math.round(v * 100)}%`
}

export function scoreColor(v: number): string {
  if (v >= 0.75) return 'var(--accent-2)'
  if (v >= 0.5) return 'var(--warn)'
  return 'var(--danger)'
}
