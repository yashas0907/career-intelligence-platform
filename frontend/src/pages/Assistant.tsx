import { useRef, useState } from 'react'
import { api } from '../lib/api'
import { Card, Spinner } from '../components/ui'
import type { ChatSource } from '../lib/types'

interface Msg {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
  method?: string
  streaming?: boolean
}

const SUGGESTIONS = [
  'What skills am I missing for this role?',
  'Which of my projects are most relevant, and why?',
  'What should I learn first to improve my fit?',
  'Why did this job get the score it did?',
]

export default function Assistant({
  analysisId,
  jobTitle,
}: {
  analysisId: string | null
  jobTitle: string | null
}) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  const scrollDown = () =>
    requestAnimationFrame(() => logRef.current?.scrollTo({ top: 1e9, behavior: 'smooth' }))

  function send(q: string) {
    if (!q.trim() || busy) return
    setBusy(true)
    setInput('')
    setMsgs((m) => [...m, { role: 'user', content: q }, { role: 'assistant', content: '', streaming: true }])
    scrollDown()

    api.chatStream(q, analysisId, {
      onMeta: (sources) => {
        setMsgs((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.streaming) copy[copy.length - 1] = { ...last, sources }
          return copy
        })
      },
      onToken: (text) => {
        setMsgs((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.streaming) copy[copy.length - 1] = { ...last, content: last.content + text }
          return copy
        })
        scrollDown()
      },
      onDone: (method) => {
        setMsgs((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.streaming) copy[copy.length - 1] = { ...last, method, streaming: false }
          return copy
        })
        setBusy(false)
        scrollDown()
      },
      onError: (msg) => {
        setMsgs((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last?.streaming) copy[copy.length - 1] = { ...last, content: msg, streaming: false }
          return copy
        })
        setBusy(false)
      },
    })
  }

  return (
    <Card
      title="Career assistant"
      sub={`Answers stream live and are grounded in your uploaded documents via retrieval${jobTitle ? ` · context: ${jobTitle}` : ''}`}
    >
      <div className="chat-wrap">
        <div className="chat-log" ref={logRef}>
          {msgs.length === 0 && (
            <div className="empty" style={{ margin: 'auto 0' }}>
              <span className="big">💬</span>
              Ask anything about your resume or the analyzed jobs.<br />
              Every answer cites the document chunks it was built from.
            </div>
          )}
          {msgs.map((m, i) => (
            <div className={`msg ${m.role}`} key={i}>
              {m.content || (m.streaming ? <Spinner /> : null)}
              {m.streaming && m.content ? <span className="cursor" aria-hidden>▍</span> : null}
              {m.sources && m.sources.length > 0 && !m.streaming ? (
                <div className="srcs">
                  {m.sources.slice(0, 3).map((s, j) => (
                    <div className="src" key={j}>
                      <code>{s.source.toUpperCase()}</code>
                      <span style={{ opacity: .8 }}>{s.excerpt.slice(0, 110)}…</span>
                      <span style={{ opacity: .55 }}>(sim {s.similarity})</span>
                    </div>
                  ))}
                </div>
              ) : null}
              {m.method === 'rag+extractive' && !m.streaming ? (
                <div className="src" style={{ marginTop: 8 }}><code>offline mode · quoted evidence</code></div>
              ) : null}
            </div>
          ))}
          {busy && !msgs[msgs.length - 1]?.content && (
            <div className="msg assistant"><Spinner />&nbsp;Retrieving evidence…</div>
          )}
        </div>

        <div className="chat-input">
          <textarea
            placeholder="Ask about your skills, gaps, projects…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send(input)
              }
            }}
          />
          <button className="btn primary" disabled={busy || !input.trim()} onClick={() => send(input)}>
            {busy ? <Spinner /> : 'Send'}
          </button>
        </div>

        <div className="suggest">
          {SUGGESTIONS.map((s) => (
            <button key={s} onClick={() => send(s)} disabled={busy}>{s}</button>
          ))}
        </div>
      </div>
    </Card>
  )
}
