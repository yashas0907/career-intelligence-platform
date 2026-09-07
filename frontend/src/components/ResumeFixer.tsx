import { useState } from 'react'
import { api } from '../lib/api'
import { Card, ErrorBox, Spinner } from './ui'

interface FixState {
  fixed_text: string
  changes: Array<{ section: string; change: string }>
  llm_used: boolean
}

/** "Fix my resume" panel: runs the fixer against the active job (or none) and
 *  shows the improved resume inline with a change log + downloads. */
export function ResumeFixer({ resumeId, jobId }: { resumeId: string; jobId: string | null }) {
  const [result, setResult] = useState<FixState | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function runFix() {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const r = await api.fixResume(resumeId, jobId)
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Fix failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card
      title="Fix my resume"
      sub="Restructures your resume in-place: clean sections, ATS-safe formatting, job-keyword aligned skills, stronger bullets. Nothing is invented — only your own stated facts are reorganized."
    >
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 14 }}>
        <button className="btn primary" onClick={runFix} disabled={busy}>
          {busy ? <><Spinner /> Fixing…</> : '🔧 Fix my resume'}
        </button>
        {result && (
          <>
            <a className="btn" href={api.fixDownloadUrl(resumeId, 'docx', jobId)}>⬇ Download .docx</a>
            <a className="btn" href={api.fixDownloadUrl(resumeId, 'txt', jobId)}>⬇ Download .txt</a>
          </>
        )}
      </div>

      {error && <ErrorBox onDismiss={() => setError(null)}>{error}</ErrorBox>}

      {result && (
        <>
          <div className="warn-box" style={{ marginBottom: 14 }}>
            {result.llm_used
              ? 'Bullets were strengthened with AI — strictly limited to facts already in your resume (no-fabrication guard active).'
              : 'Deterministic mode: structure, formatting, ordering and keywords fixed. Add a free Gemini key (see docs) to also rewrite weak bullets with AI.'}
          </div>

          <h3 style={{ fontSize: '.92rem', margin: '0 0 8px' }}>What changed ({result.changes.length})</h3>
          <ul className="section-list" style={{ marginBottom: 18 }}>
            {result.changes.map((c, i) => (
              <li key={i}>
                <span className="pill blue" style={{ marginRight: 8, padding: '2px 8px', fontSize: '.66rem' }}>{c.section}</span>
                {c.change}
              </li>
            ))}
          </ul>

          <h3 style={{ fontSize: '.92rem', margin: '0 0 8px' }}>Fixed resume</h3>
          <pre className="explain" style={{ maxHeight: 420, overflowY: 'auto' }}>{result.fixed_text}</pre>
        </>
      )}
    </Card>
  )
}
