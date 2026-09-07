import { useState } from 'react'
import { api, fmtPct } from '../lib/api'
import { Card, ErrorBox, ScoreRing, BreakdownList, Spinner } from '../components/ui'
import { SkillGroups } from '../components/SkillGroups'
import { ResumeFixer } from '../components/ResumeFixer'
import type { MatchAnalysis, ResumeProfile } from '../lib/types'

interface JobDraft {
  key: number
  title: string
  description: string
}

let nextKey = 1

export default function JobAnalyzer({
  resume,
  onAnalyzed,
  onAnalysisSelect,
}: {
  resume: ResumeProfile | null
  onAnalyzed: (results: MatchAnalysis[], ranking: MatchAnalysis[]) => void
  onAnalysisSelect: (a: MatchAnalysis) => void
}) {
  const [drafts, setDrafts] = useState<JobDraft[]>([{ key: nextKey++, title: '', description: '' }])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [liveResults, setLiveResults] = useState<MatchAnalysis[]>([])
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)

  function update(key: number, patch: Partial<JobDraft>) {
    setDrafts((ds) => ds.map((d) => (d.key === key ? { ...d, ...patch } : d)))
  }

  async function analyze() {
    if (!resume) return
    const valid = drafts.filter((d) => d.description.trim().length >= 30)
    if (!valid.length) {
      setError('Each job description needs at least 30 characters of text.')
      return
    }
    setBusy(true)
    setError(null)
    setLiveResults([])
    setProgress({ done: 0, total: valid.length })
    try {
      const collected: MatchAnalysis[] = []
      await api.analyzeJobsStream(
        resume.resume_id,
        valid.map((d) => ({ title: d.title.trim() || undefined, description: d.description.trim() })),
        (a) => {
          collected.push(a)
          setLiveResults([...collected])          // render as each lands
          setProgress({ done: collected.length, total: valid.length })
        },
      )
      onAnalyzed(collected, collected)
      if (collected[0]) onAnalysisSelect(collected[0])
      setProgress(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  if (!resume) {
    return <div className="empty"><span className="big">📥</span>Upload a resume first to analyze jobs.</div>
  }

  return (
    <div>
      {drafts.map((d, idx) => (
        <Card
          key={d.key}
          title={`Job description ${idx + 1}`}
          right={drafts.length > 1 ? (
            <button
              className="btn danger"
              style={{ padding: '4px 10px', fontSize: '.78rem', marginLeft: 'auto' }}
              onClick={() => setDrafts((ds) => ds.filter((x) => x.key !== d.key))}
            >
              Remove
            </button>
          ) : undefined}
        >
          <label className="field">
            <span>Job title (optional — auto-detected if blank)</span>
            <input
              type="text"
              placeholder="e.g. Machine Learning Intern"
              value={d.title}
              onChange={(e) => update(d.key, { title: e.target.value })}
            />
          </label>
          <label className="field" style={{ marginBottom: 0 }}>
            <span>Job description</span>
            <textarea
              placeholder="Paste the full job description here — including the requirements list…"
              value={d.description}
              onChange={(e) => update(d.key, { description: e.target.value })}
            />
          </label>
        </Card>
      ))}

      <div style={{ display: 'flex', gap: 10, margin: '16px 0 30px', flexWrap: 'wrap' }}>
        <button
          className="btn"
          onClick={() => setDrafts((ds) => [...ds, { key: nextKey++, title: '', description: '' }])}
          disabled={drafts.length >= 10 || busy}
        >
          + Add another job
        </button>
        <button className="btn primary" onClick={analyze} disabled={busy}>
          {busy ? <Spinner /> : '▶'} Analyze match
        </button>
      </div>

      {busy && progress && (
        <Card title="Analyzing…" sub={`Results stream in live — ${progress.done}/${progress.total} done`}>
          <div className="meter" style={{ marginBottom: 12 }}>
            <span style={{ width: `${(progress.done / Math.max(1, progress.total)) * 100}%`, background: 'var(--accent)' }} />
          </div>
          {liveResults.map((a, i) => (
            <div className="score-row" key={i}>
              <span className="lbl">{a.job_title}</span>
              <span className="pct" style={{ color: a.overall_score >= 0.75 ? 'var(--accent-2)' : a.overall_score >= 0.5 ? 'var(--warn)' : 'var(--danger)' }}>
                {fmtPct(a.overall_score)}
              </span>
            </div>
          ))}
        </Card>
      )}

      {error && <ErrorBox onDismiss={() => setError(null)}>{error}</ErrorBox>}
    </div>
  )
}

export function AnalysisDetail({ a }: { a: MatchAnalysis }) {
  return (
    <div>
      <div className="grid-2">
        <Card title={`Match analysis — ${a.job_title}`} sub={`Explanation via ${a.explanation_method} pipeline · every number computed deterministically`}>
          <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap' }}>
            <ScoreRing value={a.overall_score} />
            <div style={{ flex: 1, minWidth: 220 }}>
              <BreakdownList analysis={a} />
            </div>
          </div>
        </Card>

        <Card title="Why this score" sub="Evidence-backed explanation">
          <pre className="explain">{a.explanation}</pre>
        </Card>
      </div>

      <div style={{ marginTop: 16 }}>
        <Card title="Skill comparison" sub="Every skill classified with the reason and evidence">
          <SkillGroups comp={a.skill_comparison} />
        </Card>
      </div>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <Card title="ATS-oriented review" sub={a.ats.disclaimer}>
          {Object.entries(a.ats.subscores).map(([name, val]) => (
            <div className="ats-bar" key={name}>
              <span className="n">{name}</span>
              <div className="meter">
                <span style={{ width: `${Math.round(val * 100)}%`, background: val >= 0.7 ? 'var(--accent-2)' : val >= 0.4 ? 'var(--warn)' : 'var(--danger)' }} />
              </div>
              <span style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{fmtPct(val)}</span>
            </div>
          ))}
          {a.ats.recommendations.length > 0 && (
            <>
              <h3 style={{ fontSize: '.88rem', margin: '14px 0 6px' }}>ATS recommendations</h3>
              <ul className="section-list">
                {a.ats.recommendations.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </>
          )}
        </Card>

        <Card title="Improvement plan" sub="Generated from the actual gaps — no invented experience, ever">
          {a.recommendations.map((r, i) => (
            <div className="rec-item" key={i}>
              <div className="ic">{r.type === 'learn' ? '📚' : r.type === 'build' ? '🛠️' : '✍️'}</div>
              <div>
                <b>{r.title}</b>
                <span className={`pill ${r.priority === 'high' ? 'red' : r.priority === 'medium' ? 'amber' : 'grey'}`} style={{ marginLeft: 8, padding: '1px 8px', fontSize: '.64rem' }}>
                  {r.priority}
                </span>
                <p>{r.detail}</p>
              </div>
            </div>
          ))}
        </Card>
      </div>

      <div style={{ marginTop: 16 }}>
        <ResumeFixer resumeId={a.resume_id} jobId={a.job_id} />
      </div>
    </div>
  )
}
