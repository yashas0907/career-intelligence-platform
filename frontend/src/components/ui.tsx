import type { ReactNode } from 'react'
import { fmtPct, scoreColor } from '../lib/api'
import type { MatchAnalysis } from '../lib/types'

export function ScoreRing({ value, size = 132, label = 'MATCH' }: { value: number; size?: number; label?: string }) {
  const r = size / 2 - 9
  const c = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(1, value))
  const col = scoreColor(pct)
  return (
    <div className="score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="rgba(255,255,255,.09)" strokeWidth="9" fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={col}
          strokeWidth="9"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          style={{ transition: 'stroke-dashoffset .7s ease' }}
        />
      </svg>
      <div className="val" style={{ color: col }}>
        {fmtPct(pct)}
        <small>{label}</small>
      </div>
    </div>
  )
}

export function BreakdownList({ analysis }: { analysis: MatchAnalysis }) {
  const rows: Array<[string, string, number]> = [
    ['Skills', 'skills', analysis.breakdown.skills],
    ['Experience', 'experience', analysis.breakdown.experience],
    ['Projects', 'projects', analysis.breakdown.projects],
    ['Education', 'education', analysis.breakdown.education],
    ['Semantic Relevance', 'semantic', analysis.breakdown.semantic],
  ]
  return (
    <div>
      {rows.map(([label, key, val]) => (
        <div className="score-row" key={key}>
          <span className="lbl">{label}</span>
          <span className="wt">w {fmtPct(analysis.weights[key])}</span>
          <span className="pct" style={{ color: scoreColor(val) }}>{fmtPct(val)}</span>
        </div>
      ))}
    </div>
  )
}

export function Card({ title, sub, children, right }: { title?: ReactNode; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="card">
      {title && <h2>{title}{right}</h2>}
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  )
}

export function ErrorBox({ children, onDismiss }: { children: ReactNode; onDismiss?: () => void }) {
  return (
    <div className="err-box" role="alert">
      {children}
      {onDismiss && (
        <button className="nav-btn" style={{ float: 'right', padding: '0 6px' }} onClick={onDismiss} aria-label="dismiss">
          ✕
        </button>
      )}
    </div>
  )
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />
}
