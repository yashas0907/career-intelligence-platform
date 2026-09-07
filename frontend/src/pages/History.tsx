import { Card } from '../components/ui'
import { fmtPct } from '../lib/api'
import type { HistoryItem } from '../lib/types'

export default function History({
  items,
  onOpen,
}: {
  items: HistoryItem[]
  onOpen: (a: HistoryItem) => void
}) {
  if (items.length === 0) {
    return <div className="empty"><span className="big">🕘</span>No analyses yet — they'll appear here with their full breakdowns.</div>
  }
  return (
    <Card title="Analysis history" sub="Every analysis is persisted with its full breakdown, so scores stay explainable later.">
      {items.map((h) => (
        <div className="history-item" key={h.analysis_id} onClick={() => onOpen(h)}>
          <div className="l">{h.job_title}</div>
          <div className="r">
            <time>{new Date(h.created_at).toLocaleString()}</time>
            <span
              className="pill"
              style={{
                color: h.overall_score >= 0.7 ? '#6ee7b7' : h.overall_score >= 0.5 ? '#fcd34d' : '#fda4a4',
                borderColor: 'var(--card-edge)',
              }}
            >
              {fmtPct(h.overall_score)}
            </span>
          </div>
        </div>
      ))}
    </Card>
  )
}
