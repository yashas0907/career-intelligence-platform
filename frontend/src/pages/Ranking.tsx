import { Card } from '../components/ui'
import { api, fmtPct } from '../lib/api'
import { useEffect, useState } from 'react'
import type { MatchAnalysis, RankedJob } from '../lib/types'

export default function Ranking({
  resume,
  ranking,
  analyses,
  onAnalysisSelect,
}: {
  resume: MatchAnalysis extends never ? never : { resume_id: string } | null
  ranking: RankedJob[]
  analyses: MatchAnalysis[]
  onAnalysisSelect: (a: MatchAnalysis) => void
}) {
  const [localRanking, setLocalRanking] = useState<RankedJob[]>(ranking)

  useEffect(() => {
    if (!resume) return
    api.recommendations(resume.resume_id).then(setLocalRanking).catch(() => setLocalRanking(ranking))
  }, [resume, ranking])

  if (!resume) {
    return <div className="empty"><span className="big">🏆</span>Upload a resume first.</div>
  }

  if (localRanking.length === 0) {
    return <div className="empty"><span className="big">🏆</span>Analyze some jobs to see which fits you best.</div>
  }

  const byId = new Map(analyses.map((a) => [a.job_id, a]))

  return (
    <Card title="Job fit ranking" sub="Ranked by computed overall score; ties broken by skill match. Click a row for the full analysis.">
      {localRanking.map((r) => (
        <div
          className="rank-row"
          key={r.job_id}
          style={{ cursor: byId.has(r.job_id) ? 'pointer' : 'default' }}
          onClick={() => {
            const found = byId.get(r.job_id)
            if (found) onAnalysisSelect(found)
          }}
        >
          <div className="pos">#{r.rank}</div>
          <div className="mid">
            <div className="t">{r.title}</div>
            {r.reasons.map((why, i) => <div className="why" key={i}>{why}</div>)}
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontWeight: 800, fontSize: '1.15rem' }}>{fmtPct(r.overall_score)}</div>
          </div>
        </div>
      ))}
    </Card>
  )
}
