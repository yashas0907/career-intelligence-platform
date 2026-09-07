import { ScoreRing, Card, Spinner } from '../components/ui'
import { fmtPct } from '../lib/api'
import type { AppStore, PageId } from '../lib/store'

export default function Dashboard({
  store,
  go,
  onSeedDemo,
  demoBusy,
}: {
  store: AppStore
  go: (p: PageId) => void
  onSeedDemo: () => void
  demoBusy: boolean
}) {
  const hasResume = !!store.resume
  const nJobs = store.analyses.length
  const best = store.ranking[0] ?? store.analyses[0] ?? null

  return (
    <div>
      <div className="hero">
        <h1>
          Understand exactly <em>why</em> you match — <em>why</em> you don't.
        </h1>
        <p>
          Upload your resume, paste job descriptions, and get an explainable compatibility
          breakdown — deterministic scoring, skill-gap analysis, an ATS-oriented review and a
          career assistant grounded in your own documents.
        </p>
        <div className="hero-actions">
          <button className="btn primary" onClick={() => go(hasResume ? 'jobs' : 'upload')}>
            {hasResume ? 'Analyze a job →' : 'Upload your resume →'}
          </button>
          <button className="btn" onClick={onSeedDemo} disabled={demoBusy}>
            {demoBusy ? <><Spinner /> Loading demo…</> : '⚡ Try the live demo'}
          </button>
          {hasResume && (
            <button className="btn ghost" onClick={() => go('assistant')}>Ask the AI assistant</button>
          )}
        </div>
        <div className="steps">
          <div className="step-card"><span className="n">1</span><h3>Upload resume</h3><p>PDF, DOCX or TXT. Parsed into a structured profile with evidence for every extracted skill.</p></div>
          <div className="step-card"><span className="n">2</span><h3>Add jobs</h3><p>Paste up to 10 job descriptions. Requirements are extracted and normalized against a skill taxonomy.</p></div>
          <div className="step-card"><span className="n">3</span><h3>See the breakdown</h3><p>Skills, experience, projects, education and semantic similarity — each scored and weighted transparently. Results stream in live.</p></div>
          <div className="step-card"><span className="n">4</span><h3>Act on it</h3><p>Ranked job fit, missing-skills learning plan, ATS-oriented checks, and a RAG assistant with citations.</p></div>
        </div>
      </div>

      <div className="grid-3" style={{ marginTop: 26 }}>
        <Card title="Resume">
          {hasResume ? (
            <>
              <p className="sub" style={{ margin: '0 0 10px' }}>
                {store.resume!.profile.name ?? 'Candidate'} — {store.resume!.filename}
              </p>
              <div className="kv">
                <dt>Skills detected</dt><dd>{Object.keys(store.resume!.profile.skills ?? {}).length}</dd>
                <dt>Experience</dt><dd>{store.resume!.profile.experience?.length ?? 0} role(s)</dd>
                <dt>Projects</dt><dd>{store.resume!.profile.projects?.length ?? 0}</dd>
              </div>
              <button className="btn" style={{ marginTop: 14 }} onClick={() => go('upload')}>View profile</button>
            </>
          ) : (
            <p className="sub" style={{ margin: 0 }}>No resume yet. <button className="btn" style={{ padding: '6px 12px', marginLeft: 8 }} onClick={() => go('upload')}>Upload</button></p>
          )}
        </Card>

        <Card title="Jobs analyzed">
          {nJobs > 0 ? (
            <>
              <p className="sub" style={{ margin: '0 0 10px' }}>{nJobs} role(s) analyzed against your profile</p>
              {store.ranking.slice(0, 3).map((r) => (
                <div className="score-row" key={r.job_id}>
                  <span className="lbl">{r.title}</span>
                  <span className="pct">{fmtPct(r.overall_score)}</span>
                </div>
              ))}
              <button className="btn" style={{ marginTop: 14 }} onClick={() => go('ranking')}>Full ranking</button>
            </>
          ) : (
            <p className="sub" style={{ margin: 0 }}>No jobs yet. <button className="btn" style={{ padding: '6px 12px', marginLeft: 8 }} disabled={!hasResume} onClick={() => go('jobs')}>Add jobs</button></p>
          )}
        </Card>

        <Card title="Best fit so far">
          {best ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
              <ScoreRing value={best.overall_score} size={104} />
              <div>
                <div style={{ fontWeight: 650, marginBottom: 4 }}>{best.title}</div>
                {best.reasons?.[0]}
              </div>
            </div>
          ) : (
            <p className="sub" style={{ margin: 0 }}>Analyze jobs to see your best-fit role.</p>
          )}
        </Card>
      </div>

      <div className="footer-note">
        AI Career Intelligence Platform — deterministic scoring you can audit (see docs/scoring.md),
        LLMs only where they add value. Backend status: {store.health?.status ?? '…'} ·
        embeddings: {store.health?.embedding_backend ?? '…'} · LLM: {store.health?.llm_available ? 'enabled' : 'offline mode (heuristic explanations)'}.
      </div>
    </div>
  )
}
