import { useEffect, useRef, useState } from 'react'
import { api } from './lib/api'
import { clearSession, loadSession, saveSession, type StoredSession } from './lib/session'
import { emptyStore, type AppStore, type PageId } from './lib/store'
import type { HealthInfo, HistoryItem, MatchAnalysis, ResumeProfile } from './lib/types'
import Dashboard from './pages/Dashboard'
import UploadResume from './pages/UploadResume'
import JobAnalyzer, { AnalysisDetail } from './pages/JobAnalyzer'
import Ranking from './pages/Ranking'
import Assistant from './pages/Assistant'
import History from './pages/History'

const NAV: Array<{ id: PageId; label: string }> = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'upload', label: 'Resume' },
  { id: 'jobs', label: 'Analyze Jobs' },
  { id: 'results', label: 'Results' },
  { id: 'ranking', label: 'Ranking' },
  { id: 'assistant', label: 'Assistant' },
]

export default function App() {
  const [store, setStore] = useState<AppStore>(emptyStore)
  const [page, setPage] = useState<PageId>('dashboard')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [booting, setBooting] = useState(true)
  const [demoBusy, setDemoBusy] = useState(false)
  const sess = useRef<StoredSession>(loadSession())

  // Boot: health check + restore session from localStorage
  useEffect(() => {
    void (async () => {
      api.health()
        .then((h: HealthInfo) => setStore((s) => ({ ...s, health: h })))
        .catch(() => {})
      const s = sess.current
      if (s.resumeId) {
        try {
          const resume = await api.getResume(s.resumeId)
          setStore((st) => ({ ...st, resume }))
          // restore last analysis if it still exists
          const ids = s.analysisIds.length ? s.analysisIds : s.activeAnalysisId ? [s.activeAnalysisId] : []
          const restored: MatchAnalysis[] = []
          for (const id of ids) {
            try { restored.push(await api.getAnalysis(id)) } catch { /* deleted */ }
          }
          if (restored.length) {
            setStore((st) => ({
              ...st,
              analyses: restored,
              activeAnalysis: restored.find((a) => a.analysis_id === s.activeAnalysisId) ?? restored[0],
            }))
          }
        } catch {
          // stale session (DB cleared) — forget it
          clearSession()
          sess.current = { resumeId: null, activeAnalysisId: null, analysisIds: [] }
        }
      }
      setBooting(false)
    })()
  }, [])

  // Persist session on meaningful changes
  useEffect(() => {
    sess.current = {
      resumeId: store.resume?.resume_id ?? null,
      activeAnalysisId: store.activeAnalysis?.analysis_id ?? null,
      analysisIds: store.analyses.map((a) => a.analysis_id),
    }
    saveSession(sess.current)
  }, [store.resume, store.activeAnalysis, store.analyses])

  // History follows the active resume
  useEffect(() => {
    if (store.resume) {
      api.resumeHistory(store.resume.resume_id)
        .then((r) => setHistory(r.analyses))
        .catch(() => {})
    } else {
      setHistory([])
    }
  }, [store.resume, store.analyses])

  const go = (p: PageId) => setPage(p)

  const onUploaded = (r: ResumeProfile) => {
    setStore((s) => ({ ...s, resume: r, error: null, analyses: [], activeAnalysis: null }))
    setPage('jobs')
  }

  const seedDemo = async () => {
    setDemoBusy(true)
    try {
      const demo = await api.seedDemo()
      onUploaded(demo)
    } catch {
      setStore((s) => ({ ...s, error: 'Could not load demo resume.' }))
    } finally {
      setDemoBusy(false)
    }
  }

  const startFresh = () => {
    clearSession()
    sess.current = { resumeId: null, activeAnalysisId: null, analysisIds: [] }
    setStore(emptyStore)
    setPage('dashboard')
  }

  const openHistoryItem = async (h: HistoryItem) => {
    try {
      const a = await api.getAnalysis(h.analysis_id)
      setStore((s) => ({ ...s, activeAnalysis: a, analyses: s.analyses.some((x) => x.analysis_id === a.analysis_id) ? s.analyses : [...s.analyses, a] }))
      setPage('results')
    } catch { /* stale */ }
  }

  const hasData = !!store.resume
  const navDisabled: Partial<Record<PageId, boolean>> = {
    jobs: !hasData,
    results: !store.activeAnalysis,
    ranking: !hasData,
    assistant: !hasData,
  }

  if (booting) {
    return (
      <main className="page">
        <div className="loading-pane">Restoring your session…</div>
      </main>
    )
  }

  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <div className="brand">
            <img src="/favicon.svg" alt="logo" />
            Career&nbsp;Intelligence
            <span className="tag">AI · Explainable</span>
          </div>
          <nav className="header-nav">
            {NAV.map((n) => (
              <button
                key={n.id}
                className={`nav-btn ${page === n.id ? 'active' : ''}`}
                disabled={navDisabled[n.id]}
                onClick={() => go(n.id)}
              >
                {n.label}
              </button>
            ))}
            {hasData && (
              <button className="nav-btn" onClick={startFresh} title="Clear session and start over">↺ New</button>
            )}
          </nav>
        </div>
      </header>

      <main className="page">
        {store.error && <div className="err-box">{store.error}</div>}

        {page === 'dashboard' && <Dashboard store={store} go={go} onSeedDemo={seedDemo} demoBusy={demoBusy} />}

        {page === 'upload' && (
          <UploadResume resume={store.resume} onUploaded={onUploaded} />
        )}

        {page === 'jobs' && (
          <JobAnalyzer
            resume={store.resume}
            onAnalyzed={(results) => {
              setStore((s) => ({ ...s, analyses: results, activeAnalysis: results[0] ?? null, error: null }))
              setPage('results')
            }}
            onAnalysisSelect={(a) => setStore((s) => ({ ...s, activeAnalysis: a }))}
          />
        )}

        {page === 'results' && (
          store.activeAnalysis ? (
            <>
              {store.analyses.length > 1 && (
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
                  {store.analyses.map((a) => (
                    <button
                      key={a.analysis_id}
                      className={`btn ${a.analysis_id === store.activeAnalysis?.analysis_id ? 'primary' : ''}`}
                      style={{ padding: '7px 13px', fontSize: '.84rem' }}
                      onClick={() => setStore((s) => ({ ...s, activeAnalysis: a }))}
                    >
                      {a.job_title}
                    </button>
                  ))}
                </div>
              )}
              <AnalysisDetail a={store.activeAnalysis} />
            </>
          ) : (
            <div className="empty"><span className="big">📊</span>No analysis selected yet.</div>
          )
        )}

        {page === 'ranking' && (
          <Ranking
            resume={store.resume ? { resume_id: store.resume.resume_id } : null}
            ranking={store.analyses.map((a) => ({
              rank: 0,
              job_id: a.job_id,
              title: a.job_title,
              overall_score: a.overall_score,
              breakdown: a.breakdown,
              reasons: [],
            }))}
            analyses={store.analyses}
            onAnalysisSelect={(a) => {
              setStore((s) => ({ ...s, activeAnalysis: a }))
              setPage('results')
            }}
          />
        )}

        {page === 'assistant' && (
          <Assistant
            analysisId={store.activeAnalysis?.analysis_id ?? null}
            jobTitle={store.activeAnalysis?.job_title ?? null}
          />
        )}

        {page === 'history' && <History items={history} onOpen={openHistoryItem} />}
      </main>
    </>
  )
}
