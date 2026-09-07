/** Session persistence: survives page refreshes via localStorage. */

const KEY = 'career-intel-session-v1'

export interface StoredSession {
  resumeId: string | null
  activeAnalysisId: string | null
  analysisIds: string[]
}

export function loadSession(): StoredSession {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return { resumeId: null, activeAnalysisId: null, analysisIds: [] }
    const s = JSON.parse(raw)
    return {
      resumeId: typeof s.resumeId === 'string' ? s.resumeId : null,
      activeAnalysisId: typeof s.activeAnalysisId === 'string' ? s.activeAnalysisId : null,
      analysisIds: Array.isArray(s.analysisIds) ? s.analysisIds.filter((x: unknown) => typeof x === 'string') : [],
    }
  } catch {
    return { resumeId: null, activeAnalysisId: null, analysisIds: [] }
  }
}

export function saveSession(s: StoredSession): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s))
  } catch { /* storage unavailable (private mode) — session just won't persist */ }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(KEY)
  } catch { /* ignore */ }
}
