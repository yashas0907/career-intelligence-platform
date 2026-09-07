import type { MatchAnalysis, RankedJob, ResumeProfile, HistoryItem, HealthInfo } from './types'

export interface AppStore {
  // data
  health: HealthInfo | null
  resume: ResumeProfile | null
  analyses: MatchAnalysis[]
  activeAnalysis: MatchAnalysis | null
  ranking: RankedJob[]
  history: HistoryItem[]
  // ui state
  uploading: boolean
  analyzing: boolean
  busy: boolean
  error: string | null
}

export type PageId = 'dashboard' | 'upload' | 'jobs' | 'results' | 'ranking' | 'assistant' | 'history'

export const emptyStore: AppStore = {
  health: null,
  resume: null,
  analyses: [],
  activeAnalysis: null,
  ranking: [],
  history: [],
  uploading: false,
  analyzing: false,
  busy: false,
  error: null,
}
