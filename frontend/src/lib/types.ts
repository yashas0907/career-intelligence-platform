/** Shared API types mirroring backend Pydantic schemas. */

export interface SkillMatchDetail {
  skill: string
  display: string
  category: string
  requirement: 'required' | 'preferred'
  status: 'exact' | 'transferable' | 'missing'
  score: number
  via?: string | null
  evidence: string[]
}

export interface SkillComparison {
  score: number
  details: SkillMatchDetail[]
  strong: SkillMatchDetail[]
  transferable: SkillMatchDetail[]
  missing_required: SkillMatchDetail[]
  missing_preferred: SkillMatchDetail[]
}

export interface ATSAnalysis {
  overall: number
  subscores: Record<string, number>
  notes: Record<string, string[]>
  recommendations: string[]
  disclaimer: string
}

export interface Recommendation {
  type: 'learn' | 'build' | 'improve'
  priority: 'high' | 'medium' | 'low'
  title: string
  detail: string
  signal: string
}

export interface ComponentScore {
  score: number
  evidence?: string[]
  years?: number
  semantic?: number
  best_project?: string
  best_similarity?: number
  skill_coverage?: number
}

export interface MatchAnalysis {
  analysis_id: string
  resume_id: string
  job_id: string
  job_title: string
  overall_score: number
  breakdown: {
    skills: number
    experience: number
    projects: number
    education: number
    semantic: number
  }
  weights: Record<string, number>
  components: {
    skills: SkillComparison
    experience: ComponentScore
    projects: ComponentScore
    education: ComponentScore
    semantic: ComponentScore
  }
  skill_comparison: SkillComparison
  ats: ATSAnalysis
  recommendations: Recommendation[]
  explanation: string
  explanation_method: string
  created_at?: string
}

export interface RankedJob {
  rank: number
  job_id: string
  title: string
  overall_score: number
  breakdown: Record<string, number>
  reasons: string[]
}

export interface ChatSource {
  source: string
  label: string
  similarity: number
  excerpt: string
}

export interface ChatResponse {
  answer: string
  sources: ChatSource[]
  method: string
  analysis_id: string | null
}

export interface ResumeProfile {
  resume_id: string
  filename: string
  parse_method: string
  extraction_status: string
  profile: {
    name?: string | null
    contact?: Record<string, string | null>
    summary?: string | null
    education?: Array<Record<string, unknown>>
    experience?: Array<Record<string, unknown>>
    projects?: Array<{ name: string; description?: string[] }>
    certifications?: string[]
    skills?: Record<string, { category: string; evidence: string[] }>
    total_years_experience?: number | null
    sections_detected?: string[]
  }
  warnings: string[]
  created_at: string
}

export interface HealthInfo {
  status: string
  version: string
  llm_available: boolean
  embedding_backend: string
  database: string
}

export interface HistoryItem {
  analysis_id: string
  job_id: string
  job_title: string
  overall_score: number
  created_at: string
}
