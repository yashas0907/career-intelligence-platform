import { useRef, useState, type DragEvent } from 'react'
import { api } from '../lib/api'
import { Card, ErrorBox, Spinner } from '../components/ui'
import type { ResumeProfile } from '../lib/types'

const catColor: Record<string, string> = {
  language: 'blue', framework: 'green', library: 'green', tool: 'amber',
  cloud: 'blue', database: 'amber', concept: 'grey', domain: 'grey', soft: 'grey',
}

export default function UploadResume({
  resume,
  onUploaded,
}: {
  resume: ResumeProfile | null
  onUploaded: (r: ResumeProfile) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  async function upload(file: File) {
    setError(null)
    setUploading(true)
    try {
      const r = await api.uploadResume(file)
      onUploaded(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) void upload(f)
  }

  const p = resume?.profile

  return (
    <div>
      <div className="grid-2">
        <Card title="Upload resume" sub="PDF, DOCX or TXT · max 10 MB · scanned/image-only PDFs aren't supported">
          <div
            className={`dropzone ${dragOver ? 'over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            aria-label="Upload resume file"
          >
            {uploading ? <><Spinner /> <span style={{ marginLeft: 8 }}>Parsing & extracting…</span></> : (
              <>
                <span className="big">📄</span>
                Drop your resume here or <u>browse</u>
                <small>The file is parsed, skills are extracted with evidence, and a structured profile is built.</small>
              </>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void upload(f)
            }}
          />
          {error && <ErrorBox onDismiss={() => setError(null)}>{error}</ErrorBox>}
        </Card>

        <Card title="Candidate profile" sub={resume ? `Parsed via ${resume.parse_method} · extraction: ${resume.extraction_status}` : 'Upload a resume to see the structured profile'}>
          {resume && p ? (
            <>
              <div className="kv" style={{ marginBottom: 14 }}>
                <dt>Name</dt><dd>{p.name ?? '—'}</dd>
                <dt>Email</dt><dd>{p.contact?.email ?? '—'}</dd>
                <dt>Phone</dt><dd>{p.contact?.phone ?? '—'}</dd>
                <dt>Experience</dt><dd>{p.total_years_experience != null ? `~${p.total_years_experience} yrs` : '—'}</dd>
                <dt>Sections</dt><dd>{(p.sections_detected ?? []).join(', ') || '—'}</dd>
              </div>
              {resume.warnings.length > 0 && (
                <div className="warn-box">{resume.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}</div>
              )}
              <h3 style={{ fontSize: '.9rem', margin: '14px 0 8px' }}>
                Extracted skills ({Object.keys(p.skills ?? {}).length})
              </h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7 }}>
                {Object.entries(p.skills ?? {}).map(([id, meta]) => (
                  <span key={id} className={`pill ${catColor[meta.category] ?? 'grey'}`} title={meta.evidence?.[0] ?? ''}>
                    {id.replace(/-/g, ' ')} <span className="cat">{meta.category}</span>
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="empty"><span className="big">🗂️</span>No profile yet</div>
          )}
        </Card>
      </div>

      {resume && p && (
        <div className="grid-2" style={{ marginTop: 16 }}>
          <Card title="Education & experience">
            {((p.education ?? []) as Array<Record<string, unknown>>).map((e, i) => (
              <ul key={i} className="section-list" style={{ marginBottom: 12 }}>
                <li><span className="t">{String(e.raw ?? e.degree ?? 'Education entry')}</span></li>
                {e.institution ? <li className="d">{String(e.institution)}</li> : null}
              </ul>
            ))}
            {((p.experience ?? []) as Array<Record<string, unknown>>).map((x, i) => (
              <ul key={i} className="section-list" style={{ marginBottom: 12 }}>
                <li>
                  <span className="t">{String(x.title ?? 'Role')}</span>
                  {x.date_range ? <span className="d" style={{ float: 'right' }}>{String(x.date_range)}</span> : null}
                </li>
                {Array.isArray(x.bullets) && (x.bullets as string[]).slice(0, 2).map((b, j) => (
                  <li className="d" key={j}>• {b}</li>
                ))}
              </ul>
            ))}
          </Card>
          <Card title="Projects & certifications">
            {(p.projects ?? []).map((proj, i) => (
              <ul key={i} className="section-list" style={{ marginBottom: 12 }}>
                <li><span className="t">{proj.name}</span></li>
                {(proj.description ?? []).slice(0, 2).map((d, j) => <li className="d" key={j}>• {d}</li>)}
              </ul>
            ))}
            {(p.certifications ?? []).length > 0 && (
              <>
                <h3 style={{ fontSize: '.9rem', margin: '12px 0 6px' }}>Certifications</h3>
                <ul className="section-list">
                  {(p.certifications ?? []).map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              </>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
