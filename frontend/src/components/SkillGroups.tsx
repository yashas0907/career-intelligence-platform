import type { SkillMatchDetail, SkillComparison } from '../lib/types'

const statusPill: Record<SkillMatchDetail['status'], string> = {
  exact: 'green',
  transferable: 'amber',
  missing: 'red',
}

export function SkillItem({ d }: { d: SkillMatchDetail }) {
  return (
    <div className={`skill-item ${d.status === 'transferable' ? 'transferable' : ''}`}>
      <b>
        <span>
          {d.display}{' '}
          <span className="pill grey" style={{ marginLeft: 6, padding: '1px 7px', fontSize: '.62rem' }}>
            {d.requirement}
          </span>
        </span>
        <span className={`pill ${statusPill[d.status]}`} style={{ padding: '2px 8px', fontSize: '.66rem' }}>
          {d.status === 'exact' ? '✓ match' : d.status === 'transferable' ? '~ related' : '✗ gap'}
        </span>
      </b>
      {d.evidence[0] && <span className="ev">{d.evidence[0]}</span>}
    </div>
  )
}

export function SkillGroups({ comp }: { comp: SkillComparison }) {
  const groups: Array<[string, string, SkillMatchDetail[]]> = [
    ['✓ Strong matches', 'Skills in your resume that this job needs.', comp.strong],
    ['~ Transferable', 'Not exact matches, but related skills earn partial credit.', comp.transferable],
    ['✗ Missing required', 'Required skills not found — your highest-priority gaps.', comp.missing_required],
    ['− Missing preferred', 'Nice-to-have skills that would differentiate you.', comp.missing_preferred],
  ]
  return (
    <>
      {groups.map(([title, sub, items]) =>
        items.length > 0 ? (
          <div key={title} style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: '.92rem', margin: '0 0 2px' }}>{title} <span className="pill grey">{items.length}</span></h3>
            <p className="sub" style={{ margin: '0 0 10px', fontSize: '.78rem' }}>{sub}</p>
            <div className="skill-grid">
              {items.map((d) => <SkillItem key={d.skill} d={d} />)}
            </div>
          </div>
        ) : null,
      )}
    </>
  )
}
