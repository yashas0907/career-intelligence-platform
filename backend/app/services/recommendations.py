"""Resume improvement recommendation engine (deterministic, evidence-based).

Rules are derived from the match result — every recommendation cites the signal
that triggered it. The engine NEVER suggests inventing experience; gap-filling
suggestions are always "learn / build / quantify", phrased safely.
"""

from __future__ import annotations

from typing import Any

from app.services.skills import related_skills, skill_display

def _learning_priority(skill_id: str, requirement: str) -> int:
    # required core tech > required soft/domain > preferred
    if requirement == "required":
        return 2
    return 1


def build_recommendations(result: dict[str, Any], profile: dict[str, Any], parsed_jd: dict[str, Any]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    comps = result["components"]
    skills = comps["skills"]

    # 1) missing required skills -> learning plan
    missing_req = skills.get("missing_required", [])
    for d in missing_req[:5]:
        rec = {
            "type": "learn",
            "priority": "high" if d["category"] in {"language", "framework", "library", "tool", "cloud", "database"} else "medium",
            "title": f"Learn {d['display']}",
            "detail": (
                f"Required by the job but not found in your resume. "
                f"Build one small project using {d['display']} and add it to your Projects section with concrete outcomes."
            ),
            "signal": f"missing_required:{d['skill']}",
        }
        # suggest a bridge from related known skill
        cand = set((profile.get("skills") or {}).keys())
        bridges = sorted(related_skills(d["skill"]) & cand)
        if bridges:
            rec["detail"] += f" You already know related {skill_display(bridges[0])}, which lowers the ramp-up."
        recs.append(rec)

    # 2) missing preferred skills -> lower priority
    for d in skills.get("missing_preferred", [])[:3]:
        recs.append({
            "type": "learn",
            "priority": "low",
            "title": f"Consider picking up {d['display']} (preferred)",
            "detail": f"A preferred (nice-to-have) skill for this role. Not critical, but it differentiates candidates.",
            "signal": f"missing_preferred:{d['skill']}",
        })

    # 3) projects evidence
    proj = comps["projects"]
    if proj.get("score", 1) < 0.5 and not (profile.get("projects") or []):
        recs.append({
            "type": "build",
            "priority": "high",
            "title": "Add a projects section",
            "detail": "No projects were detected. For early-career candidates, 2-3 projects with measurable outcomes are the strongest signal.",
            "signal": "projects:empty",
        })
    elif proj.get("score", 1) < 0.5:
        recs.append({
            "type": "build",
            "priority": "medium",
            "title": "Align one project with this role",
            "detail": f"Your projects scored {proj['score']:.0%} relevance. Add a project that uses the job's core stack: "
            + ", ".join(skill_display(s) for s in (parsed_jd.get("required_skills") or [])[:4]) + ".",
            "signal": f"projects_score:{proj.get('score')}",
        })

    # 4) experience phrasing
    exp = comps["experience"]
    if exp.get("years", 0) < 1 and (profile.get("experience") or []):
        recs.append({
            "type": "improve",
            "priority": "medium",
            "title": "Quantify your experience bullets",
            "detail": "Rewrite experience bullets as: action verb + technology + measurable result (e.g. 'Built X with Y, reducing Z by N%'). Avoid duration-only descriptions.",
            "signal": "experience:weak_bullets",
        })

    # 5) education clarity
    edu = comps["education"]
    if edu.get("score", 1) < 0.6:
        recs.append({
            "type": "improve",
            "priority": "low",
            "title": "Make education explicit",
            "detail": "State your degree, institution and (expected) graduation year under a clear 'Education' heading — ATS parsers look for these keywords.",
            "signal": "education:unclear",
        })

    # 6) summary tailoring
    if not profile.get("summary"):
        recs.append({
            "type": "improve",
            "priority": "low",
            "title": "Add a 2-3 line summary",
            "detail": "A short summary containing the role's key terms improves both human scan and keyword coverage.",
            "signal": "summary:missing",
        })

    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order.get(r["priority"], 3))
    return recs[:10]


def rank_jobs(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank job analyses for the recommendation view. Ties broken by skills score."""
    ranked = sorted(
        analyses,
        key=lambda a: (a["overall_score"], a["components"]["skills"]["score"]),
        reverse=True,
    )
    out = []
    for rank, a in enumerate(ranked, start=1):
        reasons = []
        bd = a["breakdown"]
        reasons.append(f"Skills match {bd['skills']:.0%}, semantic fit {bd['semantic']:.0%}")
        strong = a["components"]["skills"].get("strong", [])
        if strong:
            reasons.append("Strengths: " + ", ".join(d["display"] for d in strong[:4]))
        gaps = a["components"]["skills"].get("missing_required", [])
        if gaps:
            reasons.append("Main gaps: " + ", ".join(d["display"] for d in gaps[:4]))
        out.append({
            "rank": rank,
            "job_id": a["job_id"],
            "title": a.get("job_title", "Role"),
            "overall_score": a["overall_score"],
            "breakdown": bd,
            "reasons": reasons,
        })
    return out
