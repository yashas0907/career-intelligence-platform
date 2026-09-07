"""ATS-oriented analysis (deterministic heuristic checks).

Honest scope: this does NOT replicate any proprietary ATS. It checks signals that
are publicly known to matter for keyword-based parsing: coverage of the JD's
terminology, presence of standard sections, formatting risks (tables/columns in
DOCX, contact discoverability), and readability proxies.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.skills import skill_display

_EXPECTED_SECTIONS = ["experience", "education", "skills", "projects"]


def _contact_score(profile: dict[str, Any]) -> tuple[float, list[str]]:
    contact = profile.get("contact") or {}
    have = [k for k in ("email", "phone") if contact.get(k)]
    score = len(have) / 2
    notes = []
    if "email" not in have:
        notes.append("Email not detected — most ATS map applications via email")
    if "phone" not in have:
        notes.append("Phone number not detected")
    if score == 1.0:
        notes.append("Email and phone both present")
    return score, notes


def _section_score(profile: dict[str, Any]) -> tuple[float, list[str]]:
    detected = set(profile.get("sections_detected") or [])
    missing = [s for s in _EXPECTED_SECTIONS if s not in detected]
    score = 1 - len(missing) / len(_EXPECTED_SECTIONS)
    notes = []
    if missing:
        notes.append(f"Sections not detected: {', '.join(missing)} (explicit section headings help ATS parsing)")
    else:
        notes.append("All core sections detected (experience, education, skills, projects)")
    return score, notes


def _keyword_score(parsed_jd: dict[str, Any], profile: dict[str, Any]) -> tuple[float, list[str]]:
    cand = set((profile.get("skills") or {}).keys())
    req = set(parsed_jd.get("required_skills") or [])
    pref = set(parsed_jd.get("preferred_skills") or [])
    if not req:
        return 0.7, ["No explicit required keywords found in JD"]
    covered = cand & req
    score = len(covered) / len(req)
    missing = req - cand
    notes = []
    if missing:
        notes.append("Missing job keywords: " + ", ".join(skill_display(s) for s in sorted(missing)[:8]))
    if covered:
        notes.append(f"Covers {len(covered)}/{len(req)} required keywords")
    return score, notes


def _format_score(profile: dict[str, Any], resume_text: str, parse_method: str) -> tuple[float, list[str]]:
    score = 1.0
    notes = []
    if "|" in resume_text and resume_text.count("|") > 10:
        score -= 0.25
        notes.append("Many pipe characters detected — possible table-based layout, which some parsers read poorly")
    if parse_method == "pdf":
        # check for common image-heavy PDF symptom: little text per page
        pages = max(1, resume_text.count("\f") + 1)
        density = len(resume_text) / pages
        if density < 300:
            score -= 0.3
            notes.append(f"Low text density (~{density:.0f} chars/page) — PDF may contain images/scans")
    if re.search(r"[①-⑭]", resume_text):
        score -= 0.1
        notes.append("Unusual unicode symbols detected — can confuse older parsers")
    if score >= 0.9:
        notes.append("No major formatting risks detected")
    return max(0.0, score), notes


def _readability_score(resume_text: str) -> tuple[float, list[str]]:
    lines = [l for l in resume_text.splitlines() if l.strip()]
    if not lines:
        return 0.0, ["Empty document"]
    long_lines = sum(1 for l in lines if len(l) > 160)
    score = 1.0 - min(1.0, long_lines / max(1, len(lines)) * 2)
    notes = []
    if long_lines:
        notes.append(f"{long_lines} very long line(s) (>160 chars) — consider breaking into bullets")
    bullets = sum(1 for l in lines if re.match(r"^\s*[-•*]", l))
    if bullets == 0:
        score -= 0.1
        notes.append("No bullet points detected — bullets improve scannability for humans and ATS")
    return max(0.0, min(1.0, score)), notes


_SUBSCORE_WEIGHTS = {"keywords": 0.35, "sections": 0.20, "formatting": 0.20, "contact": 0.10, "readability": 0.15}


def analyze_ats(profile: dict[str, Any], resume_text: str, parsed_jd: dict[str, Any], parse_method: str) -> dict[str, Any]:
    kw, kw_notes = _keyword_score(parsed_jd, profile)
    sec, sec_notes = _section_score(profile)
    fmt, fmt_notes = _format_score(profile, resume_text, parse_method)
    con, con_notes = _contact_score(profile)
    rd, rd_notes = _readability_score(resume_text)

    subscores = {"keywords": kw, "sections": sec, "formatting": fmt, "contact": con, "readability": rd}
    overall = sum(_SUBSCORE_WEIGHTS[k] * v for k, v in subscores.items())

    recommendations: list[str] = []
    if kw < 0.6:
        missing = ", ".join(
            skill_display(s)
            for s in sorted(set(parsed_jd.get("required_skills") or []) - set((profile.get("skills") or {}).keys()))[:6]
        )
        if missing:
            recommendations.append(f"Mirror the job's terminology naturally: {missing}")
    if sec < 1.0:
        recommendations.append("Add explicit standard section headings (Experience, Education, Skills, Projects)")
    for note in fmt_notes:
        if note.startswith(("Many", "Low", "Unusual")):
            recommendations.append(note)
    if not con_notes or "not detected" in " ".join(con_notes):
        recommendations.append("Ensure email and phone are in plain text near the top of the resume")

    return {
        "overall": round(overall, 4),
        "subscores": {k: round(v, 4) for k, v in subscores.items()},
        "notes": {
            "keywords": kw_notes,
            "sections": sec_notes,
            "formatting": fmt_notes,
            "contact": con_notes,
            "readability": rd_notes,
        },
        "recommendations": recommendations[:8],
        "disclaimer": "Heuristic, ATS-oriented analysis based on publicly known parsing signals — not a replica of any proprietary system.",
    }
