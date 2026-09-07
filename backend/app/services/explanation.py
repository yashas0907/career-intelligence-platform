"""Explanation layer: converts computed scores into transparent explanations.

Two generators:
1. Deterministic template explanation — always available, directly derived from
   scoring evidence. This is the source of truth.
2. LLM narration (optional) — rephrases the *already computed* facts; it is
   explicitly instructed not to invent numbers. Fallback = template output.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.ai.llm import LLMUnavailableError, llm
from app.services.skills import skill_display

logger = logging.getLogger("app.explain")

_COMPONENT_LABELS = {
    "skills": "Skills",
    "experience": "Experience",
    "projects": "Projects",
    "education": "Education",
    "semantic": "Semantic Relevance",
}


def deterministic_explanation(result: dict[str, Any], job_title: str) -> str:
    lines: list[str] = [f"Match Score: {result['overall_score']:.0%} for “{job_title}”", "", "Why:"]

    comps = result["components"]
    skills = comps["skills"]

    for d in skills.get("strong", [])[:6]:
        mark = "✓" if d["requirement"] == "required" else "+"
        lines.append(f"  {mark} {d['display']} appears in resume" + (f" — {d['evidence'][0][:90]}" if d.get("evidence") else ""))

    for d in skills.get("transferable", [])[:4]:
        lines.append(f"  ~ {d['evidence'][0]}")

    exp = comps["experience"]
    lines.append(f"  • Experience: {exp['evidence'][0]}")
    proj = comps["projects"]
    if proj.get("best_project"):
        lines.append(f"  • Best project evidence: {proj['best_project']} ({proj['best_similarity']:.0%} similar)")
    edu = comps["education"]
    lines.append(f"  • Education: {edu['evidence'][0]}")

    missing_req = skills.get("missing_required", [])
    if missing_req:
        lines.append("")
        lines.append("Gaps:")
        for d in missing_req[:8]:
            lines.append(f"  ✗ {d['display']} (required) not detected")

    missing_pref = skills.get("missing_preferred", [])
    for d in missing_pref[:4]:
        lines.append(f"  − {d['display']} (preferred) not detected")

    lines.append("")
    lines.append("Component breakdown:")
    for key, label in _COMPONENT_LABELS.items():
        val = result["breakdown"][key]
        w = result["weights"][key]
        lines.append(f"  {label:<20} {val:.0%}  (weight {w:.0%})")

    return "\n".join(lines)


def llm_explanation(result: dict[str, Any], job_title: str) -> str | None:
    """Ask the LLM to narrate the computed facts. Never lets it recompute scores."""
    if not llm.available:
        return None
    facts = {
        "job_title": job_title,
        "overall_score": result["overall_score"],
        "breakdown": result["breakdown"],
        "strong_skills": [d["display"] for d in result["components"]["skills"]["strong"][:10]],
        "transferable": [d["display"] for d in result["components"]["skills"]["transferable"][:5]],
        "missing_required": [d["display"] for d in result["components"]["skills"]["missing_required"][:10]],
        "experience_evidence": result["components"]["experience"]["evidence"],
        "project_evidence": result["components"]["projects"].get("best_project"),
    }
    import json

    try:
        raw = llm.chat(
            "You are a career advisor explaining a pre-computed candidate-job match analysis. "
            "Use ONLY the JSON facts provided. Do NOT invent new skills, scores or numbers. "
            "Write 4-6 concise sentences: verdict, top strengths, critical gaps, one actionable next step. "
            "Be honest about weaknesses.",
            f"Facts:\n{json.dumps(facts, indent=2)}",
        )
        return raw.strip()
    except (LLMUnavailableError, Exception) as exc:  # noqa: BLE001
        logger.warning("LLM explanation failed: %s", exc)
        return None


def explain(result: dict[str, Any], job_title: str) -> tuple[str, str]:
    """Returns (explanation_text, method)."""
    template = deterministic_explanation(result, job_title)
    narrated = llm_explanation(result, job_title)
    if narrated:
        return f"{narrated}\n\n---\nDetailed breakdown (deterministic):\n{template}", "hybrid"
    return template, "heuristic"
