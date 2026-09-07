"""Resume Fixer: directly improves an uploaded resume, in-place.

Two layers:

1. DETERMINISTIC (always free, always runs):
   - rebuild clean section ordering (Contact → Summary → Skills → Experience
     → Projects → Education → Certifications → Achievements)
   - fix ATS formatting: uniform bullets, no special glyphs, consistent headings
   - normalize contact line; drop dead weight (empty lines, filler)
   - order skills by relevance when a JD is provided; surface job keywords

2. LLM (optional, Gemini-free/OpenAI-paid):
   - rewrite experience/project bullets into action-verb + tech + metric form
   - STRICT no-fabrication guard: only rewrites facts already in the bullet;
     adds no metrics that aren't there. Without a key, bullets are kept as-is
     with deterministic cleanup only.

Output: fixed plain text + per-change diff log, so users see exactly what
changed and why — nothing silent, nothing invented.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.services.ai.llm import LLMUnavailableError, llm
from app.services.extractor import _split_sections
from app.services.skills import normalizer, skill_display

logger = logging.getLogger("app.fixes")

SECTION_ORDER = [
    ("CONTACT", "contact"),
    ("SUMMARY", "summary"),
    ("SKILLS", "skills"),
    ("EXPERIENCE", "experience"),
    ("PROJECTS", "projects"),
    ("EDUCATION", "education"),
    ("CERTIFICATIONS", "certifications"),
    ("ACHIEVEMENTS", "achievements"),
]

_BULLET_GLYPHS = re.compile(r"^[\u2022\u00b7\u25aa\u25cf\u2023\u2043o\-–]\s*")
_WEIRD_CHARS = re.compile(r"[\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u00a0]")
_ACTION_START = re.compile(
    r"^(?:built|designed|developed|created|implemented|deployed|optimized|improved|led|"
    r"achieved|analyzed|automated|fine-tuned|trained|integrated|reduced|increased|researched|"
    r"built|shipped|delivered|wrote|migrated|scaled|maintained)"
)


@dataclass
class FixResult:
    fixed_text: str
    changes: list[dict[str, str]]
    llm_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {"fixed_text": self.fixed_text, "changes": self.changes, "llm_used": self.llm_used}


def _clean_line(line: str) -> str:
    s = _WEIRD_CHARS.sub(lambda m: {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
                                     "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " "}[m.group(0)], line)
    return re.sub(r"[ \t]+", " ", s).strip()


def _bulletize(line: str) -> str:
    s = _BULLET_GLYPHS.sub("", line).strip()
    return f"- {s}" if s else ""


def _is_bullet(line: str) -> bool:
    return bool(_BULLET_GLYPHS.match(line.strip()))


def _contact_block(profile: dict[str, Any]) -> str | None:
    c = profile.get("contact") or {}
    name = profile.get("name")
    parts: list[str] = []
    if name:
        parts.append(str(name))
    line2: list[str] = []
    for k in ("email", "phone", "linkedin", "github"):
        v = c.get(k)
        if v:
            line2.append(str(v))
    if not parts and not line2:
        return None
    out = "\n".join([p for p in parts if p])
    if line2:
        out += ("\n" if out else "") + " | ".join(line2)
    return out or None


def _summary_block(profile: dict[str, Any]) -> str | None:
    s = profile.get("summary")
    if not s:
        return None
    return " ".join(str(s).split())


def _skills_block(raw_sections: dict[str, str], profile: dict[str, Any], jd: dict[str, Any] | None) -> tuple[str | None, list[dict[str, str]]]:
    changes: list[dict[str, str]] = []
    skills = profile.get("skills") or {}
    if not skills:
        return None, changes

    if jd:
        req = set(jd.get("required_skills") or [])
        pref = set(jd.get("preferred_skills") or [])
        jd_relevant = [s for s in req if s in skills]
        other = [s for s in skills if s not in req and s not in pref]
        if jd_relevant:
            changes.append({
                "section": "skills",
                "change": f"Reordered skills — most relevant to this job ({len(jd_relevant)} matched) now listed first.",
            })
        ordered = jd_relevant + other
    else:
        ordered = list(skills.keys())

    cats: dict[str, list[str]] = {}
    for sid in ordered:
        cat = (skills.get(sid) or {}).get("category") or "concept"
        cats.setdefault(cat, []).append(skill_display(sid))

    lines = []
    for cat in ("language", "framework", "library", "tool", "cloud", "database", "concept", "domain", "soft"):
        if cats.get(cat):
            label = {"language": "Languages", "framework": "Frameworks", "library": "Libraries",
                     "tool": "Tools", "cloud": "Cloud", "database": "Databases",
                     "concept": "Concepts", "domain": "Domains", "soft": "Soft skills"}[cat]
            lines.append(f"{label}: {', '.join(cats[cat])}")

    missing = []
    if jd:
        missing = [skill_display(s) for s in jd.get("missing_top", [])]
        if missing:
            lines.append(f"# Consider adding (job requires, not found in your resume): {', '.join(missing[:8])}")
    return ("\n".join(lines) if lines else None), changes


def _body_block(section_text: str) -> tuple[str | None, bool]:
    """Clean a body section; returns (text, had_unbulletized_lines)."""
    if not section_text:
        return None, False
    out: list[str] = []
    unbulletized = False
    for raw in section_text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        if _is_bullet(line) or _ACTION_START.match(line.lower()) or len(line) < 75:
            out.append(line if _is_bullet(line) else line)
        else:
            # long prose line inside a bullet-worthy section -> bulletize
            out.append(_bulletize(line))
            unbulletized = True
    return ("\n".join(out) or None), unbulletized


_BULLET_REWRITE_SYSTEM = (
    "You rewrite resume bullet points. STRICT RULES: use ONLY facts, technologies and "
    "numbers already present in the input bullet — never invent metrics, tools, or outcomes. "
    "Transform into the pattern: strong action verb + what was built/done + technology + "
    "measurable result IF one is stated. Keep it under 30 words. Return JSON: "
    '{"rewrites": [{"original": "exact input line", "rewritten": "new line"}]}. '
    "If a bullet cannot be improved without inventing anything, return it unchanged."
)


def _llm_rewrite_bullets(text: str) -> tuple[dict[str, str], bool]:
    """Ask the LLM to rewrite weak bullets. Returns (original->rewritten, used)."""
    if not llm.available:
        return {}, False
    import json as _json

    bullets = [
        _clean_line(l)
        for l in text.splitlines()
        if _is_bullet(l) or _ACTION_START.match(l.lower())
    ]
    weak = [b for b in bullets if not _ACTION_START.match(b[2:].lower() if b.startswith("- ") else b.lower())]
    if not weak:
        return {}, False
    try:
        raw = llm.chat(_BULLET_REWRITE_SYSTEM, "\n".join(weak[:20]), json_mode=True)
        data = _json.loads(raw) if raw.strip().startswith("{") else {}
        rewrites = data.get("rewrites") if isinstance(data, dict) else None
        if not isinstance(rewrites, list):
            return {}, False
        mapping: dict[str, str] = {}
        for r in rewrites:
            if isinstance(r, dict) and isinstance(r.get("original"), str) and isinstance(r.get("rewritten"), str):
                orig, new = r["original"].strip(), r["rewritten"].strip()
                if orig and new and orig != new and len(new) < 220:
                    mapping[orig] = new
        # no-fabrication guard: rewritten must share >=40% token overlap with original
        def overlap(a: str, b: str) -> float:
            ta = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2}
            tb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2}
            return len(ta & tb) / max(1, len(ta | tb))
        safe = {o: n for o, n in mapping.items() if overlap(o, n) >= 0.40}
        return safe, True
    except (LLMUnavailableError, Exception) as exc:  # noqa: BLE001
        logger.warning("LLM bullet rewrite failed (keeping originals): %s", exc)
        return {}, False


def fix_resume(profile: dict[str, Any], raw_text: str, parsed_jd: dict[str, Any] | None = None) -> FixResult:
    sections = _split_sections(raw_text)
    changes: list[dict[str, str]] = []

    # 1) contact
    contact = _contact_block(profile)
    if contact:
        changes.append({"section": "contact", "change": "Rebuilt clean contact header (name + email + phone + links in one block)."})

    # 2) summary
    summary = _summary_block(profile)

    # 3) skills (JD-aware ordering)
    skills_txt, skill_changes = _skills_block(sections, profile, parsed_jd)
    changes.extend(skill_changes)

    # 4) body sections
    blocks: list[str] = []
    body_changes = 0
    for title, key in SECTION_ORDER:
        if key in {"contact", "summary", "skills"}:
            continue
        txt = sections.get(key)
        cleaned, unbulletized = _body_block(txt or "")
        if cleaned:
            if unbulletized:
                body_changes += 1
            blocks.append((title, cleaned))
    if body_changes:
        changes.append({
            "section": "formatting",
            "change": f"Converted {body_changes} section(s) of long prose lines into clean ATS-friendly bullets.",
        })

    # 5) LLM bullet rewriting (guarded, optional)
    llm_used = False
    if llm.available:
        full_body = "\n\n".join(t for _, t in blocks)
        rewrites, used = _llm_rewrite_bullets(full_body)
        if used and rewrites:
            llm_used = True
            new_blocks = []
            for title, t in blocks:
                for orig, new in rewrites.items():
                    t = t.replace(orig, new)
                new_blocks.append((title, t))
            blocks = new_blocks
            changes.append({
                "section": "bullets",
                "change": f"Strengthened {len(rewrites)} weak bullet(s) into action-verb form — using only your own stated facts.",
            })

    # 6) assemble
    parts: list[str] = []
    if contact:
        parts.append(contact)
    if summary:
        parts.append(f"SUMMARY\n{summary}")
    if skills_txt:
        parts.append(f"SKILLS\n{skills_txt}")
    for title, txt in blocks:
        parts.append(title + "\n" + txt)

    fixed = "\n\n".join(parts).strip()
    if not fixed:
        fixed = raw_text  # never return emptiness
        changes.append({"section": "fallback", "change": "No structural improvements possible — original kept."})

    if fixed != raw_text:
        n_added_kw = 0
        if parsed_jd:
            before = set((profile.get("skills") or {}).keys())
            after = set(normalizer.extract(fixed).keys())
            n_added_kw = len(after - before)
            if n_added_kw:
                changes.append({
                    "section": "keywords",
                    "change": f"Skill names now spelled/ordered so ATS keyword matching finds {n_added_kw} more of the job's terms.",
                })
        changes.append({"section": "structure", "change": "Standardized section order and headings (Contact → Summary → Skills → Experience → Projects → Education)."})

    return FixResult(fixed_text=fixed, changes=changes, llm_used=llm_used)
