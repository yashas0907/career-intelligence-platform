"""Per-field extraction evaluation: precision/recall against labeled ground truth.

Run standalone:
    python tests/eval_extraction.py

Or via pytest (thresholds enforced in CI):
    python -m pytest tests/test_eval_extraction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
FIXTURES = Path(__file__).parent / "fixtures"
for p in (str(BACKEND), str(FIXTURES)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_corpus import EVAL_RESUMES  # noqa: E402

from app.services.extractor import extract_resume  # noqa: E402


def _prf(found: set, truth: set) -> tuple[float, float, float]:
    """(precision, recall, f1) with safe zero handling."""
    if not truth and not found:
        return (1.0, 1.0, 1.0)
    tp = len(found & truth)
    prec = tp / len(found) if found else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return (prec, rec, f1)


def evaluate() -> dict:
    report: dict = {"per_resume": [], "fields": {}}
    field_scores: dict[str, list[tuple[float, float, float]]] = {}

    for case in EVAL_RESUMES:
        profile = extract_resume(case["resume_text"])
        truth = case["ground_truth"]

        checks = {
            "name": (set([profile.get("name") or ""]), set([truth["name"]])),
            "email": (set([profile.get("contact", {}).get("email") or ""]), set([truth["email"]])),
            "skills": (set(profile.get("skills", {}).keys()), truth["skills"]),
            "projects": (
                {p.get("name", "") for p in profile.get("projects", [])},
                truth["projects"],
            ),
            "experience_titles": (
                {(e.get("title") or "").strip() for e in profile.get("experience", [])},
                truth["experience_titles"],
            ),
            "education_degrees": (
                {(e.get("degree") or "").lower() for e in profile.get("education", [])},
                truth["education_degrees"],
            ),
        }

        row = {"case": case["name"], "fields": {}}
        for field, (found, gt) in checks.items():
            found = {f for f in found if f}
            prec, rec, f1 = _prf(found, gt)
            row["fields"][field] = {
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
                "tp": len(found & gt),
                "found": len(found),
                "truth": len(gt),
            }
            field_scores.setdefault(field, []).append((prec, rec, f1))
        report["per_resume"].append(row)

    for field, scores in field_scores.items():
        n = len(scores)
        report["fields"][field] = {
            "precision": round(sum(s[0] for s in scores) / n, 3),
            "recall": round(sum(s[1] for s in scores) / n, 3),
            "f1": round(sum(s[2] for s in scores) / n, 3),
        }
    return report


def print_report(report: dict) -> None:
    print("=" * 62)
    print("EXTRACTION EVALUATION (labeled corpus, heuristic pipeline)")
    print("=" * 62)
    for row in report["per_resume"]:
        print(f"\nResume: {row['case']}")
        for field, m in row["fields"].items():
            print(
                f"  {field:<20} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}"
                f"  (tp={m['tp']} found={m['found']} truth={m['truth']})"
            )
    print("\n" + "-" * 62)
    print("MACRO AVERAGES")
    for field, m in report["fields"].items():
        print(f"  {field:<20} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")


if __name__ == "__main__":
    print_report(evaluate())
