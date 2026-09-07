"""CI-enforced extraction quality thresholds from the labeled corpus.

Thresholds are deliberately honest: set slightly below current measured scores
so they catch regressions without over-claiming.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
FIXTURES = Path(__file__).parent / "fixtures"
for p in (str(BACKEND), str(FIXTURES)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LLM_ENABLED", "false")

from eval_extraction import evaluate, print_report  # noqa: E402

THRESHOLDS = {
    # field: minimum acceptable macro-F1 on the labeled corpus (measured 2026-09)
    "name": 0.95,
    "email": 0.95,
    "skills": 0.80,
    "projects": 0.90,
    "experience_titles": 0.90,
    "education_degrees": 0.90,
}


def test_extraction_quality_thresholds():
    report = evaluate()
    failures = []
    for field, min_f1 in THRESHOLDS.items():
        got = report["fields"][field]["f1"]
        assert got >= min_f1, f"{field}: F1={got:.3f} below threshold {min_f1}"
        print(f"  ok  {field:<20} F1={got:.2f} (min {min_f1})")
    assert not failures


def test_report_runs():
    print_report(evaluate())  # smoke: report generation itself
