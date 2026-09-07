"""Tests: scoring engine — the methodology contract.

These tests verify scoring BEHAVES CORRECTLY for known cases:
* perfect skill overlap -> skills score 1.0
* total absence -> skills score 0.0
* weights sum to 1
* transferable credit partial
* score monotonicity: more matched skills never lowers the score
"""

import pytest

from app.services.matching.scoring import (
    TRANSFERABLE_CREDIT,
    WEIGHTS,
    analyze_match,
    compare_skills,
)

JD = {
    "required_skills": ["python", "pytorch", "docker"],
    "preferred_skills": ["aws"],
    "min_years_experience": 1.0,
    "education_requirements": ["bachelor"],
    "responsibilities": ["Build ML models and deploy services"],
    "required_bullets": ["Strong Python and PyTorch experience", "Docker deployment"],
}


def _profile(skills, years=2.0, projects=None):
    return {
        "skills": {s: {"category": "concept", "evidence": ["found"]} for s in skills},
        "total_years_experience": years,
        "projects": projects or [],
        "experience": [],
        "education": [{"degree": "B.Tech", "raw": "B.Tech Computer Science"}],
    }


class TestSkillComparison:
    def test_all_matched_scores_full(self):
        res = compare_skills(_profile(["python", "pytorch", "docker", "aws"]), JD)
        assert res["score"] == 1.0
        assert len(res["strong"]) == 4
        assert not res["missing_required"]

    def test_none_matched_scores_zero(self):
        res = compare_skills(_profile(["rust"]), JD)
        assert res["score"] == 0.0
        assert len(res["missing_required"]) == 3
        assert len(res["missing_preferred"]) == 1

    def test_transferable_partial_credit(self):
        # tensorflow is related to pytorch -> transferable
        res = compare_skills(_profile(["python", "tensorflow", "docker"]), JD)
        assert res["score"] > 0
        transferable = [d for d in res["transferable"]]
        assert any(d["skill"] == "pytorch" for d in transferable)
        # transferable must score between 0 and exact
        pytorch_detail = next(d for d in res["details"] if d["skill"] == "pytorch")
        assert 0 < pytorch_detail["score"] < 1.0
        assert pytorch_detail["via"] == "tensorflow"

    def test_required_weights_3x(self):
        # verify formula: (3r + 1p) / (3R + 1P)
        res = compare_skills(_profile(["python"]), JD)  # 1 of 3 required, 0 pref
        expected = (3 * 1) / (3 * 3 + 1 * 1)
        assert abs(res["score"] - expected) < 1e-6

    def test_preferred_counts_less(self):
        only_pref = compare_skills(_profile(["aws"]), JD)
        assert 0 < only_pref["score"] < 0.2

    def test_evidence_always_present(self):
        res = compare_skills(_profile(["python", "rust"]), JD)
        for d in res["details"]:
            assert d["evidence"], f"no evidence for {d['skill']}"


class TestOverallScoring:
    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_overall_within_bounds(self):
        resume_text = "Python developer with PyTorch Docker experience."
        result = analyze_match(_profile(["python", "pytorch", "docker", "aws"]), resume_text, JD, resume_text)
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_better_candidate_scores_higher(self):
        weak = _profile(["python"], years=0.0)
        strong = _profile(["python", "pytorch", "docker", "aws"], years=3.0)
        jd_text = "Python PyTorch Docker AWS ML role"
        r_weak = analyze_match(weak, "Python only", JD, jd_text)
        r_strong = analyze_match(strong, "Python PyTorch Docker AWS experience", JD, jd_text)
        assert r_strong["overall_score"] > r_weak["overall_score"]

    def test_experience_gap_partial(self):
        from app.services.matching.scoring import _score_years

        s_full, _ = _score_years({"total_years_experience": 2.0}, {"min_years_experience": 1.0})
        s_half, _ = _score_years({"total_years_experience": 0.5}, {"min_years_experience": 1.0})
        s_none, _ = _score_years({"total_years_experience": None}, {"min_years_experience": 1.0})
        assert s_full == 1.0
        assert 0 < s_half < s_full
        assert s_none < s_half

    def test_education_levels(self):
        from app.services.matching.scoring import score_education

        phd = score_education({"education": [{"raw": "PhD Computer Science"}]}, {"education_requirements": ["bachelor"]})
        bsc = score_education({"education": [{"raw": "B.Sc Computer Science"}]}, {"education_requirements": ["bachelor"]})
        none = score_education({"education": []}, {"education_requirements": ["bachelor"]})
        assert phd["score"] == 1.0
        assert bsc["score"] == 1.0
        assert none["score"] < 0.5

    def test_no_requirements_neutral(self):
        from app.services.matching.scoring import score_education

        res = score_education({"education": []}, {"education_requirements": []})
        assert res["score"] >= 0.6  # neutral, not punitive

    def test_projects_semantic(self):
        from app.services.matching.scoring import score_projects

        with_proj = score_projects(
            {"projects": [{"name": "RAG system", "description": ["Built RAG with LangChain and FAISS vector search"]}]},
            {"required_skills": ["python", "rag"], "required_bullets": ["Experience with RAG and vector search"]},
        )
        without = score_projects({"projects": []}, {"required_skills": ["python"], "required_bullets": ["x"]})
        assert with_proj["score"] > without["score"]

    def test_breakdown_has_all_components(self):
        result = analyze_match(_profile(["python"]), "text", JD, "text")
        assert set(result["breakdown"].keys()) == set(WEIGHTS.keys())

    def test_methodology_stated(self):
        result = analyze_match(_profile(["python"]), "t", JD, "t")
        assert "docs/scoring.md" in result["methodology"]


class TestDeterminism:
    def test_same_input_same_output(self):
        a = analyze_match(_profile(["python", "pytorch"]), "resume text", JD, "jd text")
        b = analyze_match(_profile(["python", "pytorch"]), "resume text", JD, "jd text")
        assert a["overall_score"] == b["overall_score"]
        assert a["breakdown"] == b["breakdown"]
