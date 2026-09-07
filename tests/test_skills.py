"""Tests: skill taxonomy + normalization engine."""

import pytest

from app.services.skills import (
    TAXONOMY,
    normalize_skill,
    related_skills,
    skill_category,
    skill_display,
)
from app.services.skills import SkillNormalizer


class TestNormalization:
    def test_case_insensitive(self):
        assert normalize_skill("PyTorch") == "pytorch"
        assert normalize_skill("PYTORCH") == "pytorch"

    def test_aliases(self):
        assert normalize_skill("sklearn") == "scikit-learn"
        assert normalize_skill("k8s") == "kubernetes"
        assert normalize_skill("golang") == "go"
        assert normalize_skill("JS") == "javascript"
        assert normalize_skill("py") == "python"

    def test_whitespace_normalized(self):
        assert normalize_skill("  pytorch  ") == "pytorch"

    def test_unknown_returns_none(self):
        assert normalize_skill("quantum blockchain") is None
        assert normalize_skill("") is None
        assert normalize_skill(None) is None

    def test_every_taxonomy_id_is_self_normalizing(self):
        for cid in TAXONOMY:
            assert normalize_skill(cid) == cid, f"{cid} does not normalize to itself"


class TestMetadata:
    def test_categories_valid(self):
        valid = {"language", "framework", "library", "tool", "cloud", "database", "concept", "domain", "soft"}
        for cid, meta in TAXONOMY.items():
            assert meta.category in valid, f"{cid}: bad category {meta.category}"

    def test_related_graph_bidirectional(self):
        assert "tensorflow" in related_skills("pytorch")
        assert "pytorch" in related_skills("tensorflow")

    def test_display_names(self):
        assert skill_display("pytorch") == "PyTorch"
        assert skill_display("scikit-learn") == "scikit-learn"

    def test_unknown_display_falls_back(self):
        assert skill_display("customskillname") == "Customskillname"

    def test_category_fallback(self):
        assert skill_category("not-a-skill") == "concept"


class TestExtraction:
    def test_extract_finds_skills_with_evidence(self):
        n = SkillNormalizer()
        found = n.extract("Built models with PyTorch and deployed on Kubernetes.")
        assert set(found.keys()) == {"pytorch", "kubernetes"}
        assert found["pytorch"][0]  # evidence snippet non-empty

    def test_extract_no_false_boundary_hits(self):
        n = SkillNormalizer()
        found = n.extract("I love javascripting and pythonic code")
        assert "javascript" not in found
        assert "python" not in found  # 'pythonic' should NOT match python

    def test_alias_in_text(self):
        n = SkillNormalizer()
        found = n.extract("Experience with sklearn pipelines")
        assert "scikit-learn" in found

    def test_empty_text(self):
        assert SkillNormalizer().extract("") == {}

    def test_dedup_canonical(self):
        n = SkillNormalizer()
        found = n.extract("python, PYTHON and Python are the same")
        assert list(found.keys()) == ["python"]
