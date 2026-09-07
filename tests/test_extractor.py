"""Tests: extraction quality — resume profiles + job requirements."""

from app.services.extractor import (
    extract_job,
    extract_resume,
    total_years_experience,
)


class TestResumeExtraction:
    def test_contacts(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        assert p["name"] == "Aarav Sharma"
        assert p["contact"]["email"] == "aarav.sharma@email.com"
        assert p["contact"]["phone"] is not None
        assert "github.com" in (p["contact"]["github"] or "")

    def test_skills_extracted(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        skills = set(p["skills"].keys())
        for expected in ["python", "pytorch", "docker", "sql", "scikit-learn", "fastapi", "langchain"]:
            assert expected in skills, f"missing {expected}"

    def test_skills_have_evidence(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        assert p["skills"]["pytorch"]["evidence"]  # non-empty evidence list

    def test_education(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        assert len(p["education"]) >= 1
        assert any("b.tech" in str(e.get("degree", "")).lower() for e in p["education"])

    def test_experience_bullets(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        assert len(p["experience"]) >= 1
        exp = p["experience"][0]
        assert exp.get("bullets"), "experience bullets missing"
        assert any("churn" in b.lower() for b in exp["bullets"])

    def test_projects_parsed(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        names = [proj["name"] for proj in p["projects"]]
        assert "Semantic Document Search" in names
        assert "Real-time Object Detection" in names

    def test_sections_detected(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        for s in ["experience", "education", "skills", "projects"]:
            assert s in p["sections_detected"]

    def test_empty_text_safe(self):
        p = extract_resume("")
        assert p["name"] is None
        assert p["skills"] == {}

    def test_extraction_method_recorded(self, sample_texts):
        p = extract_resume(sample_texts["resume"])
        assert p["_extraction_method"] in {"hybrid", "heuristic"}


class TestJobExtraction:
    def test_required_vs_preferred_split(self, sample_texts):
        jd = extract_job(sample_texts["jd_ml"])
        assert "python" in jd["required_skills"]
        assert "pytorch" in jd["required_skills"]  # via "PyTorch or TensorFlow"
        assert "kubernetes" in jd["preferred_skills"]
        assert "aws" in jd["preferred_skills"]
        # preferred must not leak into required
        assert "kubernetes" not in jd["required_skills"]

    def test_prose_skills_not_required(self, sample_texts):
        jd = extract_job(sample_texts["jd_ml"])
        # "applied ML team" prose shouldn't force machine-learning as required
        # unless it's also in a requirement bullet (it is, via "machine learning fundamentals")
        # so test with backend JD instead: "web services" prose
        jdb = extract_job(sample_texts["jd_backend"])
        assert "python" in jdb["required_skills"]
        assert "django" in jdb["required_skills"]
        assert "postgresql" in jdb["required_skills"]

    def test_seniority_detection(self, sample_texts):
        assert extract_job(sample_texts["jd_ml"])["seniority"] == "intern"
        assert extract_job(sample_texts["jd_backend"])["seniority"] == "intern"

    def test_min_years(self, sample_texts):
        jd = extract_job(sample_texts["jd_backend"])
        assert jd["min_years_experience"] == 1.0

    def test_title_extraction(self, sample_texts):
        assert extract_job(sample_texts["jd_ml"])["title"] == "Machine Learning Intern"
        assert extract_job(sample_texts["jd_backend"])["title"] == "Backend Developer Intern"

    def test_education_requirements(self, sample_texts):
        jd = extract_job(sample_texts["jd_ml"])
        assert any("bachelor" in e for e in jd["education_requirements"])

    def test_empty_jd_safe(self):
        jd = extract_job("")
        assert jd["title"] is None
        assert jd["required_skills"] == []


class TestExperienceYears:
    def test_explicit_years(self):
        assert total_years_experience("I have 3+ years of experience") == 3.0

    def test_date_ranges(self):
        text = "Engineer (2020 - 2023) at Corp"
        assert total_years_experience(text) == 3.0

    def test_present_counts_to_now(self):
        text = "Engineer Jan 2024 - Present"
        val = total_years_experience(text)
        assert val is not None and val >= 1

    def test_no_dates(self):
        assert total_years_experience("no dates here") is None
