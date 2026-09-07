"""Tests: ATS analyzer + recommendation engine."""

from app.services.ats import analyze_ats
from app.services.recommendations import build_recommendations, rank_jobs
from app.services.matching.scoring import analyze_match


class TestATS:
    def _profile(self, skills, sections, contact=True):
        return {
            "skills": {s: {"evidence": ["x"]} for s in skills},
            "sections_detected": sections,
            "contact": {"email": "a@b.c", "phone": "123-456-7890"} if contact else {},
        }

    def test_perfect_resume_scores_high(self, sample_texts):
        from app.services.extractor import extract_resume

        p = extract_resume(sample_texts["resume"])
        from app.services.extractor import extract_job

        jd = extract_job(sample_texts["jd_ml"])
        ats = analyze_ats(p, sample_texts["resume"], jd, "txt")
        assert ats["overall"] > 0.6

    def test_missing_sections_lower_score(self):
        base = analyze_ats(self._profile(["python"], ["skills"]), "text", {"required_skills": ["python"]}, "txt")
        full = analyze_ats(self._profile(["python"], ["experience", "education", "skills", "projects"]), "text", {"required_skills": ["python"]}, "txt")
        assert full["subscores"]["sections"] > base["subscores"]["sections"]

    def test_keyword_coverage_reflected(self):
        low = analyze_ats(self._profile([], ["skills"]), "t", {"required_skills": ["python", "docker", "aws"]}, "txt")
        high = analyze_ats(self._profile(["python", "docker", "aws"], ["skills"]), "t", {"required_skills": ["python", "docker", "aws"]}, "txt")
        assert high["subscores"]["keywords"] == 1.0
        assert low["subscores"]["keywords"] == 0.0

    def test_contact_missing_flagged(self):
        ats = analyze_ats(self._profile(["python"], ["skills"], contact=False), "t", {"required_skills": []}, "txt")
        assert ats["subscores"]["contact"] == 0.0

    def test_disclaimer_present(self, sample_texts):
        ats = analyze_ats(self._profile([], []), sample_texts["resume"], {"required_skills": []}, "txt")
        assert "not a replica" in ats["disclaimer"].lower()

    def test_recommendations_generated_when_gaps(self):
        ats = analyze_ats(self._profile([], ["skills"]), "t", {"required_skills": ["python", "docker"]}, "txt")
        assert any("python" in r.lower() for r in ats["recommendations"])


class TestRecommendations:
    def _result(self, cand_skills, jd_req, jd_pref):
        jd = {"required_skills": jd_req, "preferred_skills": jd_pref, "required_bullets": jd_req}
        profile = {"skills": {s: {"evidence": ["x"]} for s in cand_skills}, "projects": [], "experience": [], "education": []}
        return analyze_match(profile, "resume", jd, "jd"), profile, jd

    def test_missing_required_becomes_learn_rec(self):
        result, profile, jd = self._result(["python"], ["python", "kubernetes"], [])
        recs = build_recommendations(result, profile, jd)
        assert any(r["type"] == "learn" and "Kubernetes" in r["title"] for r in recs)
        assert all(r["priority"] in {"high", "medium", "low"} for r in recs)

    def test_no_invention_language(self):
        result, profile, jd = self._result([], ["kubernetes"], [])
        recs = build_recommendations(result, profile, jd)
        text = " ".join(r["detail"] for r in recs).lower()
        assert "add to resume" in text or "build" in text  # suggestions are learn/build framed

    def test_recommendations_cite_signal(self):
        result, profile, jd = self._result(["python"], ["python", "spark"], [])
        recs = build_recommendations(result, profile, jd)
        assert all(r.get("signal") for r in recs)

    def test_priorities_sorted(self):
        result, profile, jd = self._result([], ["kubernetes", "spark"], ["graphql"])
        recs = build_recommendations(result, profile, jd)
        order = {"high": 0, "medium": 1, "low": 2}
        priorities = [order[r["priority"]] for r in recs]
        assert priorities == sorted(priorities)

    def test_empty_projects_triggers_build_rec(self):
        result, profile, jd = self._result(["python"], ["python"], [])
        recs = build_recommendations(result, profile, jd)
        # projects score low with no projects
        assert any(r["signal"] == "projects:empty" for r in recs)


class TestRanking:
    def test_ranks_higher_match_first(self):
        items = [
            {
                "job_id": "low",
                "job_title": "Backend Intern",
                "overall_score": 0.55,
                "components": {"skills": {"score": 0.5}},
                "breakdown": {"skills": 0.5, "experience": 0.5, "projects": 0.5, "education": 0.8, "semantic": 0.5},
            },
            {
                "job_id": "high",
                "job_title": "ML Intern",
                "overall_score": 0.85,
                "components": {"skills": {"score": 0.9}},
                "breakdown": {"skills": 0.9, "experience": 0.7, "projects": 0.9, "education": 1.0, "semantic": 0.8},
            },
        ]
        ranked = rank_jobs(items)
        assert ranked[0]["job_id"] == "high"
        assert ranked[0]["rank"] == 1
        assert ranked[0]["reasons"]  # explanation present

    def test_tiebreak_by_skills(self):
        items = [
            {"job_id": "a", "job_title": "A", "overall_score": 0.7, "components": {"skills": {"score": 0.6}},
             "breakdown": {"skills": 0.6, "experience": 0.5, "projects": 0.5, "education": 0.8, "semantic": 0.6}},
            {"job_id": "b", "job_title": "B", "overall_score": 0.7, "components": {"skills": {"score": 0.8}},
             "breakdown": {"skills": 0.8, "experience": 0.5, "projects": 0.5, "education": 0.8, "semantic": 0.6}},
        ]
        ranked = rank_jobs(items)
        assert ranked[0]["job_id"] == "b"
