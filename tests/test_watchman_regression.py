"""THE WATCHMAN BUG REGRESSION TESTS.

The user's report: a tech resume vs a "watchman" job posting scored 68% â€”
because requirement-less JDs got free 100% on skills (0.35 weight) and
generous neutral defaults everywhere else. These tests pin the fix:

1. Skill-sparse JDs must score LOW (relevance-based), not ~68%
2. Scores must DIFFERENTIATE: watchman vs real ML role = a huge gap
3. No-requirement JDs never collect free neutral credit (dynamic re-weighting)
4. Stopword filtering kills generic-word cosine inflation
"""
import pytest

from app.services.matching.scoring import analyze_match, compare_skills, score_semantic

WATCHMAN_JD = """Watchman / Security Guard

We are hiring a security guard for our residential complex.

Requirements:
- 8th pass or above
- Physically fit and able to stand for long hours
- Night shift availability
- Good communication skills
- Previous experience preferred

Responsibilities:
- Monitoring the premises
- Reporting to the supervisor
- Maintaining records of visitors
"""

ML_JD = """Machine Learning Intern

About the role
We are looking for a machine learning intern to join our applied ML team.

Requirements:
- Strong Python skills and experience with PyTorch or TensorFlow
- Familiarity with machine learning fundamentals and scikit-learn
- Experience building and deploying models with Docker
- Understanding of SQL and data pipelines
- Currently pursuing a Bachelor's degree in CS or related field

Nice to have:
- Experience with AWS or other cloud platforms
- NLP or computer vision project experience
"""

TECH_RESUME = """Aarav Sharma
aarav.sharma@email.com | +91 98765 43210 | github.com/aaravsharma

SUMMARY
Third-year Computer Science student passionate about machine learning and building data-driven products.

EDUCATION
B.Tech in Computer Science
Indian Institute of Technology, Bombay
2023 - 2027

TECHNICAL SKILLS
Languages: Python, C++, SQL, JavaScript
ML: PyTorch, scikit-learn, pandas, NumPy, TensorFlow
Tools: Git, Docker, Linux, Jupyter

EXPERIENCE
Machine Learning Intern | TraceAI Solutions
June 2025 - August 2025
- Built a churn prediction model using scikit-learn, improving recall by 12% over baseline
- Deployed a FastAPI service on AWS with Docker, serving 5k requests/day

PROJECTS
Semantic Document Search
- Built a RAG pipeline with LangChain, Hugging Face embeddings and FAISS vector search over 10k documents

Real-time Object Detection
- Fine-tuned YOLOv8 on a custom dataset of 3k images; 42 FPS on edge GPU
"""


def _profile_from_resume():
    from app.services.extractor import extract_resume

    return extract_resume(TECH_RESUME)


def _jd(text):
    """Parsed JD exactly like the real pipeline produces."""
    from app.services.extractor import extract_job

    return extract_job(text)


class TestWatchmanBug:
    """The exact reported scenario: tech resume vs watchman JD."""

    def test_watchman_scores_low_not_68(self):
        profile = _profile_from_resume()
        result = analyze_match(profile, TECH_RESUME, _jd(WATCHMAN_JD), WATCHMAN_JD)
        assert result["overall_score"] < 0.40, (
            f"watchman scored {result['overall_score']:.0%} â€” requirement-less JDs "
            f"must NOT collect free neutral credit"
        )

    def test_watchman_vs_ml_gap_is_huge(self):
        profile = _profile_from_resume()
        watchman = analyze_match(profile, TECH_RESUME, _jd(WATCHMAN_JD), WATCHMAN_JD)
        ml = analyze_match(profile, TECH_RESUME, _jd(ML_JD), ML_JD)
        gap = ml["overall_score"] - watchman["overall_score"]
        assert gap > 0.25, f"only {gap:.0%} gap between watchman and ML role — scores not differentiating"
        assert ml["overall_score"] > 0.50

    def test_no_free_100_percent_skills(self):
        profile = _profile_from_resume()
        res = compare_skills(profile, {}, WATCHMAN_JD)
        assert res["score"] < 1.0, "skills score must never default to 100%"
        assert res.get("informative") is False, "skill-sparse JD -> non-informative component"

    def test_weights_redistributed_for_sparse_jd(self):
        profile = _profile_from_resume()
        watchman = analyze_match(profile, TECH_RESUME, _jd(WATCHMAN_JD), WATCHMAN_JD)
        assert watchman["weights_redistributed"] is True
        # non-informative components got zero weight
        assert watchman["weights"]["skills"] == 0.0
        assert watchman["weights"]["education"] == 0.0
        # informative ones absorbed the weight (responsibilities + projects + semantic)
        assert watchman["weights"]["semantic"] > 0.3
        assert watchman["weights"]["projects"] > 0.2
        assert watchman["weights"]["experience"] > 0.3

    def test_full_jd_keeps_base_weights(self):
        profile = _profile_from_resume()
        ml = analyze_match(profile, TECH_RESUME, _jd(ML_JD), ML_JD)
        assert ml["weights_redistributed"] is False
        assert abs(sum(ml["weights"].values()) - 1.0) < 1e-6
        assert abs(ml["weights"]["skills"] - 0.35) < 1e-6, "fully-specified JD keeps exact base weights"

    def test_sparse_jd_explanation_is_transparent(self):
        profile = _profile_from_resume()
        watchman = analyze_match(profile, TECH_RESUME, _jd(WATCHMAN_JD), WATCHMAN_JD)
        exp = watchman["components"]["skills"].get("note", "")
        assert "No recognizable skill requirements" in exp
        assert watchman["components"]["skills"]["informative"] is False


class TestScoreDifferentiation:
    """Different JDs must give DIFFERENT scores â€” no more one-size-fits-all 68%."""

    def test_three_different_jds_three_different_scores(self):
        profile = _profile_from_resume()
        scores = {
            "ml": analyze_match(profile, TECH_RESUME, _jd(ML_JD), ML_JD)["overall_score"],
            "watchman": analyze_match(profile, TECH_RESUME, _jd(WATCHMAN_JD), WATCHMAN_JD)["overall_score"],
            "empty_jd": analyze_match(profile, TECH_RESUME, {}, "Join our team. We work on things.")["overall_score"],
        }
        vals = sorted(set(scores.values()))
        assert len(vals) == 3, f"scores collapsed together: {scores}"
        assert scores["ml"] - scores["watchman"] > 0.3

    def test_semantic_low_for_unrelated(self):
        res = score_semantic(TECH_RESUME, {}, WATCHMAN_JD)
        assert res["score"] < 0.30, f"watchman semantic {res['score']:.0%} still too generous"

    def test_semantic_not_inflated_by_stopwords(self):
        # a JD of pure function words must score ~0 semantic
        res = score_semantic(TECH_RESUME, {}, "We are hiring for the role in the team at the company")
        assert res["score"] < 0.20


class TestStopwordFiltering:
    def test_stopwords_removed_from_tokens(self):
        from app.services.ai.embeddings import _tokens

        toks = _tokens("The Python and the Docker are in the team")
        assert "python" in toks and "docker" in toks
        assert "the" not in toks and "and" not in toks and "are" not in toks and "in" not in toks

    def test_short_words_filtered(self):
        from app.services.ai.embeddings import _tokens

        toks = _tokens("I am at it on of to a")
        assert toks == [], "single-char and stopword tokens must all be filtered"



class TestWatchmanAPI:
    """The user's exact scenario through the real API."""

    def test_watchman_low_through_api(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        r = client.post(
            "/api/jobs/analyze",
            json={
                "resume_id": up["resume_id"],
                "jobs": [
                    {"title": "watchman", "description": WATCHMAN_JD},
                    {"title": "ml", "description": ML_JD},
                ],
            },
        )
        assert r.status_code == 200
        results = r.json()
        w = next(x for x in results if x["job_title"] == "watchman")
        m = next(x for x in results if x["job_title"] == "ml")
        assert w["overall_score"] < 0.40, f"watchman {w['overall_score']:.0%} through API"
        assert m["overall_score"] > w["overall_score"]
        assert w["weights_redistributed"] is True
        assert m["weights_redistributed"] is False
        assert "No recognizable skill requirements" in w["explanation"]
