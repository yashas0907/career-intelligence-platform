"""Tests: API endpoints — full user journey + edge cases."""

import io


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["llm_available"] is False  # test env has no key
        assert body["embedding_backend"].startswith("hashing")


class TestResumeUpload:
    def test_upload_txt(self, client, sample_texts):
        r = client.post(
            "/api/resume/upload",
            files={"file": ("resume.txt", sample_texts["resume"].encode(), "text/plain")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["resume_id"]
        assert body["extraction_status"] in {"ok", "fallback"}
        assert "python" in body["profile"]["skills"]

    def test_upload_rejects_exe(self, client):
        r = client.post(
            "/api/resume/upload",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        )
        assert r.status_code == 422
        assert "Unsupported" in r.json()["detail"]

    def test_upload_rejects_empty(self, client):
        r = client.post(
            "/api/resume/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert r.status_code == 422

    def test_upload_rejects_fake_docx(self, client):
        r = client.post(
            "/api/resume/upload",
            files={"file": ("fake.docx", b"not a zip", "application/vnd...")},
        )
        assert r.status_code == 422
        assert "DOCX" in r.json()["detail"]

    def test_get_resume_after_upload(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("resume.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        r = client.get(f"/api/resume/{up['resume_id']}")
        assert r.status_code == 200
        assert r.json()["filename"] == "resume.txt"

    def test_get_missing_resume_404(self, client):
        r = client.get("/api/resume/doesnotexist")
        assert r.status_code == 404


class TestJobAnalysis:
    def test_full_flow(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("resume.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()

        r = client.post(
            "/api/jobs/analyze",
            json={
                "resume_id": up["resume_id"],
                "jobs": [
                    {"title": "ML Intern", "description": sample_texts["jd_ml"]},
                    {"title": "Backend Intern", "description": sample_texts["jd_backend"]},
                ],
            },
        )
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 2
        for res in results:
            assert 0.0 <= res["overall_score"] <= 1.0
            assert set(res["breakdown"]) == {"skills", "experience", "projects", "education", "semantic"}
            assert res["skill_comparison"]["strong"]
            assert res["explanation"]
            assert res["ats"]["overall"] > 0
            assert res["recommendations"]

        # ML role should fit better than backend for this resume
        ml = next(x for x in results if x["job_title"] == "ML Intern")
        be = next(x for x in results if x["job_title"] == "Backend Intern")
        assert ml["overall_score"] > be["overall_score"]

    def test_single_match_alias(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        r = client.post(
            "/api/match",
            json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]},
        )
        assert r.status_code == 200
        assert r.json()["analysis_id"]

    def test_analysis_retrieval(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        match = client.post(
            "/api/match",
            json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]},
        ).json()
        r = client.get(f"/api/analysis/{match['analysis_id']}")
        assert r.status_code == 200
        assert r.json()["analysis_id"] == match["analysis_id"]

    def test_unknown_resume_404(self, client, sample_texts):
        r = client.post(
            "/api/jobs/analyze",
            json={"resume_id": "missing", "jobs": [{"description": sample_texts["jd_ml"]}]},
        )
        assert r.status_code == 404

    def test_short_description_422(self, client, uploaded_resume):
        r = client.post(
            "/api/jobs/analyze",
            json={"resume_id": uploaded_resume["resume_id"], "jobs": [{"description": "too short"}]},
        )
        assert r.status_code == 422

    def test_empty_jobs_list_422(self, client, uploaded_resume):
        r = client.post(
            "/api/jobs/analyze",
            json={"resume_id": uploaded_resume["resume_id"], "jobs": []},
        )
        assert r.status_code == 422


class TestRecommendationsEndpoint:
    def test_ranking(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        client.post(
            "/api/jobs/analyze",
            json={
                "resume_id": up["resume_id"],
                "jobs": [
                    {"title": "ML", "description": sample_texts["jd_ml"]},
                    {"title": "BE", "description": sample_texts["jd_backend"]},
                ],
            },
        )
        r = client.get(f"/api/recommendations?resume_id={up['resume_id']}")
        assert r.status_code == 200
        ranked = r.json()
        assert len(ranked) == 2
        assert ranked[0]["overall_score"] >= ranked[1]["overall_score"]
        assert ranked[0]["rank"] == 1
        assert ranked[0]["reasons"]

    def test_missing_resume_param(self, client):
        r = client.get("/api/recommendations")
        assert r.status_code == 422


class TestChat:
    def test_chat_grounded_answer(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        client.post(
            "/api/jobs/analyze",
            json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]},
        )
        r = client.post("/api/chat", json={"question": "What skills am I missing for this role?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]
        assert body["method"] in {"rag+llm", "rag+extractive", "none"}
        assert body["sources"]

    def test_chat_history_persisted(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        match = client.post(
            "/api/match",
            json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]},
        ).json()
        client.post("/api/chat", json={"analysis_id": match["analysis_id"], "question": "Which projects are most relevant?"})
        r = client.get(f"/api/chat/history?analysis_id={match['analysis_id']}")
        assert r.status_code == 200
        msgs = r.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_chat_empty_question_422(self, client):
        r = client.post("/api/chat", json={"question": ""})
        assert r.status_code == 422


class TestHistory:
    def test_resume_history(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        client.post(
            "/api/jobs/analyze",
            json={"resume_id": up["resume_id"], "jobs": [{"title": "ML", "description": sample_texts["jd_ml"]}]},
        )
        r = client.get(f"/api/resume/{up['resume_id']}/history")
        assert r.status_code == 200
        assert len(r.json()["analyses"]) == 1


class TestErrorEnvelope:
    def test_validation_error_format(self, client):
        r = client.post("/api/jobs/analyze", json={"resume_id": "x", "jobs": [{"description": "no"}]})
        assert r.status_code == 422
        body = r.json()
        assert "detail" in body
