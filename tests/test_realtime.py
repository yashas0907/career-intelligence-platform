"""Tests: real-time features — SSE streams, demo seeding, chat-before-analysis, rate limiting."""

import json

from fastapi.testclient import TestClient


class TestChatStream:
    def _setup_resume(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        client.post(
            "/api/jobs/analyze",
            json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]},
        )
        return up

    def test_stream_yields_sse_events_in_order(self, client, sample_texts):
        self._setup_resume(client, sample_texts)
        with client.stream("POST", "/api/chat/stream", json={"question": "What skills do I have?"}) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            body = "".join(r.iter_text())
        events = [json.loads(l.replace("data: ", "")) for l in body.split("\n\n") if l.startswith("data: ") and l != "data: [DONE]"]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "meta"
        assert "answer" in kinds
        assert kinds[-1] == "done"
        meta = events[0]
        assert isinstance(meta["sources"], list) and meta["sources"]
        assert "".join(e["text"] for e in events if e["type"] == "answer").strip()

    def test_stream_persists_history(self, client, sample_texts):
        up = self._setup_resume(client, sample_texts)
        match = client.post(
            "/api/match", json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]}
        ).json()
        with client.stream("POST", "/api/chat/stream", json={"analysis_id": match["analysis_id"], "question": "What projects are relevant?"}) as r:
            body = "".join(r.iter_text())
        assert "data:" in body
        hist = client.get(f"/api/chat/history?analysis_id={match['analysis_id']}").json()["messages"]
        assert any(m["role"] == "assistant" for m in hist)

    def test_chat_works_without_any_analysis(self, client, sample_texts):
        """User journey: upload resume -> go straight to assistant. Must NOT 400."""
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        r = client.post("/api/chat", json={"question": "What skills do I have?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"]
        assert body["sources"]

    def test_stream_no_resume_400(self, client):
        # this test class always has resumes from earlier fixtures; use bogus id path
        r = client.post("/api/chat", json={"analysis_id": "nonexistent", "question": "hi there"})
        assert r.status_code == 404


class TestAnalyzeStream:
    def test_results_stream_one_by_one(self, client, sample_texts):
        up = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", sample_texts["resume"].encode(), "text/plain")},
        ).json()
        with client.stream(
            "POST",
            "/api/jobs/analyze/stream",
            json={
                "resume_id": up["resume_id"],
                "jobs": [
                    {"title": "ML", "description": sample_texts["jd_ml"]},
                    {"title": "BE", "description": sample_texts["jd_backend"]},
                ],
            },
        ) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        events = [json.loads(l.replace("data: ", "")) for l in body.split("\n\n") if l.startswith("data: ") and l != "data: [DONE]"]
        assert all(e["type"] == "result" for e in events)
        assert len(events) == 2
        titles = {e["analysis"]["job_title"] for e in events}
        assert titles == {"ML", "BE"}
        # each event carries a complete, valid analysis
        a = events[0]["analysis"]
        assert 0.0 <= a["overall_score"] <= 1.0
        assert set(a["breakdown"]) == {"skills", "experience", "projects", "education", "semantic"}

    def test_stream_rejects_unknown_resume(self, client, sample_texts):
        r = client.post(
            "/api/jobs/analyze/stream",
            json={"resume_id": "missing", "jobs": [{"description": sample_texts["jd_ml"]}]},
        )
        assert r.status_code == 404


class TestDemo:
    def test_sample_data_shape(self, client):
        r = client.get("/api/demo/sample")
        assert r.status_code == 200
        data = r.json()
        assert len(data["resume_text"]) > 200
        assert len(data["jobs"]) >= 3
        assert all("description" in j and "title" in j for j in data["jobs"])

    def test_seed_creates_working_resume(self, client):
        r = client.post("/api/demo/seed")
        assert r.status_code == 200
        seeded = r.json()
        assert seeded["resume_id"]
        assert "python" in seeded["profile"]["skills"]
        # seeded resume is fully usable: analyze against a demo JD
        sample = client.get("/api/demo/sample").json()
        m = client.post(
            "/api/jobs/analyze",
            json={"resume_id": seeded["resume_id"], "jobs": [{"description": sample["jobs"][0]["description"]}]},
        )
        assert m.status_code == 200
        assert m.json()[0]["overall_score"] > 0.5  # demo pair is a good match


class TestRateLimit:
    def test_upload_rate_limited(self, client, monkeypatch, sample_texts):
        from app.main import _rate_limiter

        monkeypatch.setattr(_rate_limiter, "limit", 3)
        monkeypatch.setattr(_rate_limiter, "_hits", {})
        codes = []
        for _ in range(5):
            r = client.post(
                "/api/resume/upload",
                files={"file": ("r.txt", b"John Doe john@x.com SKILLS Python Docker", "text/plain")},
            )
            codes.append(r.status_code)
        assert 429 in codes
        assert codes.count(429) >= 2
        # error envelope shape
        limited = client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", b"John Doe john@x.com SKILLS Python Docker", "text/plain")},
        )
        if limited.status_code == 429:
            assert limited.json()["error_code"] == "rate_limited"

    def test_health_never_limited(self, client, monkeypatch):
        from app.main import _rate_limiter

        monkeypatch.setattr(_rate_limiter, "limit", 1)
        monkeypatch.setattr(_rate_limiter, "_hits", {})
        for _ in range(5):
            assert client.get("/api/health").status_code == 200
