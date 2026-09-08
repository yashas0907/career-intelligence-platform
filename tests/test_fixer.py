"""Tests: resume fixer — deterministic improvements, no-fabrication, downloads."""

import io
import json


class TestFixerDeterministic:
    def _upload(self, client, text):
        return client.post(
            "/api/resume/upload",
            files={"file": ("r.txt", text.encode(), "text/plain")},
        ).json()

    MESSY = """rohan mehta
rohan.m@gmail.com

some random intro line that goes on and on and on about being passionate regarding things and stuff which should become a bullet
TECHNICAL SKILLS
python docker kubernetes postgresql

EXPERIENCE
Software Engineer | PayU
Jan 2024 - Present
Worked on REST APIs with Java and Spring Boot improving latency by 40%
- Optimized PostgreSQL queries

PROJECTS
Chat App
Built a chat app with React
"""

    def test_fix_restructures_and_reports_changes(self, client):
        up = self._upload(client, self.MESSY)
        r = client.post(f"/api/fix/{up['resume_id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["changes"], "must report what changed"
        fixed = body["fixed_text"]
        # standardized heading present
        assert "SKILLS" in fixed or "Skills" in fixed
        # long prose line got bulletized
        assert "- some random intro line" in fixed.lower() or fixed.lower().count("- ") >= 2
        # content preserved
        assert "PayU" in fixed and "React" in fixed

    def test_no_fabrication_no_numbers_added(self, client):
        up = self._upload(client, self.MESSY)
        r = client.post(f"/api/fix/{up['resume_id']}").json()
        fixed = r["fixed_text"]
        # "40%" was already in the resume; nothing NEW like "99%" may appear
        import re

        new_pcts = {p for p in re.findall(r"(\d+)%", fixed)}
        assert new_pcts <= {"40"}, f"fabricated metrics detected: {new_pcts}"

    def test_fix_with_job_context_orders_skills(self, client, sample_texts):
        up = self._upload(client, sample_texts["resume"])
        match = client.post(
            "/api/match", json={"resume_id": up["resume_id"], "jobs": [{"description": sample_texts["jd_ml"]}]}
        ).json()
        r = client.post(f"/api/fix/{up['resume_id']}?job_id={match['job_id']}").json()
        assert r["changes"]
        # JD-relevant skill section first
        assert r["fixed_text"].index("Python") < r["fixed_text"].index("JavaScript")

    def test_fix_empty_resume_keeps_original(self, client):
        up = self._upload(client, "John Doe john@doe.com Python developer with Docker")
        r = client.post(f"/api/fix/{up['resume_id']}").json()
        assert r["fixed_text"].strip()

    def test_download_txt(self, client, sample_texts):
        up = self._upload(client, sample_texts["resume"])
        r = client.get(f"/api/fix/{up['resume_id']}/download?format=txt")
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        assert "SKILLS" in r.text or "Skills" in r.text

    def test_download_docx_is_valid_word_file(self, client, sample_texts):
        up = self._upload(client, sample_texts["resume"])
        r = client.get(f"/api/fix/{up['resume_id']}/download?format=docx")
        assert r.status_code == 200
        content = b"".join(r.iter_bytes()) if hasattr(r, "iter_bytes") else r.content
        # valid DOCX = zip with word/document.xml
        import zipfile

        assert zipfile.is_zipfile(io.BytesIO(content))
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            assert "word/document.xml" in z.namelist()

    def test_download_bad_format_422(self, client, sample_texts):
        up = self._upload(client, sample_texts["resume"])
        r = client.get(f"/api/fix/{up['resume_id']}/download?format=exe")
        assert r.status_code == 422

    def test_fix_unknown_resume_404(self, client):
        assert client.post("/api/fix/missing").status_code == 404


class TestGeminiConfig:
    def test_lenient_json_handles_gemini_quirks(self):
        """Gemini free tier occasionally returns unquoted keys / trailing commas."""
        from app.services.ai.llm import extract_json

        assert extract_json('{name: "Aarav", skills: ["python"],}') == {
            "name": "Aarav",
            "skills": ["python"],
        }
        assert extract_json('```json\n{"ok": true}\n```') == {"ok": True}
        assert extract_json('prefix text {"mid": 1} suffix') == {"mid": 1}
    def _make(self, openai_key: str, gemini_key: str):
        """Dataclass field defaults are evaluated at import time, so we pass
        keys as constructor arguments instead of touching os.environ."""
        from app.core.config import Settings

        return Settings(openai_api_key=openai_key, gemini_api_key=gemini_key)

    def test_gemini_provider_resolution(self):
        # conftest sets LLM_ENABLED=false for the whole suite, so build a copy
        # with it enabled to check the full availability chain.
        from app.core.config import Settings

        s = self._make("", "test-key")
        assert s.llm_provider == "gemini"
        assert s.effective_llm_base_url.startswith("https://generativelanguage.googleapis.com")
        assert s.effective_llm_model == s.gemini_model
        assert s.effective_llm_model.startswith("gemini-")
        s_on = Settings(openai_api_key="", gemini_api_key="test-key", llm_enabled=True)
        assert s_on.llm_available is True

    def test_openai_wins_over_gemini(self):
        s = self._make("sk-x", "gk-y")
        assert s.llm_provider == "openai"

    def test_neither_key_offline(self):
        s = self._make("", "")
        assert s.llm_provider == "none"
        assert s.llm_available is False

    def test_health_reports_provider(self, client):
        r = client.get("/api/health").json()
        assert "llm_provider" in r and r["llm_provider"] in {"openai", "gemini", "none"}
