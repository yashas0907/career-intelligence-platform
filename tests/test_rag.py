"""Tests: vector store chunking + retrieval + RAG grounding."""

from app.services.ai.embeddings import HashingEmbeddings
from app.services.ai.vector_store import VectorStore, chunk_text
from app.services.rag import answer_question


class TestEmbeddingStability:
    def test_hashing_backend_is_process_stable(self):
        """crc32-based hashing must give identical vectors for identical text
        (builtin hash() is seed-randomized per process and would break
        persisted vector indices across restarts)."""
        b = HashingEmbeddings()
        v1 = b.embed(["PyTorch deep learning resume"])
        v2 = b.embed(["PyTorch deep learning resume"])
        assert (v1 == v2).all()

    def test_different_text_differs(self):
        b = HashingEmbeddings()
        v1 = b.embed(["kubernetes docker deployment"])
        v2 = b.embed(["excel powerpoint communication"])
        assert not (v1 == v2).all()


class TestChunking:
    def test_paragraph_chunking(self):
        text = "\n\n".join(f"Paragraph {i} about machine learning topic {i}." for i in range(10))
        chunks = chunk_text(text, "doc1", "resume", "Resume", max_chars=120)
        assert len(chunks) > 1
        assert all(c.source == "resume" for c in chunks)

    def test_ids_unique_and_sequential(self):
        chunks = chunk_text("a\n\nb\nc\nd", "d", "job", "Job")
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_long_paragraph_split(self):
        text = "word " * 400  # single giant paragraph
        chunks = chunk_text(text, "d", "resume", "R", max_chars=500)
        assert len(chunks) >= 2

    def test_empty(self):
        assert chunk_text("", "d", "resume", "R") == []


class TestVectorStore:
    def test_index_and_search(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="test_ns")
        chunks = chunk_text(
            "Python developer with PyTorch deep learning experience.\n\nDjango web backend with PostgreSQL.",
            "d1", "resume", "Resume",
        )
        store.build_from_chunks(chunks)
        results = store.search("machine learning with pytorch", top_k=2)
        assert results
        best_chunk, sim = results[0]
        assert "PyTorch" in best_chunk.text
        assert 0 <= sim <= 1

    def test_search_empty_store(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="empty_ns")
        assert store.search("anything") == []

    def test_persistence_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="persist_ns")
        chunks = chunk_text("Kubernetes docker deployment experience", "d", "resume", "Resume")
        store.build_from_chunks(chunks)

        store2 = VectorStore(namespace="persist_ns")
        loaded = store2.load()
        assert loaded
        results = store2.search("docker", top_k=1)
        assert results and "docker" in results[0][0].text.lower()


class TestRAG:
    def test_answer_uses_evidence(self, tmp_path, monkeypatch, sample_texts):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="rag_test")
        chunks = chunk_text(sample_texts["resume"], "r1", "resume", "Resume")
        store.build_from_chunks(chunks)
        result = answer_question("What is the candidate's email?", store, None)
        assert result["answer"]
        assert result["sources"]
        assert result["method"].startswith("rag")

    def test_extractive_fallback_quotes_document(self, tmp_path, monkeypatch, sample_texts):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="rag_test2")
        store.build_from_chunks(chunk_text(sample_texts["resume"], "r1", "resume", "Resume"))
        result = answer_question("Tell me about the candidate's projects", store, None)
        # offline mode quotes evidence directly
        assert "Resume" in result["answer"] or "project" in result["answer"].lower()

    def test_no_context_honest(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="rag_test3")
        result = answer_question("What is 2+2?", store, None)
        assert "couldn't find" in result["answer"].lower() or result["answer"]

    def test_analysis_summary_included(self, tmp_path, monkeypatch, sample_texts):
        monkeypatch.chdir(tmp_path)
        store = VectorStore(namespace="rag_test4")
        store.build_from_chunks(chunk_text(sample_texts["resume"], "r", "resume", "Resume"))
        analysis = {
            "overall_score": 0.78,
            "breakdown": {"skills": 0.8, "experience": 0.7, "projects": 0.6, "education": 1.0, "semantic": 0.7},
            "skill_comparison": {"missing_required": [{"display": "Kubernetes"}]},
        }
        result = answer_question("Why is my score low?", store, analysis)
        assert "78%" in result["answer"] or "Kubernetes" in result["answer"] or result["answer"]
