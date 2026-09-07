"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobIn(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    company: str | None = Field(default=None, max_length=200)
    description: str = Field(min_length=30, max_length=40000)

    @field_validator("description")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Job description cannot be blank")
        return v.strip()


class JobCreateResponse(BaseModel):
    job_id: str
    title: str
    parsed: dict[str, Any]
    extraction_method: str


class JobsAnalyzeRequest(BaseModel):
    resume_id: str
    jobs: list[JobIn] = Field(min_length=1, max_length=10)


class SkillMatchDetailSchema(BaseModel):
    skill: str
    display: str
    category: str
    requirement: str
    status: str
    score: float
    via: str | None = None
    evidence: list[str] = []


class RecommendationSchema(BaseModel):
    type: str
    priority: str
    title: str
    detail: str
    signal: str


class RankedJobSchema(BaseModel):
    rank: int
    job_id: str
    title: str
    overall_score: float
    breakdown: dict[str, float]
    reasons: list[str]


class AnalysisResponse(BaseModel):
    analysis_id: str
    resume_id: str
    job_id: str
    job_title: str
    overall_score: float
    breakdown: dict[str, float]
    weights: dict[str, float]
    components: dict[str, Any]
    skill_comparison: dict[str, Any]
    ats: dict[str, Any]
    recommendations: list[dict[str, Any]]
    explanation: str
    explanation_method: str


class ChatRequest(BaseModel):
    analysis_id: str | None = None
    question: str = Field(min_length=2, max_length=2000)

    @field_validator("question")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip()


class ChatSourceSchema(BaseModel):
    source: str
    label: str
    similarity: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSourceSchema]
    method: str
    analysis_id: str | None = None


class ResumeProfileResponse(BaseModel):
    resume_id: str
    filename: str
    parse_method: str
    extraction_status: str
    profile: dict[str, Any]
    warnings: list[str]
    created_at: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    llm_provider: str = "none"
    embedding_backend: str
    database: str
