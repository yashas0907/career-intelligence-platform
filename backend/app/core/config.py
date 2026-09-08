"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Optional dependency: python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings. Values come from environment variables with sane defaults."""

    # --- Application ---
    app_name: str = os.getenv("APP_NAME", "AI Career Intelligence Platform")
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = _get_bool("DEBUG", True)
    api_prefix: str = os.getenv("API_PREFIX", "/api")

    # --- CORS ---
    cors_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
            ).split(",")
            if o.strip()
        )
    )

    # --- Database ---
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Path(__file__).resolve().parents[2] / 'backend_data' / 'career_intel.db'}",
    )

    # --- Uploads / parsing ---
    max_upload_mb: int = _get_int("MAX_UPLOAD_MB", 10)
    allowed_extensions: tuple = field(
        default_factory=lambda: tuple(
            e.strip().lower() for e in os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt").split(",")
        )
    )
    max_resume_chars: int = _get_int("MAX_RESUME_CHARS", 80_000)
    max_jd_chars: int = _get_int("MAX_JD_CHARS", 40_000)

    # --- Embeddings ---
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "auto")  # openai | st | auto
    st_model: str = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- LLM ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    # FREE option: Google Gemini free tier (https://aistudio.google.com â€” no card).
    # Uses Gemini's OpenAI-compatible endpoint, so the same client works.
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    llm_enabled: bool = _get_bool("LLM_ENABLED", True)  # auto-fallback to heuristic if no key
    llm_timeout_s: int = _get_int("LLM_TIMEOUT_S", 45)
    llm_max_tokens: int = _get_int("LLM_MAX_TOKENS", 1500)

    # --- Security ---
    llm_guard_max_chars: int = _get_int("LLM_GUARD_MAX_CHARS", 12_000)
    rate_limit_per_minute: int = _get_int("RATE_LIMIT_PER_MINUTE", 30)

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file: str = os.getenv(
        "LOG_FILE",
        str(Path(__file__).resolve().parents[2] / "backend_data" / "app.log"),
    )

    @property
    def llm_provider(self) -> str:
        """openai | gemini | none (provider precedence: explicit OpenAI > free Gemini)."""
        if self.openai_api_key.strip():
            return "openai"
        if self.gemini_api_key.strip():
            return "gemini"
        return "none"

    @property
    def effective_llm_api_key(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return ""

    @property
    def effective_llm_base_url(self) -> str:
        if self.llm_provider == "openai" and self.openai_base_url:
            return self.openai_base_url
        if self.llm_provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return self.openai_base_url

    @property
    def effective_llm_model(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.openai_model

    @property
    def llm_available(self) -> bool:
        return self.llm_enabled and self.llm_provider != "none"

    def validate(self) -> None:
        valid_backends = {"openai", "st", "hashing", "auto"}
        if self.embedding_backend not in valid_backends:
            raise ValueError(f"EMBEDDING_BACKEND must be one of {valid_backends}")


def get_settings() -> Settings:
    s = Settings()
    s.validate()
    return s


settings = get_settings()

# --- Logging setup (avoid reconfiguring on hot reload) ---
if not logging.getLogger("app").handlers:
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))
    except OSError:  # pragma: no cover
        pass
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("app.config")
logger.info(
    "Config loaded: env=%s llm=%s(%s) embedding=%s",
    settings.environment,
    settings.llm_provider,
    settings.effective_llm_model if settings.llm_available else "offline",
    settings.embedding_backend,
)
