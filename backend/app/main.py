"""FastAPI application factory with observability middleware."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes as api_routes
from app.core.config import settings
from app.core.database import init_db
from app.core.rate_limit import RateLimiter

logger = logging.getLogger("app.main")

# Only guard expensive endpoints; static/health stay unthrottled.
_RATE_LIMITED_PATHS = {"/api/resume/upload", "/api/jobs/analyze", "/api/chat", "/api/match"}
_rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_per_minute)


def create_app() -> FastAPI:
    init_db()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "AI-powered resume ↔ job matching with deterministic, explainable scoring, "
            "ATS-oriented analysis, job ranking and a RAG career assistant."
        ),
        docs_url=f"{settings.api_prefix}/docs",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error on %s %s (req=%s): %s", request.method, request.url.path, request_id, exc)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error.", "error_code": "internal_error", "request_id": request_id},
            )
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        logger.info(
            "%s %s -> %s (%.1f ms) req=%s",
            request.method, request.url.path, response.status_code, duration_ms, request_id,
        )
        if duration_ms > 3000:
            logger.warning("Slow request %s %s took %.1f ms", request.method, request.url.path, duration_ms)
        return response

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        path = request.url.path
        # strip the /stream suffix so streamed variants share the budget
        base_path = path.replace("/stream", "", 1) if path.endswith("/stream") else path
        if request.method == "POST" and base_path in _RATE_LIMITED_PATHS:
            client = request.client.host if request.client else "unknown"
            allowed, retry_after = _rate_limiter.check(f"{client}:{base_path}")
            if not allowed:
                logger.warning("Rate limited %s on %s (retry in %.1fs)", client, path, retry_after)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Too many requests. Retry in {retry_after:.0f}s.",
                        "error_code": "rate_limited",
                    },
                    headers={"Retry-After": str(int(retry_after) + 1)},
                )
        return await call_next(request)

    # Validation errors -> consistent error envelope
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", [])[1:]) or "body"
        msg = first.get("msg", "Invalid request")
        return JSONResponse(
            status_code=422,
            content={"detail": f"{loc}: {msg}", "error_code": "validation_error"},
        )

    for router in (
        api_routes.health.router,
        api_routes.resumes.router,
        api_routes.jobs.router,
        api_routes.chat.router,
        api_routes.recommendations.router,
        api_routes.demo.router,
        api_routes.fixer.router,
    ):
        app.include_router(router, prefix=settings.api_prefix)

    # Serve the built frontend if present (single-service deployments).
    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
        logger.info("Serving frontend from %s", static_dir)

    return app


app = create_app()
