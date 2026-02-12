#backend/app/main.py
from __future__ import annotations
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.core.config import CORS_ORIGINS, STATIC_DIR, LOG_LEVEL
from backend.app.routes_analysis import router as analysis_router
from backend.app.routes_pages import router as pages_router
from backend.app.routes_health import router as health_router
from backend.app.routes_oov import router as oov_router
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Emotion Diary Analyzer",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(pages_router)
    app.include_router(analysis_router)
    app.include_router(health_router)
    app.include_router(oov_router)

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

    logger.info("FastAPI 앱이 초기화되었습니다. (log_level=%s)", LOG_LEVEL)
    return app

app = create_app()
