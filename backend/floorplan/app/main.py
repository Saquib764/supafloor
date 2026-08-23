"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router as floorplan_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from extraction.florence import warmup

    warmup()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="FloorPlan Dataset Engine",
        version=settings.service_version,
        description="Deterministic residential floor plan generation, validation, rendering, and scoring.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(floorplan_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.service_name}

    @app.get("/status")
    def status() -> dict[str, str]:
        return {"status": "ok", "service": settings.service_name}

    return app


app = create_app()
