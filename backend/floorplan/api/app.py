"""FastAPI application factory for the floor plan module."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from floorplan.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="FloorPlan Dataset Engine",
        version="1.0.0",
        description="Deterministic residential floor plan generation, validation, rendering, and scoring.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
