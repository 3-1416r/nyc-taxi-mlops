from __future__ import annotations

import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Protocol

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from taxi_demand_mlops.training import FEATURE_COLUMNS

LOGGER = logging.getLogger(__name__)
MODEL_URI = "models:/nyc-taxi-demand@candidate"


class PredictiveModel(Protocol):
    def predict(self, data: pd.DataFrame) -> Sequence[float]: ...


@dataclass(frozen=True)
class LoadedModel:
    model: PredictiveModel
    version: str


ModelLoader = Callable[[], LoadedModel]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pickup_zone_id: Annotated[int, Field(ge=1, le=265, description="NYC TLC pickup zone ID")]
    hour: Annotated[int, Field(ge=0, le=23, description="Pickup hour in local NYC time")]
    day_of_week: Annotated[int, Field(ge=0, le=6, description="Monday=0 through Sunday=6")]
    month: Annotated[int, Field(ge=1, le=12, description="Calendar month")]
    is_weekend: Annotated[bool, Field(description="Whether the pickup date is a weekend")]
    lag_1: Annotated[float, Field(ge=0, allow_inf_nan=False, description="Demand in this zone one hour earlier")]
    lag_24: Annotated[float, Field(ge=0, allow_inf_nan=False, description="Demand in this zone 24 hours earlier")]
    lag_168: Annotated[float, Field(ge=0, allow_inf_nan=False, description="Demand in this zone 168 hours earlier")]


class PredictionResponse(BaseModel):
    forecast_demand: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None


def load_candidate_model() -> LoadedModel:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_uri = os.getenv("MODEL_URI", MODEL_URI)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias("nyc-taxi-demand", "candidate")
    model = mlflow.pyfunc.load_model(model_uri)
    return LoadedModel(model=model, version=str(version.version))


def create_app(model_loader: ModelLoader = load_candidate_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_version = None
        try:
            loaded = model_loader()
        except MlflowException as error:
            LOGGER.exception("Unable to load candidate model from MLflow")
            app.state.model_load_error = type(error).__name__
        else:
            app.state.model = loaded.model
            app.state.model_version = loaded.version
            app.state.model_load_error = None
            LOGGER.info("Loaded demand model version %s", loaded.version)
        yield

    application = FastAPI(
        title="NYC Taxi Demand API",
        description="Forecast hourly pickups for a TLC zone using recent demand and calendar features.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_response(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.middleware("http")
    async def add_response_headers(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request, response: Response) -> HealthResponse:
        model_version = request.app.state.model_version
        if request.app.state.model is None:
            response.status_code = 503
            return HealthResponse(status="degraded", model_version=None)
        return HealthResponse(status="ok", model_version=model_version)

    @application.post("/v1/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        model = request.app.state.model
        model_version = request.app.state.model_version
        if model is None or model_version is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        features = pd.DataFrame(
            [payload.model_dump()],
            columns=FEATURE_COLUMNS,
        )
        prediction = model.predict(features)
        if len(prediction) != 1:
            LOGGER.error("Model returned %d forecasts for one request", len(prediction))
            raise HTTPException(status_code=500, detail="model returned an invalid forecast")
        forecast = float(prediction[0])
        if not math.isfinite(forecast) or forecast < 0:
            LOGGER.error("Model returned an invalid demand forecast: %r", forecast)
            raise HTTPException(status_code=500, detail="model returned an invalid forecast")
        return PredictionResponse(
            forecast_demand=forecast,
            model_version=model_version,
        )

    return application


app = create_app()
