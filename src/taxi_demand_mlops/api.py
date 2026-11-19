from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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

from taxi_demand_mlops.monitoring import (
    API_REQUEST_DURATION,
    API_REQUESTS,
    DEFAULT_PROFILE_PATH,
    PROMETHEUS_APP,
    InputDriftMonitor,
    load_reference_profile,
    observe_prediction,
)
from taxi_demand_mlops.training import (
    FEATURE_COLUMNS,
    prepare_model_features,
)

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "nyc-taxi-demand"
SERVING_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{SERVING_ALIAS}"


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


def load_serving_model() -> LoadedModel:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("MODEL_NAME", MODEL_NAME)
    model_alias = os.getenv("MODEL_ALIAS", SERVING_ALIAS)
    model_uri = os.getenv("MODEL_URI", f"models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(model_name, model_alias)
    model = mlflow.pyfunc.load_model(model_uri)
    return LoadedModel(model=model, version=str(version.version))


def create_app(model_loader: ModelLoader = load_serving_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.model_version = None
        profile_path = Path(
            os.getenv("TAXI_MONITORING_PROFILE", str(DEFAULT_PROFILE_PATH))
        )
        app.state.input_drift_monitor = InputDriftMonitor(
            load_reference_profile(profile_path)
        )
        try:
            loaded = model_loader()
        except MlflowException as error:
            LOGGER.exception("Unable to load champion model from MLflow")
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
    application.mount("/metrics", PROMETHEUS_APP)

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
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = (
                f"{(time.perf_counter() - started) * 1000:.2f}"
            )
            return response
        finally:
            if not request.url.path.startswith("/metrics"):
                route = request.scope.get("route")
                route_path = getattr(route, "path", "unmatched")
                API_REQUESTS.labels(
                    request.method,
                    route_path,
                    str(status_code),
                ).inc()
                API_REQUEST_DURATION.labels(
                    request.method,
                    route_path,
                ).observe(time.perf_counter() - started)

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

        features = prepare_model_features(
            pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
        )
        prediction = model.predict(features)
        if len(prediction) != 1:
            LOGGER.error("Model returned %d forecasts for one request", len(prediction))
            raise HTTPException(status_code=500, detail="model returned an invalid forecast")
        forecast = float(prediction[0])
        if not math.isfinite(forecast) or forecast < 0:
            LOGGER.error("Model returned an invalid demand forecast: %r", forecast)
            raise HTTPException(status_code=500, detail="model returned an invalid forecast")
        request.app.state.input_drift_monitor.observe(payload.model_dump())
        observe_prediction(model_version, forecast)
        LOGGER.info(
            "prediction_record=%s",
            json.dumps(
                {
                    "request_id": request.state.request_id,
                    "predicted_at": datetime.now(UTC).isoformat(),
                    "model_version": model_version,
                    "forecast_demand": forecast,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return PredictionResponse(
            forecast_demand=forecast,
            model_version=model_version,
        )

    return application


app = create_app()
