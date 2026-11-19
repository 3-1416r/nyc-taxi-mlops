import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from taxi_demand_mlops.api import LoadedModel, create_app
from taxi_demand_mlops.training import FEATURE_COLUMNS

VALID_REQUEST = {
    "pickup_zone_id": 161,
    "hour": 14,
    "day_of_week": 2,
    "month": 3,
    "is_weekend": False,
    "lag_1": 8,
    "lag_24": 12,
    "lag_168": 10,
}


class FakeModel:
    def __init__(self, predictions: list[float] | None = None) -> None:
        self.predictions = predictions or [7.25]
        self.received: pd.DataFrame | None = None

    def predict(self, data: pd.DataFrame) -> list[float]:
        self.received = data
        return self.predictions


def test_health_and_prediction_use_candidate_model_once() -> None:
    model = FakeModel()
    loader_calls = 0

    def loader() -> LoadedModel:
        nonlocal loader_calls
        loader_calls += 1
        return LoadedModel(model=model, version="1")

    with TestClient(create_app(loader)) as client:
        health = client.get("/health")
        prediction = client.post("/v1/predict", json=VALID_REQUEST)

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_version": "1"}
    assert prediction.status_code == 200
    assert prediction.json() == {"forecast_demand": 7.25, "model_version": "1"}
    assert prediction.headers["x-request-id"]
    assert float(prediction.headers["x-process-time-ms"]) >= 0
    assert loader_calls == 1
    assert model.received is not None
    assert model.received.columns.tolist() == FEATURE_COLUMNS
    assert model.received.iloc[0].to_dict() == VALID_REQUEST


@pytest.mark.parametrize(
    "updates",
    [
        {"hour": 24},
        {"day_of_week": 7},
        {"pickup_zone_id": 0},
        {"lag_24": -1},
        {"unrecognized": 1},
    ],
)
def test_prediction_rejects_invalid_input(updates: dict[str, object]) -> None:
    model = FakeModel()
    with TestClient(create_app(lambda: LoadedModel(model, "1"))) as client:
        response = client.post("/v1/predict", json={**VALID_REQUEST, **updates})

    assert response.status_code == 422


def test_prediction_rejects_non_finite_lag() -> None:
    model = FakeModel()
    request_body = (
        '{"pickup_zone_id":161,"hour":14,"day_of_week":2,"month":3,'
        '"is_weekend":false,"lag_1":NaN,"lag_24":12,"lag_168":10}'
    )
    with TestClient(create_app(lambda: LoadedModel(model, "1"))) as client:
        response = client.post(
            "/v1/predict",
            content=request_body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 422


def test_health_and_prediction_fail_safely_when_registry_is_unavailable() -> None:
    def loader() -> LoadedModel:
        raise MlflowException("registry unavailable")

    with TestClient(create_app(loader)) as client:
        health = client.get("/health")
        prediction = client.post("/v1/predict", json=VALID_REQUEST)

    assert health.status_code == 503
    assert health.json() == {"status": "degraded", "model_version": None}
    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "model not loaded"}


@pytest.mark.parametrize("predictions", [[-1.0], [float("inf")], [1.0, 2.0]])
def test_invalid_model_output_is_reported_as_server_error(
    predictions: list[float],
) -> None:
    model = FakeModel(predictions)
    with TestClient(create_app(lambda: LoadedModel(model, "1"))) as client:
        response = client.post("/v1/predict", json=VALID_REQUEST)

    assert response.status_code == 500
    assert response.json() == {"detail": "model returned an invalid forecast"}
