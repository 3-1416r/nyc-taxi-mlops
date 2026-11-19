from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion, RegisteredModel

MODEL_NAME = "nyc-taxi-demand"
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"
PREVIOUS_CHAMPION_TAG = "taxi_demand.previous_champion_version"
PROMOTION_STATUS_TAG = "taxi_demand.promotion_status"
ROLLBACK_STATUS_TAG = "taxi_demand.rollback_status"


class ModelRegistry(Protocol):
    def get_registered_model(self, name: str) -> RegisteredModel: ...

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion: ...

    def get_model_version(self, name: str, version: str) -> ModelVersion: ...

    def get_run(self, run_id: str) -> Run: ...

    def set_registered_model_alias(
        self, name: str, alias: str, version: str
    ) -> None: ...

    def set_model_version_tag(
        self, name: str, version: str, key: str, value: str
    ) -> None: ...

    def set_tag(self, run_id: str, key: str, value: str) -> None: ...


@dataclass(frozen=True)
class PromotionResult:
    status: str
    candidate_version: str
    champion_version: str
    reason: str


def ensure_champion_initialized(
    client: ModelRegistry,
    model_name: str = MODEL_NAME,
) -> str:
    aliases = client.get_registered_model(model_name).aliases
    if CHAMPION_ALIAS in aliases:
        return str(aliases[CHAMPION_ALIAS])
    if CANDIDATE_ALIAS not in aliases:
        raise ValueError(
            f"Model {model_name!r} has no @{CANDIDATE_ALIAS} alias to initialize"
        )
    version = str(aliases[CANDIDATE_ALIAS])
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    return version


def promotion_decision(
    *,
    candidate_mae: float,
    candidate_wape: float,
    champion_mae: float,
    champion_wape: float,
    minimum_mae_improvement: float = 0.01,
    maximum_wape_regression: float = 0.02,
) -> tuple[bool, str]:
    scores = (candidate_mae, candidate_wape, champion_mae, champion_wape)
    if not all(math.isfinite(score) and score >= 0 for score in scores):
        raise ValueError("Promotion metrics must be finite and non-negative")
    if champion_mae == 0 or champion_wape == 0:
        raise ValueError("Cannot compare against a zero-error champion")
    if (
        not math.isfinite(minimum_mae_improvement)
        or minimum_mae_improvement < 0
        or minimum_mae_improvement >= 1
    ):
        raise ValueError("minimum_mae_improvement must be in [0, 1)")
    if not math.isfinite(maximum_wape_regression) or maximum_wape_regression < 0:
        raise ValueError("maximum_wape_regression must be finite and non-negative")

    required_mae = champion_mae * (1 - minimum_mae_improvement)
    allowed_wape = champion_wape * (1 + maximum_wape_regression)
    if candidate_mae > required_mae:
        return (
            False,
            (
                f"MAE {candidate_mae:.4f} did not improve by "
                f"{minimum_mae_improvement:.1%} over champion MAE {champion_mae:.4f}"
            ),
        )
    if candidate_wape > allowed_wape:
        return (
            False,
            (
                f"WAPE {candidate_wape:.4f} exceeds the allowed "
                f"{maximum_wape_regression:.1%} regression from {champion_wape:.4f}"
            ),
        )
    return True, "candidate passed the MAE improvement and WAPE guardrails"


def promote_candidate(
    client: MlflowClient,
    *,
    model_name: str = MODEL_NAME,
    minimum_mae_improvement: float = 0.01,
    maximum_wape_regression: float = 0.02,
) -> PromotionResult:
    candidate = client.get_model_version_by_alias(model_name, CANDIDATE_ALIAS)
    champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    candidate_version = str(candidate.version)
    champion_version = str(champion.version)
    if candidate_version == champion_version:
        return PromotionResult(
            "already_champion",
            candidate_version,
            champion_version,
            "candidate alias already points to champion",
        )
    if candidate.tags.get(ROLLBACK_STATUS_TAG) == "rolled_back":
        raise ValueError(
            "The candidate version was rolled back; register a new version before promotion"
        )

    run = client.get_run(candidate.run_id)
    required_metrics = {
        "candidate_validation_mae",
        "candidate_validation_wape",
        "champion_validation_mae",
        "champion_validation_wape",
    }
    missing_metrics = required_metrics.difference(run.data.metrics)
    if missing_metrics:
        raise ValueError(
            f"Candidate run is missing promotion metrics: {sorted(missing_metrics)}"
        )
    accepted, reason = promotion_decision(
        candidate_mae=run.data.metrics["candidate_validation_mae"],
        candidate_wape=run.data.metrics["candidate_validation_wape"],
        champion_mae=run.data.metrics["champion_validation_mae"],
        champion_wape=run.data.metrics["champion_validation_wape"],
        minimum_mae_improvement=minimum_mae_improvement,
        maximum_wape_regression=maximum_wape_regression,
    )
    if not accepted:
        client.set_model_version_tag(
            model_name, candidate_version, PROMOTION_STATUS_TAG, "rejected"
        )
        client.set_model_version_tag(
            model_name, candidate_version, "taxi_demand.promotion_reason", reason
        )
        client.set_tag(candidate.run_id, "promotion_status", "rejected")
        client.set_tag(candidate.run_id, "promotion_reason", reason)
        return PromotionResult(
            "rejected",
            candidate_version,
            champion_version,
            reason,
        )

    client.set_model_version_tag(
        model_name,
        candidate_version,
        PREVIOUS_CHAMPION_TAG,
        champion_version,
    )
    client.set_model_version_tag(
        model_name, candidate_version, PROMOTION_STATUS_TAG, "pending"
    )
    client.set_registered_model_alias(
        model_name,
        CHAMPION_ALIAS,
        candidate_version,
    )
    client.set_model_version_tag(
        model_name, candidate_version, PROMOTION_STATUS_TAG, "promoted"
    )
    client.set_model_version_tag(
        model_name,
        candidate_version,
        "taxi_demand.promoted_at",
        datetime.now(UTC).isoformat(),
    )
    client.set_tag(candidate.run_id, "promotion_status", "promoted")
    client.set_tag(candidate.run_id, "previous_champion_version", champion_version)
    return PromotionResult(
        "promoted",
        candidate_version,
        candidate_version,
        reason,
    )


def rollback_champion(
    client: MlflowClient,
    *,
    reason: str,
    model_name: str = MODEL_NAME,
) -> tuple[str, str]:
    cleaned_reason = reason.strip()
    if not cleaned_reason:
        raise ValueError("A non-empty rollback reason is required")
    current = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    previous_version = current.tags.get(PREVIOUS_CHAMPION_TAG)
    if not previous_version:
        raise ValueError(
            f"Champion v{current.version} has no recorded previous champion"
        )
    previous = client.get_model_version(model_name, previous_version)
    current_version = str(current.version)
    target_version = str(previous.version)

    client.set_model_version_tag(
        model_name, current_version, ROLLBACK_STATUS_TAG, "pending"
    )
    client.set_model_version_tag(
        model_name, current_version, "taxi_demand.rollback_reason", cleaned_reason
    )
    client.set_registered_model_alias(
        model_name,
        CHAMPION_ALIAS,
        target_version,
    )
    timestamp = datetime.now(UTC).isoformat()
    client.set_model_version_tag(
        model_name, current_version, ROLLBACK_STATUS_TAG, "rolled_back"
    )
    client.set_model_version_tag(
        model_name,
        current_version,
        "taxi_demand.rolled_back_to_version",
        target_version,
    )
    client.set_model_version_tag(
        model_name, current_version, "taxi_demand.rolled_back_at", timestamp
    )
    client.set_tag(current.run_id, "rollback_reason", cleaned_reason)
    client.set_tag(current.run_id, "rolled_back_to_version", target_version)
    return current_version, target_version
