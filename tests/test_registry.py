from types import SimpleNamespace

import pytest

from taxi_demand_mlops.registry import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    PREVIOUS_CHAMPION_TAG,
    PROMOTION_STATUS_TAG,
    ROLLBACK_STATUS_TAG,
    ensure_champion_initialized,
    promote_candidate,
    promotion_decision,
    rollback_champion,
)


class FakeRegistry:
    def __init__(
        self,
        aliases: dict[str, str],
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.aliases = aliases
        self.versions = {
            version: SimpleNamespace(
                version=version,
                run_id=f"run-{version}",
                tags={},
            )
            for version in set(aliases.values())
        }
        self.runs = {
            f"run-{version}": SimpleNamespace(
                data=SimpleNamespace(metrics=metrics or {}, tags={})
            )
            for version in self.versions
        }

    def get_registered_model(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(aliases=self.aliases)

    def get_model_version_by_alias(
        self,
        name: str,
        alias: str,
    ) -> SimpleNamespace:
        return self.versions[self.aliases[alias]]

    def get_model_version(self, name: str, version: str) -> SimpleNamespace:
        return self.versions[version]

    def get_run(self, run_id: str) -> SimpleNamespace:
        return self.runs[run_id]

    def set_registered_model_alias(
        self,
        name: str,
        alias: str,
        version: str,
    ) -> None:
        self.aliases[alias] = version

    def set_model_version_tag(
        self,
        name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.versions[version].tags[key] = value

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.runs[run_id].data.tags[key] = value


def _candidate_metrics(
    *,
    candidate_mae: float = 8.5,
    candidate_wape: float = 0.19,
) -> dict[str, float]:
    return {
        "candidate_validation_mae": candidate_mae,
        "candidate_validation_wape": candidate_wape,
        "champion_validation_mae": 10.0,
        "champion_validation_wape": 0.20,
    }


def test_champion_initialization_is_safe_and_idempotent() -> None:
    client = FakeRegistry({CANDIDATE_ALIAS: "1"})

    assert ensure_champion_initialized(client) == "1"
    assert ensure_champion_initialized(client) == "1"
    assert client.aliases[CHAMPION_ALIAS] == "1"


def test_promotion_moves_champion_and_records_rollback_version() -> None:
    client = FakeRegistry(
        {CANDIDATE_ALIAS: "2", CHAMPION_ALIAS: "1"},
        _candidate_metrics(),
    )

    result = promote_candidate(client)

    assert result.status == "promoted"
    assert result.champion_version == "2"
    assert client.aliases[CHAMPION_ALIAS] == "2"
    assert client.versions["2"].tags[PROMOTION_STATUS_TAG] == "promoted"
    assert client.versions["2"].tags[PREVIOUS_CHAMPION_TAG] == "1"


@pytest.mark.parametrize(
    ("candidate_mae", "candidate_wape"),
    [(9.95, 0.19), (8.5, 0.25)],
)
def test_promotion_rejects_candidates_that_fail_either_guardrail(
    candidate_mae: float,
    candidate_wape: float,
) -> None:
    client = FakeRegistry(
        {CANDIDATE_ALIAS: "2", CHAMPION_ALIAS: "1"},
        _candidate_metrics(
            candidate_mae=candidate_mae,
            candidate_wape=candidate_wape,
        ),
    )

    result = promote_candidate(client)

    assert result.status == "rejected"
    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert client.versions["2"].tags[PROMOTION_STATUS_TAG] == "rejected"


def test_rollback_restores_previous_champion_and_marks_failed_candidate() -> None:
    client = FakeRegistry({CANDIDATE_ALIAS: "2", CHAMPION_ALIAS: "2"})
    client.versions["2"].tags[PREVIOUS_CHAMPION_TAG] = "1"
    client.versions["1"] = SimpleNamespace(
        version="1",
        run_id="run-1",
        tags={},
    )
    client.runs["run-1"] = SimpleNamespace(data=SimpleNamespace(metrics={}, tags={}))

    current, restored = rollback_champion(
        client,
        reason="Validation quality regressed",
    )

    assert (current, restored) == ("2", "1")
    assert client.aliases[CHAMPION_ALIAS] == "1"
    assert client.versions["2"].tags[ROLLBACK_STATUS_TAG] == "rolled_back"
    with pytest.raises(ValueError, match="was rolled back"):
        promote_candidate(client)


def test_promotion_decision_requires_meaningful_mae_gain_and_wape_guardrail() -> None:
    accepted, _ = promotion_decision(
        candidate_mae=9,
        candidate_wape=0.202,
        champion_mae=10,
        champion_wape=0.2,
    )
    rejected, reason = promotion_decision(
        candidate_mae=9.95,
        candidate_wape=0.19,
        champion_mae=10,
        champion_wape=0.2,
    )

    assert accepted
    assert not rejected
    assert "MAE" in reason
