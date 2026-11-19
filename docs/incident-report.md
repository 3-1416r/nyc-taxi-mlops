# Controlled input-drift incident report

## Summary

This is a reproducible demonstration, not an incident affecting live taxi
customers. The replay intentionally multiplies the April demand-lag features by
3 so the API's recent-demand distribution diverges from its March reference
profile. Prometheus detects the shift, and the alert watcher trains and evaluates
a candidate before changing the serving alias.

## Detection and response

The API's 300-prediction rolling window reported the following Population Stability
Index (PSI) values after the 600-request replay:

| Feature | PSI | Alert threshold |
| --- | ---: | ---: |
| `lag_1` | 0.9998 | 0.2 |
| `lag_24` | 0.9705 | 0.2 |
| `lag_168` | 1.1013 | 0.2 |

Each `TaxiInputDrift` alert remained above threshold for the configured 30 seconds
and entered the firing state. The watcher fingerprinted the active alert set and
created one retraining run for it.

The chronological training partition combined January-February, March, and the
first 80% of April. The final 20% of April by pickup hour was held out. With the
same lag scale of 3 used in the replay, MLflow recorded:

| Metric | Existing champion v1 | Candidate v3 |
| --- | ---: | ---: |
| Validation MAE | 25.3414 | 3.3584 |
| Validation WAPE | 1.4440 | 0.1914 |

Candidate v3 passed the configured 1% minimum MAE improvement and 2% maximum WAPE
regression checks, so `@champion` moved to v3. The alert-triggered run is
`3811949057bf48b4a0462530899e6299`.

## Delayed outcomes and rollback

The 600 replay responses were joined with their matching April outcomes by request
ID and model version. Those requests were served by v1 before promotion; their
delayed-label metrics were MAE 7.533 and WAPE 2.429, logged to the v1 run
`43f2713e28924678af9eb6d4fabf2788`. These replay-wide scores use a different
sample from the final-20%-of-April promotion holdout, so they are not directly
comparable to the validation table above.

Rollback was exercised against the local registry before the alert-triggered run:
the then-current champion v2 was restored to its recorded previous version v1.
The subsequent drift-triggered promotion moved the alias to v3. The rollback
operation records its reason, source version, target version, and timestamp in
MLflow; an API process must be restarted to load the restored alias.

## Cause, impact, and limitations

The cause was the intentional 3x scaling in `scripts/simulate_drift.py`, not a
change in the official TLC data. No live users or production service were
affected. The exercise validates the monitoring, deduplication, chronological
retraining, quality gate, registry promotion, delayed-label join, and rollback
mechanisms; it does not establish a production drift rate or guarantee future
model quality. A real deployment should investigate the source of a drift alert
and evaluate labels from its actual outcome system before promotion or rollback.

## Reproduction

Follow [the demo runbook](demo-runbook.md) from a fresh terminal session. The
checked-in data profile and DVC dataset must be present, and MLflow and Docker
Desktop must be available.
