# NYC Taxi Demand MLOps

Build a reproducible MLOps system that forecasts hourly NYC taxi demand and can
detect, retrain, and recover from data or model problems.

## Project requirements

- Download NYC TLC trip records and aggregate demand by pickup zone and hour.
- Create recent-hour and calendar features; split data chronologically, not randomly.
- Define reference and later drift periods.
- Version data with DVC. Do not commit raw or processed datasets to Git.
- Track parameters, metrics, Git commit, data fingerprint, and artifacts with MLflow.
- Register models and serve predictions through a tested FastAPI service.
- Monitor service health, input drift, prediction behavior, and model quality when
  delayed labels are available.
- Demonstrate drift alerting, a retraining trigger, and recovery by model rollback.
- Document one end-to-end incident and a complete reproduction procedure.

The model's score is less important than reproducibility, monitoring, automation,
and recovery.

## Development plan

1. Establish the data source, reproducible download, hourly aggregation, and
   chronological splits.
2. Add DVC and an MLflow training workflow with a simple baseline model.
3. Register the model and build the FastAPI prediction service and integration tests.
4. Containerize the service and add CI build and smoke tests.
5. Add drift monitoring, alerts, retraining, and rollback.
6. Document reproduction and incident recovery; prepare a live demo and backup.

## Team and sharing

This is being built solo for now. The project repository is private; add the teacher
as a collaborator once their GitHub account is known. The project requirements call
for a three-person group, so confirm with the instructor that solo work is acceptable.

## Development environment

Python 3.11 is the initial target. Setup and run commands will be added alongside
the first working pipeline so they can be verified end to end.