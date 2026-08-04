# AQI Feature Pipeline

This project fetches current weather and air-quality data from OpenWeather for DHA Phase 4, Lahore, then derives a single feature row you can later send to a feature store or use for model training.

## What it does

- Pulls weather data from OpenWeather
- Pulls AQI data from OpenWeather using Lahore coordinates
- Derives time-based features like hour, day, month, and weekend flag
- Builds lag, rolling, and forecast-target features for training
- Stores raw JSON snapshots under `data/raw`
- Appends processed features to `data/processed/feature_snapshot.csv`
- Writes the curated training table to `data/feature_store/aqi_feature_table.csv`

## Setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and fill in your API keys.

## Run

```bash
python aqi_pipeline.py
```

You can also override the defaults:

```bash
python aqi_pipeline.py --latitude 31.4697 --longitude 74.3984 --location-label "DHA Phase 4, Lahore, Pakistan"
```

## Historical Backfill

To generate training data from a historical range, use:

```bash
python aqi_pipeline.py --backfill-start 2026-07-10T00:00:00Z --backfill-end 2026-07-18T23:00:00Z --backfill-step-hours 1
```

Backfill mode pulls historical weather data from OpenWeather One Call timemachine when available and historical air-pollution data from OpenWeather's air pollution history endpoint, then recomputes lag, rolling, and forecast target columns across the full historical frame.

Note: if your account does not have One Call history access, the script falls back to a pollution-only historical backfill so you can still generate training rows and targets.

## Notes

- The OpenWeather air-pollution endpoint uses the same coordinates as the weather lookup.
- The OpenWeather key should stay in environment variables, not in source control.
- If you have a more exact DHA Phase 4 pin, update the latitude and longitude in `.env`.

## Model Training

Train models from the feature store with:

```bash
python aqi_train.py
```

The trainer loads `data/feature_store/aqi_feature_table.csv`, builds a time-based train/test split, compares tabular regression models, and selects the best one by RMSE. It currently compares Ridge regression and Random Forest, with an optional TensorFlow dense network if you install TensorFlow and pass `--include-tensorflow`.

Training artifacts are written to `data/model_registry/`, including the serialized model, preprocessing pipeline, evaluation metrics, and a `latest_model.json` pointer.

## Automated Runs

This repository includes GitHub Actions workflows under `.github/workflows/`:

- `feature_pipeline.yml` runs every hour and updates the checked-in feature store files.
- `training_pipeline.yml` runs every day, refreshes the feature snapshot, trains the model, and uploads the model registry folder as a workflow artifact.

Required GitHub settings:

- `OPENWEATHER_API_KEY` as a repository secret
- `AQI_LATITUDE`, `AQI_LONGITUDE`, and `AQI_LOCATION_LABEL` as repository variables if you want to override the Lahore defaults

The hourly feature workflow commits updated `data/processed/feature_snapshot.csv` and `data/feature_store/aqi_feature_table.csv` back to the repository, which gives the daily trainer a persistent feature table to read from on the next run.

## Web App and API

The serving layer includes:

- `api_server.py` (FastAPI) to load feature store + latest model registry artifact and expose prediction endpoints
- `dashboard.py` (Streamlit) to show real-time AQI, forecasts, model comparison, EDA trends, SHAP explanations, and hazard alerts

### Start API

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Start Dashboard

```bash
streamlit run dashboard.py
```

If your API runs on a custom URL, set:

```bash
AQI_API_URL=http://your-host:8000
```

before launching Streamlit.

### API Endpoints

- `GET /health`
- `GET /latest`
- `GET /forecast?horizon_hours=72`
- `GET /eda`
- `GET /explain?top_k=10`
- `POST /reload`

---

## Industry-readiness checklist (minimal, low-cost / free-tier)

This project is intentionally kept lightweight and free-friendly. The following small changes and practices make it suitable for an internship / small-team production workflow without requiring paid cloud resources.

1. Secrets and config
   - Never commit real API keys. Use `.env` locally and store secrets in your CI (GitHub Actions: repository Secrets).
   - A `.env.example` is present; copy it to `.env` and fill values locally.

2. Reproducible environment
   - Use the included `requirements.txt` and create an isolated virtual environment:
     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate    # Windows PowerShell
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - Optionally pin exact versions with `pip freeze > requirements.lock` for reproducible CI runs.

3. CI / scheduled runs (already present)
   - GitHub Actions workflows under `.github/workflows/` run the feature pipeline hourly and the trainer daily.
   - Add `OPENWEATHER_API_KEY` to repository Secrets before enabling workflows.

4. Model registry and evaluation
   - Trained models are saved under `data/model_registry/` and `latest_model.json` points to the chosen model.
   - The trainer compares multiple models (Ridge, RandomForest, and optional TensorFlow network). Evaluation metrics: RMSE, MAE, R².
   - For categorical AQI alerts the serving layer already computes labels and alert levels; if you want a classification confusion matrix, add a small classifier training step (not required for numeric forecasting).

5. Observability and safety
   - The API exposes `/health` for basic liveness checks.
   - Workflows upload artifacts (feature snapshot and model registry) so you can inspect outputs from scheduled runs.

6. Local/dev deployment (no paid cloud required)
   - Run the feature pipeline locally:
     ```bash
     python aqi_pipeline.py
     ```
   - Train locally (writes a model into `data/model_registry/`):
     ```bash
     python aqi_train.py
     ```
   - Start the API locally:
     ```bash
     uvicorn api_server:app --host 0.0.0.0 --port 8000
     ```
   - Start the Streamlit dashboard (connects to the API):
     ```bash
     streamlit run dashboard.py
     ```
   - A Dockerfile is included for running the API in a container (useful for consistent dev environments):
     ```bash
     docker build -t aqi-api:local .
     docker run -p 8000:8000 --env-file .env aqi-api:local
     ```

7. Next recommended improvements (optional, but low-effort)
   - Add a small test suite (pytest) to cover pipeline core functions (parsing, feature building, training run smoke test).
   - Add minimal linting (flake8) in CI.
   - Add MLflow or local metadata tracking if you want richer experiment tracking (optional — MLflow can run locally and store artifacts on disk).
   - Move secrets in CI to GitHub Actions secrets and remove any local secrets from repository (already applied to `.env`).

If you want, the next step I can perform now is:
- add a lightweight test and CI job for linting/testing, and/or
- add a small pytest smoke test for the pipeline + trainer, and update the workflows to run the test.

Tell me which of those additional tasks to implement, or I can stop here and just leave the README and Dockerfile changes applied.
