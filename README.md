<div align="center">

# 🌫️ AQI Forecast

**Hourly air-quality forecasting for Lahore, PK — built to be honest about its own accuracy.**

![Python](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006ACC?style=flat-square)
![Status](https://img.shields.io/badge/status-collecting%20data-yellow?style=flat-square)

</div>

---

## Architecture

```mermaid
flowchart LR
    A[🛰️ OpenWeather API] -->|hourly scrape| B[(Feature Store<br/>aqi_feature_table.csv)]
    B --> C{Training Pipeline<br/>aqi_train.py}
    C --> D[Persistence Baseline]
    C --> E[Model Zoo + CV Search]
    E --> F[(Model Registry<br/>joblib + metrics.json)]
    D & E --> G[Rolling Backtest]
    G --> H[📋 Verdict]
    F --> I[📊 Dashboard]

    style A fill:#2b2d42,color:#fff
    style B fill:#8d99ae,color:#000
    style C fill:#ef233c,color:#fff
    style F fill:#8d99ae,color:#000
    style I fill:#3a86ff,color:#fff
```

**Pipeline stages**

| Stage | What happens |
|---|---|
| **Scraper** | Polls OpenWeather hourly for weather + pollutant readings, appends to the feature store |
| **Feature Store** | Flat CSV with raw readings, engineered lags/rolling means, and 1h/24h/72h forward targets |
| **Training** | Delta-target regression (predict *change*, not raw AQI) with leakage-safe feature selection |
| **Evaluation** | Persistence baseline + changed-rows breakdown + rolling-origin backtest — never a single trusted number |
| **Registry** | Every run's model, preprocessor, and metrics saved and versioned by timestamp |
| **Dashboard** | Visualizes AQI trend history, live forecast vs. actual, and backtest verdict over time |

---

## Models

Three tabular regressors, each tuned via `RandomizedSearchCV` over `TimeSeriesSplit`, competing on the same delta-target task:

| Model | Why it's here |
|---|---|
| 🌲 **XGBoost** | Captures nonlinear interactions between weather + pollutant features |
| 🌳 **Random Forest** | Regularization-friendly baseline, resistant to overfitting on small data |
| 📈 **Elastic Net** | Linear + sparse — a check against the tree models overfitting noise |

The winner is picked by RMSE (or by error on rows where AQI *actually changed*, once enough of those exist) — never assumed.

---

## Dashboard

A lightweight view over the model registry, showing:

- 📉 Recent AQI trend vs. 1h-ahead forecast
- ✅ Live comparison against the persistence baseline
- 🔁 Rolling backtest win-rate over time, so accuracy claims are always checkable at a glance

*(planned / in progress — surfaces the same `VERDICT` the CLI prints, as a chart instead of a log line)*

---

## Quick start

```bash
pip install pandas numpy scikit-learn xgboost joblib scipy
python aqi_train.py --feature-store-path data/feature_store/aqi_feature_table.csv
```

---

## Deploying the dashboard

The dashboard and API are separate services. Streamlit Community Cloud runs
`dashboard.py`, but it does not run `api_server.py` on the same machine.
Therefore, `http://127.0.0.1:8000` only works when the API is running locally.

1. Deploy this repository as a web service on a host such as Render, Railway,
   or Fly.io. Use the included `Dockerfile`, or start the service with:

   ```bash
   uvicorn api_server:app --host 0.0.0.0 --port $PORT
   ```

   The service must expose the repository files under `data/`, including
   `data/feature_store/` and `data/model_registry/`. Verify it responds at
   `https://<api-host>/health`.
2. In the Streamlit app settings, add the environment variable
   `AQI_API_URL=https://<api-host>` (no `/latest` suffix), then reboot the app.

For local development, start the API in one terminal and the dashboard in
another:

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
streamlit run dashboard.py
```

Do not use `127.0.0.1` as `AQI_API_URL` in the deployed Streamlit app; it
refers to the Streamlit container itself, not your computer or the API host.

---

<div align="center">
<sub>Built with an unusual amount of paranoia about fooling itself.</sub>
</div>
