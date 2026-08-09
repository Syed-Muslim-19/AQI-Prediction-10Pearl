from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.getenv("AQI_API_URL", "http://127.0.0.1:8000")

AQI_COLORS = {
    "Good": "#22C55E",
    "Fair": "#84CC16",
    "Moderate": "#F59E0B",
    "Poor": "#F97316",
    "Very Poor": "#EF4444",
}

st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #FAFBFD;
    }
    h1, h2, h3 {
        color: #1E293B;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 18px rgba(30, 41, 59, 0.06);
        border: 1px solid #EEF1F6;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        box-shadow: 0 2px 18px rgba(30, 41, 59, 0.06);
    }
    .stAlert {
        border-radius: 12px;
        box-shadow: 0 2px 16px rgba(30, 41, 59, 0.05);
    }
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 16px rgba(30, 41, 59, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AQI Forecast Dashboard")
st.caption("Real-time and forecasted AQI from feature store + model registry")


def fetch_json(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


try:
    latest = fetch_json("/latest")
    forecast = fetch_json("/forecast?horizon_hours=72")
    eda = fetch_json("/eda")
    explain = fetch_json("/explain?top_k=10")
except Exception as exc:
    st.error(f"Failed to reach API at {API_URL}. Error: {exc}")
    st.info("Start API first: uvicorn api_server:app --host 0.0.0.0 --port 8000")
    st.stop()


def render_gauge(aqi_value: float, aqi_label: str) -> go.Figure:
    color = AQI_COLORS.get(aqi_label, "#3B82F6")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi_value,
            number={"font": {"size": 36, "color": "#1E293B"}},
            gauge={
                "axis": {"range": [1, 5], "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [1, 2], "color": "#DCFCE7"},
                    {"range": [2, 3], "color": "#ECFCCB"},
                    {"range": [3, 4], "color": "#FEF3C7"},
                    {"range": [4, 5], "color": "#FEE2E2"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180, margin=dict(l=20, r=20, t=10, b=10))
    return fig


gauge_col, m1, m2, m3, m4 = st.columns([1, 1, 1, 1, 1])
with gauge_col:
    st.plotly_chart(render_gauge(float(latest.get("aqi", 0) or 0), latest.get("aqi_label", "N/A")), use_container_width=True, config={"displayModeBar": False})
with m1:
    st.metric("Current AQI", latest.get("aqi", "N/A"))
with m2:
    st.metric("AQI Label", latest.get("aqi_label", "N/A"))
with m3:
    st.metric("Model", forecast.get("model_name", "N/A"))
with m4:
    st.metric("Model Trained At", forecast.get("trained_at_utc", "N/A"))

alert_level = latest.get("alert_level")
if alert_level == "hazard":
    st.error("Hazard Alert: AQI is in Poor or Very Poor zone.")
elif alert_level == "warning":
    st.warning("Warning: AQI is in Moderate zone.")
else:
    st.success("Air quality is in Good or Fair zone.")


def light_theme(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B"),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.update_xaxes(gridcolor="#EEF1F6")
    fig.update_yaxes(gridcolor="#EEF1F6")
    return fig


st.subheader("Forecast Comparison")
ml_df = pd.DataFrame(forecast.get("ml_forecast", []))
base_1_df = pd.DataFrame(forecast.get("baselines", {}).get("persistence", []))
base_2_df = pd.DataFrame(forecast.get("baselines", {}).get("moving_average_24h", []))

if not ml_df.empty:
    merged = pd.concat([
        ml_df[["timestamp_utc", "aqi_pred", "model"]],
        base_1_df[["timestamp_utc", "aqi_pred", "model"]],
        base_2_df[["timestamp_utc", "aqi_pred", "model"]],
    ], ignore_index=True)
    merged["timestamp_utc"] = pd.to_datetime(merged["timestamp_utc"], utc=True, errors="coerce")
    fig = px.line(
        merged, x="timestamp_utc", y="aqi_pred", color="model", title="72-Hour AQI Forecast",
        color_discrete_sequence=["#3B82F6", "#94A3B8", "#CBD5E1"],
    )
    fig = light_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    upcoming = ml_df.head(24).copy()
    st.dataframe(upcoming[["step_hour", "timestamp_utc", "aqi_pred", "aqi_label", "alert_level"]], use_container_width=True)
else:
    st.info("No forecast data available yet.")


st.subheader("Exploratory Data Analysis")
eda_col1, eda_col2, eda_col3, eda_col4 = st.columns(4)
eda_col1.metric("Rows", eda.get("row_count", 0))
eda_col2.metric("AQI Mean", round(float(eda.get("aqi_mean", 0.0)), 3))
eda_col3.metric("AQI Min", round(float(eda.get("aqi_min", 0.0)), 3))
eda_col4.metric("AQI Max", round(float(eda.get("aqi_max", 0.0)), 3))
st.caption(f"Trend slope (positive means rising AQI): {eda.get('trend_slope', 0.0):.6f}")

hourly_df = pd.DataFrame(eda.get("hourly_avg", []))
daily_df = pd.DataFrame(eda.get("daily_avg", []))

chart_col1, chart_col2 = st.columns(2)
if not hourly_df.empty:
    fig_hourly = px.bar(hourly_df, x="hour", y="aqi", title="Average AQI by Hour", color_discrete_sequence=["#F59E0B"])
    fig_hourly = light_theme(fig_hourly)
    chart_col1.plotly_chart(fig_hourly, use_container_width=True)
if not daily_df.empty:
    fig_daily = px.line(daily_df, x="date", y="aqi", title="Average AQI by Day", color_discrete_sequence=["#22C55E"])
    fig_daily = light_theme(fig_daily)
    chart_col2.plotly_chart(fig_daily, use_container_width=True)


st.subheader("Feature Importance (SHAP)")
if explain.get("enabled"):
    shap_df = pd.DataFrame(explain.get("top_features", []))
    if not shap_df.empty:
        fig_shap = px.bar(
            shap_df.sort_values("abs_shap"),
            x="abs_shap",
            y="feature",
            orientation="h",
            title="Top SHAP Feature Contributions",
            color_discrete_sequence=["#3B82F6"],
        )
        fig_shap = light_theme(fig_shap)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.dataframe(shap_df, use_container_width=True)
    else:
        st.info("No SHAP values returned.")
else:
    st.info(explain.get("message", "SHAP explanation is unavailable."))
