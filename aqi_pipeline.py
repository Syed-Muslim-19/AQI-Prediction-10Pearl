"""AQI feature pipeline for Lahore DHA Phase 4.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

DEFAULT_LOCATION_LABEL = "DHA Phase 4, Lahore, Pakistan"
DEFAULT_LATITUDE = 31.4697
DEFAULT_LONGITUDE = 74.3984
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_FEATURE_STORE_DIR = Path("data/feature_store")
OPENWEATHER_ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_AIR_POLLUTION_ENDPOINT = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_TIMEMACHINE_ENDPOINT = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
OPENWEATHER_AIR_POLLUTION_HISTORY_ENDPOINT = "https://api.openweathermap.org/data/2.5/air_pollution/history"
DEFAULT_FORECAST_HORIZONS = (1, 24, 72)


@dataclass(frozen=True)
class PipelineSettings:
    openweather_api_key: str
    latitude: float
    longitude: float
    location_label: str
    output_dir: Path
    raw_dir: Path
    feature_store_dir: Path


def load_settings(args: argparse.Namespace) -> PipelineSettings:
    openweather_api_key = args.openweather_api_key or os.getenv("OPENWEATHER_API_KEY")
    if not openweather_api_key:
        raise ValueError("Missing OpenWeather API key. Set OPENWEATHER_API_KEY or pass --openweather-api-key.")

    latitude = args.latitude if args.latitude is not None else float(os.getenv("AQI_LATITUDE", DEFAULT_LATITUDE))
    longitude = args.longitude if args.longitude is not None else float(os.getenv("AQI_LONGITUDE", DEFAULT_LONGITUDE))
    location_label = args.location_label or os.getenv("AQI_LOCATION_LABEL", DEFAULT_LOCATION_LABEL)
    output_dir = Path(args.output_dir or os.getenv("AQI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    raw_dir = Path(args.raw_dir or os.getenv("AQI_RAW_DIR", DEFAULT_RAW_DIR))
    feature_store_dir = Path(args.feature_store_dir or os.getenv("AQI_FEATURE_STORE_DIR", DEFAULT_FEATURE_STORE_DIR))

    return PipelineSettings(
        openweather_api_key=openweather_api_key,
        latitude=latitude,
        longitude=longitude,
        location_label=location_label,
        output_dir=output_dir,
        raw_dir=raw_dir,
        feature_store_dir=feature_store_dir,
    )


def fetch_openweather_current(latitude: float, longitude: float, api_key: str) -> dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(OPENWEATHER_ENDPOINT, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_openweather_air_pollution(latitude: float, longitude: float, api_key: str) -> dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": api_key,
    }
    response = requests.get(OPENWEATHER_AIR_POLLUTION_ENDPOINT, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload


def fetch_openweather_air_pollution_history(
    latitude: float,
    longitude: float,
    api_key: str,
    start_ts: int,
    end_ts: int,
) -> dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "start": start_ts,
        "end": end_ts,
        "appid": api_key,
    }
    response = requests.get(OPENWEATHER_AIR_POLLUTION_HISTORY_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_openweather_weather_history(
    latitude: float,
    longitude: float,
    api_key: str,
    timestamp: datetime,
) -> dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "dt": int(timestamp.timestamp()),
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(OPENWEATHER_TIMEMACHINE_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "v" in value:
            return value["v"]
        if "value" in value:
            return value["value"]
    if isinstance(value, (int, float)):
        return value
    return None


def _safe_int(value: Any) -> Optional[int]:
    scalar = _scalar(value)
    return None if scalar is None else int(scalar)


def _aqi_label(value: Any) -> Optional[str]:
    return {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor",
    }.get(value)


def _parse_utc_datetime(value: str) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid UTC datetime value: {value}")
    return parsed.to_pydatetime()


def normalize_weather_history_payload(
    weather_payload: dict[str, Any],
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    raw_data = (weather_payload.get("data") or [{}])[0]
    return {
        "coord": {"lat": weather_payload.get("lat", latitude), "lon": weather_payload.get("lon", longitude)},
        "dt": raw_data.get("dt"),
        "main": {
            "temp": raw_data.get("temp"),
            "feels_like": raw_data.get("feels_like"),
            "pressure": raw_data.get("pressure"),
            "humidity": raw_data.get("humidity"),
            "temp_min": raw_data.get("temp", raw_data.get("temp_min")),
            "temp_max": raw_data.get("temp", raw_data.get("temp_max")),
        },
        "wind": {
            "speed": raw_data.get("wind_speed"),
            "deg": raw_data.get("wind_deg"),
        },
        "clouds": {"all": raw_data.get("clouds")},
        "rain": raw_data.get("rain", {}),
        "snow": raw_data.get("snow", {}),
        "visibility": raw_data.get("visibility"),
        "weather": raw_data.get("weather", []),
    }


def build_feature_row(
    openweather_payload: dict[str, Any] | None,
    air_pollution_payload: dict[str, Any],
    location_label: str,
    latitude: float | None = None,
    longitude: float | None = None,
    observed_at_override: datetime | None = None,
) -> dict[str, Any]:
    if openweather_payload is None:
        observed_at = observed_at_override or datetime.now(timezone.utc)
        main = {}
        wind = {}
        clouds = {}
        rain = {}
        snow = {}
        coord = {"lat": latitude, "lon": longitude}
        weather_primary = {}
    else:
        weather_ts = openweather_payload.get("dt")
        observed_at = datetime.fromtimestamp(weather_ts, tz=timezone.utc) if weather_ts else (observed_at_override or datetime.now(timezone.utc))
        main = openweather_payload.get("main", {})
        wind = openweather_payload.get("wind", {})
        clouds = openweather_payload.get("clouds", {})
        rain = openweather_payload.get("rain", {})
        snow = openweather_payload.get("snow", {})
        coord = openweather_payload.get("coord", {"lat": latitude, "lon": longitude})
        weather_primary = (openweather_payload.get("weather") or [{}])[0]

    pollution_data = (air_pollution_payload.get("list") or [{}])[0]
    pollution_main = pollution_data.get("main", {})
    pollution_components = pollution_data.get("components", {})
    aqi_time = pollution_data.get("dt")
    aqi_observed_at = datetime.fromtimestamp(aqi_time, tz=timezone.utc) if aqi_time else observed_at

    aqi_value = pollution_main.get("aqi")

    feature_row = {
        "location_label": location_label,
        "latitude": coord.get("lat"),
        "longitude": coord.get("lon"),
        "observed_at_utc": aqi_observed_at.isoformat(),
        "hour": aqi_observed_at.hour,
        "day": aqi_observed_at.day,
        "month": aqi_observed_at.month,
        "day_of_week": aqi_observed_at.weekday(),
        "is_weekend": int(aqi_observed_at.weekday() >= 5),
        "weather_main": weather_primary.get("main"),
        "weather_description": weather_primary.get("description"),
        "temperature_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "temp_min_c": main.get("temp_min"),
        "temp_max_c": main.get("temp_max"),
        "humidity_pct": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "visibility_m": openweather_payload.get("visibility") if openweather_payload is not None else None,
        "wind_speed_mps": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "cloud_cover_pct": clouds.get("all"),
        "rain_1h_mm": rain.get("1h"),
        "snow_1h_mm": snow.get("1h"),
        "aqi": aqi_value,
        "aqi_label": _aqi_label(aqi_value),
        "co": _scalar(pollution_components.get("co")),
        "no": _scalar(pollution_components.get("no")),
        "no2": _scalar(pollution_components.get("no2")),
        "o3": _scalar(pollution_components.get("o3")),
        "so2": _scalar(pollution_components.get("so2")),
        "pm2_5": _scalar(pollution_components.get("pm2_5")),
        "pm10": _scalar(pollution_components.get("pm10")),
        "nh3": _scalar(pollution_components.get("nh3")),
        "air_pollution_source": "openweather",
    }

    return feature_row


def build_backfill_feature_frame(
    settings: PipelineSettings,
    start_at: datetime,
    end_at: datetime,
    step_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if end_at < start_at:
        raise ValueError("Backfill end must be greater than or equal to backfill start.")
    if step_hours < 1:
        raise ValueError("Backfill step must be at least 1 hour.")

    timestamps = pd.date_range(start=start_at, end=end_at, freq=f"{step_hours}h", tz="UTC")
    if timestamps.empty:
        raise ValueError("No backfill timestamps were generated from the requested range.")

    pollution_history = fetch_openweather_air_pollution_history(
        settings.latitude,
        settings.longitude,
        settings.openweather_api_key,
        int(start_at.timestamp()),
        int(end_at.timestamp()),
    )
    pollution_by_timestamp = {
        int(entry.get("dt")): entry
        for entry in pollution_history.get("list", [])
        if entry.get("dt") is not None
    }

    rows = []
    weather_snapshots = []
    for timestamp in timestamps:
        pollution_entry = pollution_by_timestamp.get(int(timestamp.timestamp()))
        if pollution_entry is None:
            continue

        normalized_weather_payload: dict[str, Any] | None = None
        weather_mode = "pollution_only"
        try:
            weather_history_payload = fetch_openweather_weather_history(
                settings.latitude,
                settings.longitude,
                settings.openweather_api_key,
                timestamp.to_pydatetime(),
            )
            normalized_weather_payload = normalize_weather_history_payload(
                weather_history_payload,
                settings.latitude,
                settings.longitude,
            )
            weather_mode = "historical_weather"
        except requests.HTTPError as exc:
            status_code = getattr(exc.response, "status_code", None)
            if status_code not in {401, 403, 404}:
                raise

        rows.append(
            build_feature_row(
                normalized_weather_payload,
                {"list": [pollution_entry]},
                settings.location_label,
                settings.latitude,
                settings.longitude,
                timestamp.to_pydatetime(),
            )
        )
        weather_snapshots.append(
            {
                "requested_at_utc": timestamp.isoformat(),
                "weather_mode": weather_mode,
                "weather": normalized_weather_payload,
                "air_pollution": pollution_entry,
            }
        )

    if not rows:
        raise ValueError("No backfill feature rows were built. Check the requested time range and API access.")

    feature_frame = pd.DataFrame(rows).sort_values("observed_at_utc").reset_index(drop=True)
    raw_summary = {
        "location_label": settings.location_label,
        "start_at_utc": start_at.isoformat(),
        "end_at_utc": end_at.isoformat(),
        "step_hours": step_hours,
        "row_count": len(feature_frame),
        "weather_snapshots": weather_snapshots,
        "air_pollution_history": pollution_history,
    }
    return feature_frame, raw_summary


def build_training_table(history_frame: pd.DataFrame) -> pd.DataFrame:
    frame = history_frame.copy()
    frame["observed_at_utc"] = pd.to_datetime(frame["observed_at_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["observed_at_utc"]).sort_values(["location_label", "observed_at_utc"]).reset_index(drop=True)

    numeric_columns = [
        "aqi",
        "temperature_c",
        "feels_like_c",
        "temp_min_c",
        "temp_max_c",
        "humidity_pct",
        "pressure_hpa",
        "visibility_m",
        "wind_speed_mps",
        "wind_deg",
        "cloud_cover_pct",
        "rain_1h_mm",
        "snow_1h_mm",
        "pm2_5",
        "pm10",
        "no",
        "no2",
        "o3",
        "so2",
        "co",
        "nh3",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = frame.groupby("location_label", group_keys=False)
    frame["aqi_lag_1"] = grouped["aqi"].shift(1)
    frame["aqi_lag_24"] = grouped["aqi"].shift(24)
    frame["aqi_change_rate_1h"] = frame["aqi"] - frame["aqi_lag_1"]
    frame["aqi_pct_change_1h"] = frame["aqi_change_rate_1h"] / frame["aqi_lag_1"].replace(0, pd.NA)
    frame["aqi_3h_mean"] = grouped["aqi"].transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
    frame["aqi_24h_mean"] = grouped["aqi"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())
    frame["pm2_5_24h_mean"] = grouped["pm2_5"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())
    frame["temperature_24h_mean"] = grouped["temperature_c"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())
    frame["humidity_24h_mean"] = grouped["humidity_pct"].transform(lambda series: series.shift(1).rolling(24, min_periods=1).mean())

    for horizon in DEFAULT_FORECAST_HORIZONS:
        frame[f"target_aqi_t_plus_{horizon}h"] = grouped["aqi"].shift(-horizon)

    return frame


def write_feature_outputs(
    feature_frame: pd.DataFrame,
    settings: PipelineSettings,
    raw_payloads: list[tuple[str, Any]],
    raw_prefix: str,
) -> tuple[Path, Path]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.feature_store_dir.mkdir(parents=True, exist_ok=True)

    processed_path = settings.output_dir / "feature_snapshot.csv"
    feature_store_path = settings.feature_store_dir / "aqi_feature_table.csv"
    raw_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix, payload in raw_payloads:
        (settings.raw_dir / f"{raw_prefix}_{suffix}_{raw_stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if processed_path.exists():
        existing = pd.read_csv(processed_path)
        feature_frame = pd.concat([existing, feature_frame], ignore_index=True)
    feature_frame.to_csv(processed_path, index=False)

    training_frame = build_training_table(feature_frame)
    training_frame.to_csv(feature_store_path, index=False)
    return processed_path, feature_store_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch weather and OpenWeather AQI data and build AQI features.")
    parser.add_argument("--openweather-api-key", dest="openweather_api_key")
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--location-label", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--feature-store-dir", default=None)
    parser.add_argument("--backfill-start", default=None)
    parser.add_argument("--backfill-end", default=None)
    parser.add_argument("--backfill-step-hours", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args)
    if args.backfill_start and args.backfill_end:
        start_at = _parse_utc_datetime(args.backfill_start)
        end_at = _parse_utc_datetime(args.backfill_end)
        feature_frame, raw_summary = build_backfill_feature_frame(settings, start_at, end_at, args.backfill_step_hours)
        processed_path, feature_store_path = write_feature_outputs(
            feature_frame,
            settings,
            [("history", raw_summary)],
            "backfill",
        )
        print(f"Saved processed features to {processed_path}")
        print(f"Saved training features and targets to {feature_store_path}")
        print(feature_frame.head().to_string(index=False))
        return

    openweather_payload = fetch_openweather_current(settings.latitude, settings.longitude, settings.openweather_api_key)
    air_pollution_payload = fetch_openweather_air_pollution(settings.latitude, settings.longitude, settings.openweather_api_key)
    feature_row = build_feature_row(openweather_payload, air_pollution_payload, settings.location_label)
    processed_path, feature_store_path = write_feature_outputs(
        pd.DataFrame([feature_row]),
        settings,
        [("weather", openweather_payload), ("air_pollution", air_pollution_payload)],
        "openweather",
    )
    print(f"Saved processed features to {processed_path}")
    print(f"Saved training features and targets to {feature_store_path}")
    print(pd.DataFrame([feature_row]).to_string(index=False))


if __name__ == "__main__":
    main()
