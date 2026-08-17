from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import json
import statistics

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Folder utama repository.
# File ini berada di api/index.py, sehingga parent.parent
# mengarah kembali ke root repository.
BASE_DIR = Path(__file__).resolve().parent.parent


# Instance FastAPI HARUS bernama "app".
app = FastAPI(
    title="Flowrate Forecasting API",
    version="1.0.0"
)


# Bentuk data yang akan dikirim oleh halaman Next.js.
class ForecastRequest(BaseModel):
    site: str = Field(
        min_length=1,
        max_length=10
    )

    model: Literal[
        "baseline",
        "xgboost",
        "statistical"
    ] = "baseline"

    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365
    )


# Hubungan kode site dengan nama file JSON.
SITE_FILES = {
    "b": "added_data_site_b.json",
    "k": "added_data_site_k.json",
    "kl": "added_data_site_kl.json",
    "l": "added_data_site_l.json",
    "m": "added_data_site_m.json",
    "s": "added_data_site_s.json",
    "t": "added_data_site_t.json",
}


@app.get("/api/health")
def health():
    """
    Endpoint sederhana untuk memastikan backend Python aktif.
    """

    return {
        "status": "ok",
        "service": "flowrate-forecasting"
    }


def load_site_json(site: str):
    """
    Membuka file JSON berdasarkan site yang dipilih.
    """

    normalized_site = site.lower().strip()
    filename = SITE_FILES.get(normalized_site)

    if not filename:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown site: {site}"
        )

    file_path = BASE_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Required data file was not found: "
                f"{filename}"
            )
        )

    try:
        with file_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid JSON structure in {filename}: "
                f"{exc}"
            )
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to read {filename}: "
                f"{exc}"
            )
        ) from exc


def extract_numeric_flowrates(data):
    """
    Menelusuri seluruh isi JSON dan mengambil angka dari field
    yang namanya berkaitan dengan flowrate.

    Fungsi ini masih bersifat umum karena struktur JSON aktual
    belum dipetakan secara khusus.
    """

    values = []

    possible_flowrate_keys = {
        "flow",
        "flowrate",
        "flow_rate",
        "flow rate",
        "debit",
        "discharge",
    }

    def convert_to_number(value):
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            cleaned_value = (
                value.strip()
                .replace(",", "")
            )

            try:
                return float(cleaned_value)
            except ValueError:
                return None

        return None

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = (
                    str(key)
                    .lower()
                    .strip()
                )

                key_is_flowrate = (
                    normalized_key
                    in possible_flowrate_keys
                    or "flowrate" in normalized_key
                    or "flow_rate" in normalized_key
                )

                if key_is_flowrate:
                    numeric_value = convert_to_number(item)

                    if numeric_value is not None:
                        values.append(numeric_value)

                walk(item)

        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)

    return values


def create_baseline_forecast(
    values,
    horizon_days
):
    """
    Membuat baseline sederhana berdasarkan rata-rata maksimal
    30 nilai flowrate terbaru yang ditemukan.
    """

    if not values:
        raise HTTPException(
            status_code=500,
            detail=(
                "No numeric flowrate values were detected "
                "in the selected site JSON. The JSON field "
                "mapping must be adjusted to match the "
                "actual data structure."
            )
        )

    recent_values = values[-30:]
    recent_average = statistics.mean(recent_values)

    forecast_rows = []
    first_forecast_date = (
        date.today()
        + timedelta(days=1)
    )

    for index in range(horizon_days):
        forecast_date = (
            first_forecast_date
            + timedelta(days=index)
        )

        forecast_rows.append(
            {
                "date": forecast_date.isoformat(),
                "predicted_flowrate": round(
                    recent_average,
                    4
                ),
                "precipitation": None,
                "et0": None,
            }
        )

    return recent_average, forecast_rows


@app.post("/api/forecast")
def forecast(request: ForecastRequest):
    """
    Endpoint yang dipanggil oleh tombol Run forecast.
    """

    site_data = load_site_json(request.site)

    flowrate_values = extract_numeric_flowrates(
        site_data
    )

    if request.model == "baseline":
        forecast_average, forecast_rows = (
            create_baseline_forecast(
                values=flowrate_values,
                horizon_days=request.horizon_days
            )
        )

    elif request.model == "xgboost":
        raise HTTPException(
            status_code=501,
            detail=(
                "The XGBoost workflow from the original "
                "Streamlit application has not yet been "
                "connected to the Vercel API. Please use "
                "the Baseline model for the first test."
            )
        )

    elif request.model == "statistical":
        raise HTTPException(
            status_code=501,
            detail=(
                "The statistical workflow from the original "
                "Streamlit application has not yet been "
                "connected to the Vercel API. Please use "
                "the Baseline model for the first test."
            )
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported forecasting model."
        )

    historical_average = statistics.mean(
        flowrate_values
    )

    return {
        "site": request.site,
        "model": request.model,
        "horizon_days": request.horizon_days,
        "historical_average": round(
            historical_average,
            4
        ),
        "forecast_average": round(
            forecast_average,
            4
        ),
        "data_points_detected": len(
            flowrate_values
        ),
        "forecast": forecast_rows,
    }
