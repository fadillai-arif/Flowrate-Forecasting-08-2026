import os
import pandas as pd
from datetime import datetime

HISTORY_DIR = "data"
HISTORY_FILE = "data/forecast_history.csv"

HISTORY_COLS = [
    "run_timestamp", "site_name", "origin_month",
    "target_month", "model_name", "output_type", "yhat",
]


def save_forecast_run(forecast_df, model_name, origin_month, site_name):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for _, row in forecast_df.iterrows():
        target_month = pd.to_datetime(row["date"]).strftime("%Y-%m")
        base = {
            "run_timestamp": run_ts,
            "site_name": site_name,
            "origin_month": origin_month,
            "target_month": target_month,
            "model_name": model_name,
        }
        for output_type in ["forecast_flowrate", "lower_95", "upper_95"]:
            if output_type in forecast_df.columns and pd.notna(row.get(output_type)):
                r = base.copy()
                r["output_type"] = output_type
                r["yhat"] = float(row[output_type])
                rows.append(r)
    if not rows:
        return
    new_df = pd.DataFrame(rows)[HISTORY_COLS]
    if os.path.exists(HISTORY_FILE):
        existing = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(HISTORY_FILE, index=False)


def load_forecast_history(site_name=None):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLS)
    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLS)
    if site_name:
        df = df[df["site_name"] == site_name].copy()
    return df


def clear_forecast_history(site_name=None):
    if not os.path.exists(HISTORY_FILE):
        return
    if site_name is None:
        os.remove(HISTORY_FILE)
        return
    df = pd.read_csv(HISTORY_FILE)
    df = df[df["site_name"] != site_name]
    df.to_csv(HISTORY_FILE, index=False)
