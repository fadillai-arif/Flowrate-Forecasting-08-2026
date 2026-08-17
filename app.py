import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import itertools
import openmeteo_requests
import requests_cache
from retry_requests import retry
import xgboost as xgb
import json
import os
from utils.forecast_history import save_forecast_run, load_forecast_history, clear_forecast_history
from utils.forecast_accuracy import (
    select_latest_forecasts, build_accuracy_table,
    pick_winner_per_month, get_recommendation, window_summary_metrics,
)
from utils.backtest import run_rolling_backtest

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Flowrate Forecasting App", layout="wide")

SITES = {
    "Site B": {
        "data_file": "attached_assets/Pasted-date-flowrate-rainfall-2016-01-37-8-84-02878607-2016-02_1771560863535.txt",
        "added_data_file": "added_data_site_b.json",
        "latitude": 3.238704,
        "longitude": 98.526123,
    },
    "Site T": {
        "data_file": "attached_assets/site_t_flowrate.txt",
        "added_data_file": "added_data_site_t.json",
        "latitude": -5.445672,
        "longitude": 104.668215,
        "has_water_level": True,
    },
    "Site L": {
        "data_file": "attached_assets/site_l_flowrate.txt",
        "added_data_file": "added_data_site_l.json",
        "latitude": -6.765598,
        "longitude": 106.827264,
    },
    "Site K": {
        "data_file": "attached_assets/site_k_flowrate.txt",
        "added_data_file": "added_data_site_k.json",
        "latitude": -6.752897,
        "longitude": 106.746108,
    },
    "Site S": {
        "data_file": "attached_assets/site_s_flowrate.txt",
        "added_data_file": "added_data_site_s.json",
        "latitude": -6.736265,
        "longitude": 107.686182,
        "has_water_level": True,
    },
    "Site KL": {
        "data_file": "attached_assets/site_kl_flowrate.txt",
        "added_data_file": "added_data_site_kl.json",
        "latitude": -7.53718,
        "longitude": 110.47984,
    },
    "Site M": {
        "data_file": "attached_assets/site_m_flowrate.txt",
        "added_data_file": "added_data_site_m.json",
        "latitude": -8.253142,
        "longitude": 115.304637,
        "has_water_level": True,
    },
}

WEATHER_VARS = ["precipitation", "et0"]
WEATHER_LABELS = {
    "precipitation": "Precipitation (mm)",
    "et0": "ET0 Evapotranspiration (mm)",
}

TRANSLATIONS = {
    "ID": {
        "Data & Analisis": "Data & Analisis",
        "Stationarity Check": "Stationarity Check",
        "Model & Forecast": "Model & Forecast",
        "Forecast Accuracy": "Forecast Accuracy",
        "Input Data Baru": "Input Data Baru",
        "Export": "Export",
        "pilih_lokasi": "Pilih Lokasi",
        "header_data": "Data Overview",
        "total_data": "Total Data",
        "periode": "Periode",
        "flowrate_terakhir": "Flowrate Terakhir",
        "bulan": "bulan",
        "weather_ok": "Data cuaca Open-Meteo berhasil dimuat",
        "subheader_chart": "Data Flowrate & Cuaca",
        "subheader_lag": "Lag Correlation: Variabel Cuaca → Flowrate",
        "subheader_lag_wl": "Lag Correlation: Variabel Cuaca → Water Level",
        "best_lag": "Best Lag",
        "subheader_table": "Data Tabel",
        "header_stat": "Stationarity Check (ADF Test)",
        "stat_desc": "Augmented Dickey-Fuller test untuk mengecek apakah data stasioner (p-value < 0.05 = stasioner).",
        "stasioner": "Stasioner",
        "tidak_stasioner": "Tidak Stasioner",
        "setelah_diff": "Setelah differencing",
        "header_model": "Model & Forecast",
        "pilih_model": "Pilih Model",
        "test_size": "Test Size (bulan)",
        "forecast_months_label": "Forecast (bulan ke depan)",
        "asumsi_cuaca": "Asumsi Cuaca untuk Forecast (0 = pakai data terakhir)",
        "btn_run_model": "Jalankan Model",
        "model_success": "Model berhasil dijalankan!",
        "subheader_perf": "Flowrate Model Performance",
        "subheader_chart_fc": "Flowrate Forecast Chart",
        "subheader_table_fc": "Tabel Forecast Flowrate (Preview)",
        "subheader_perf_wl": "Water Level Model Performance",
        "subheader_chart_wl": "Water Level Forecast Chart",
        "subheader_table_wl": "Tabel Forecast Water Level (Preview)",
        "header_accuracy": "Forecast Accuracy — Rolling Backtest",
        "acc_desc": ("Evaluasi akurasi model secara *k*-step ahead walk-forward: "
                     "untuk setiap bulan evaluasi **T**, model dilatih menggunakan data sampai **T − k bulan**, "
                     "kemudian diprediksi secara iteratif sejauh **k langkah** ke depan menggunakan cuaca aktual, "
                     "lalu hasilnya dibandingkan dengan flowrate aktual bulan T."),
        "slider_n": "Jumlah bulan evaluasi",
        "slider_lag": "Forecast lag (bulan ke depan)",
        "multisel_models": "Model yang dievaluasi",
        "btn_run_eval": "Jalankan Evaluasi",
        "info_run": "Klik **Jalankan Evaluasi** untuk memulai backtest.",
        "warn_model": "Pilih minimal satu model.",
        "avg_acc": "Rata-rata Akurasi (%)",
        "mape": "MAPE (%)",
        "mae": "MAE",
        "subheader_rekomendasi": "Rekomendasi Model",
        "horizon_title": "Analisis Forecast Horizon (1–6 Bulan ke Depan)",
        "horizon_spinner": "Menganalisis forecast horizon 1–6 bulan dengan model terbaik…",
        "horizon_subtitle": "Akurasi rata-rata per forecast lag — model terbaik: **{model}**",
        "horizon_lag_col": "Lag (bulan ke depan)",
        "horizon_avg_col": "Rata-rata Akurasi (%)",
        "horizon_above90_col": "Bulan ≥90%",
        "horizon_total_col": "Total Bulan",
        "horizon_status_col": "Status",
        "horizon_ok": "✅ Andal",
        "horizon_warn": "⚠️ Cukup",
        "horizon_bad": "❌ Kurang",
        "horizon_recommend_ok": (
            "Untuk lokasi **{site}** dengan model **{model}**, "
            "forecast dapat dipercaya hingga **{n} bulan ke depan** (rata-rata akurasi ≥90%). "
            "Gunakan forecast maksimal **{n} bulan** saat presentasi ke manajemen."
        ),
        "horizon_recommend_partial": (
            "Untuk lokasi **{site}** dengan model **{model}**, "
            "lag berikut mencapai akurasi ≥90%: **{lags}**. "
            "Hindari lag yang tidak andal saat presentasi ke manajemen."
        ),
        "horizon_recommend_none": (
            "Untuk lokasi **{site}** dengan model **{model}**, "
            "tidak ada forecast horizon yang mencapai akurasi ≥90%. "
            "Pertimbangkan lebih banyak data historis atau gunakan perkiraan dengan kehati-hatian."
        ),
        "subheader_summary": "Ringkasan per Bulan — Model Terbaik",
        "subheader_detail": "Detail Semua Model per Bulan",
        "chart1_title": "Chart 1 — Akurasi per Bulan per Model",
        "chart2_title": "Chart 2 — Prediksi vs Aktual",
        "prediksi": "Prediksi",
        "aktual": "Aktual",
        "bulan_col": "Bulan",
        "model_col": "Model",
        "abs_error_col": "Abs Error",
        "akurasi_col": "Akurasi (%)",
        "lag_arrow": "← forecast dari",
        "forecast_lag_label": "Forecast lag",
        "header_input": "Input Data Baru",
        "desc_input": "Tambahkan beberapa data sekaligus. Data cuaca otomatis diambil dari Open-Meteo.",
        "form_label": "**Isi flowrate di bawah** (data cuaca akan otomatis diambil dari Open-Meteo):",
        "btn_add": "Tambah Semua Data",
        "subheader_added": "Data yang Sudah Ditambahkan",
        "col_aksi": "Aksi",
        "btn_hapus": "🗑️ Hapus",
        "btn_reset_all": "🗑️ Reset Semua Data Tambahan",
        "header_export": "Export Hasil",
        "subheader_export_fc": "Export Forecast ke CSV",
        "btn_download_fc": "Download Forecast CSV",
        "subheader_preview": "Preview Export",
        "subheader_export_full": "Export Data Lengkap + Forecast",
        "btn_download_full": "Download Data Lengkap + Forecast CSV",
        "info_no_model": "Jalankan model terlebih dahulu di tab 'Model & Forecast' untuk mengekspor hasil.",
    },
    "EN": {
        "Data & Analisis": "Data & Analysis",
        "Stationarity Check": "Stationarity Check",
        "Model & Forecast": "Model & Forecast",
        "Forecast Accuracy": "Forecast Accuracy",
        "Input Data Baru": "Add New Data",
        "Export": "Export",
        "pilih_lokasi": "Select Location",
        "header_data": "Data Overview",
        "total_data": "Total Data",
        "periode": "Period",
        "flowrate_terakhir": "Latest Flowrate",
        "bulan": "months",
        "weather_ok": "Open-Meteo weather data loaded successfully",
        "subheader_chart": "Flowrate & Weather Data",
        "subheader_lag": "Lag Correlation: Weather Variables → Flowrate",
        "subheader_lag_wl": "Lag Correlation: Weather Variables → Water Level",
        "best_lag": "Best Lag",
        "subheader_table": "Data Table",
        "header_stat": "Stationarity Check (ADF Test)",
        "stat_desc": "Augmented Dickey-Fuller test to check whether data is stationary (p-value < 0.05 = stationary).",
        "stasioner": "Stationary",
        "tidak_stasioner": "Not Stationary",
        "setelah_diff": "After differencing",
        "header_model": "Model & Forecast",
        "pilih_model": "Select Model",
        "test_size": "Test Size (months)",
        "forecast_months_label": "Forecast (months ahead)",
        "asumsi_cuaca": "Weather Assumptions for Forecast (0 = use latest data)",
        "btn_run_model": "Run Model",
        "model_success": "Model ran successfully!",
        "subheader_perf": "Flowrate Model Performance",
        "subheader_chart_fc": "Flowrate Forecast Chart",
        "subheader_table_fc": "Flowrate Forecast Table (Preview)",
        "subheader_perf_wl": "Water Level Model Performance",
        "subheader_chart_wl": "Water Level Forecast Chart",
        "subheader_table_wl": "Water Level Forecast Table (Preview)",
        "header_accuracy": "Forecast Accuracy — Rolling Backtest",
        "acc_desc": ("Walk-forward *k*-step ahead accuracy evaluation: "
                     "for each evaluation month **T**, the model is trained on data up to **T − k months**, "
                     "then iteratively forecast **k steps** ahead using actual weather, "
                     "and compared against the actual flowrate for month T."),
        "slider_n": "Number of evaluation months",
        "slider_lag": "Forecast lag (months ahead)",
        "multisel_models": "Models to evaluate",
        "btn_run_eval": "Run Evaluation",
        "info_run": "Click **Run Evaluation** to start the backtest.",
        "warn_model": "Select at least one model.",
        "avg_acc": "Average Accuracy (%)",
        "mape": "MAPE (%)",
        "mae": "MAE",
        "subheader_rekomendasi": "Model Recommendation",
        "horizon_title": "Forecast Horizon Analysis (1–6 Months Ahead)",
        "horizon_spinner": "Analysing forecast horizons 1–6 months with best model…",
        "horizon_subtitle": "Average accuracy per forecast lag — best model: **{model}**",
        "horizon_lag_col": "Lag (months ahead)",
        "horizon_avg_col": "Avg Accuracy (%)",
        "horizon_above90_col": "Months ≥90%",
        "horizon_total_col": "Total Months",
        "horizon_status_col": "Status",
        "horizon_ok": "✅ Reliable",
        "horizon_warn": "⚠️ Marginal",
        "horizon_bad": "❌ Unreliable",
        "horizon_recommend_ok": (
            "For location **{site}** using model **{model}**, "
            "forecasts are reliable up to **{n} months ahead** (avg accuracy ≥90%). "
            "Present forecasts up to **{n} months** to management."
        ),
        "horizon_recommend_partial": (
            "For location **{site}** using model **{model}**, "
            "the following lags reach ≥90% accuracy: **{lags}**. "
            "Avoid unreliable lags when presenting to management."
        ),
        "horizon_recommend_none": (
            "For location **{site}** using model **{model}**, "
            "no forecast horizon achieves ≥90% average accuracy. "
            "Consider more historical data or use forecasts with caution."
        ),
        "subheader_summary": "Monthly Summary — Best Model",
        "subheader_detail": "All Models Detail per Month",
        "chart1_title": "Chart 1 — Accuracy per Month per Model",
        "chart2_title": "Chart 2 — Forecast vs Actual",
        "prediksi": "Forecast",
        "aktual": "Actual",
        "bulan_col": "Month",
        "model_col": "Model",
        "abs_error_col": "Abs Error",
        "akurasi_col": "Accuracy (%)",
        "lag_arrow": "← forecast from",
        "forecast_lag_label": "Forecast lag",
        "header_input": "Add New Data",
        "desc_input": "Add multiple data points at once. Weather data is automatically fetched from Open-Meteo.",
        "form_label": "**Fill in flowrate below** (weather data will be fetched automatically from Open-Meteo):",
        "btn_add": "Add All Data",
        "subheader_added": "Added Data",
        "col_aksi": "Action",
        "btn_hapus": "🗑️ Delete",
        "btn_reset_all": "🗑️ Reset All Added Data",
        "header_export": "Export Results",
        "subheader_export_fc": "Export Forecast to CSV",
        "btn_download_fc": "Download Forecast CSV",
        "subheader_preview": "Export Preview",
        "subheader_export_full": "Export Full Data + Forecast",
        "btn_download_full": "Download Full Data + Forecast CSV",
        "info_no_model": "Run a model in the 'Model & Forecast' tab first to export results.",
    },
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    # Samakan semua variasi nama kolom water level jadi 'water_level'
    if "water level" in df.columns and "water_level" not in df.columns:
        rename_map["water level"] = "water_level"
    if "Water Level" in df.columns and "water_level" not in df.columns:
        rename_map["Water Level"] = "water_level"
    if "WaterLevel" in df.columns and "water_level" not in df.columns:
        rename_map["WaterLevel"] = "water_level"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


@st.cache_data(ttl=3600)
def fetch_openmeteo_monthly(start_date, end_date, latitude, longitude):
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum", "et0_fao_evapotranspiration"],
        "timezone": "Asia/Bangkok",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_precipitation = daily.Variables(0).ValuesAsNumpy()
    daily_et0 = daily.Variables(1).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["precipitation"] = daily_precipitation
    daily_data["et0"] = daily_et0

    daily_df = pd.DataFrame(data=daily_data)
    daily_df["date"] = daily_df["date"].dt.tz_localize(None)
    daily_df["year_month"] = daily_df["date"].dt.to_period("M")

    monthly = daily_df.groupby("year_month").agg(
        precipitation=("precipitation", "sum"),
        et0=("et0", "sum"),
    ).reset_index()
    monthly["date"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.drop(columns=["year_month"])
    for col in WEATHER_VARS:
        monthly[col] = monthly[col].round(2)

    return monthly


@st.cache_data
def load_initial_data(data_file):
    df = pd.read_csv(data_file, sep="\t")
    df = normalize_columns(df)  # ✅ TAMBAHKAN INI
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    return df



@st.cache_data
def load_data_with_weather(data_file, latitude, longitude):
    df = load_initial_data(data_file)
    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date_adj = (df["date"].iloc[-1] + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    weather = fetch_openmeteo_monthly(start_date, end_date_adj, latitude, longitude)
    merged = pd.merge(df, weather, on="date", how="left")
    if "rainfall" in merged.columns:
        merged["precipitation"] = merged["precipitation"].fillna(merged["rainfall"])
    for col in WEATHER_VARS:
        merged[col] = merged[col].ffill()
    return merged


def load_added_data(added_data_file):
    if os.path.exists(added_data_file):
        try:
            with open(added_data_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_added_data(rows, added_data_file):
    with open(added_data_file, "w") as f:
        json.dump(rows, f, indent=2)


def get_data(site_config):
    df = load_data_with_weather(site_config["data_file"], site_config["latitude"], site_config["longitude"]).copy()
    added_rows = load_added_data(site_config["added_data_file"])
    if len(added_rows) > 0:
        added_df = pd.DataFrame(added_rows)
        added_df = normalize_columns(added_df) 
        added_df["date"] = pd.to_datetime(added_df["date"])
        for col in WEATHER_VARS:
            if col not in added_df.columns:
                added_df[col] = np.nan
        if "rainfall" not in added_df.columns:
            added_df["rainfall"] = added_df.get("precipitation", np.nan)
        df = pd.concat([df, added_df], ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)
        for col in WEATHER_VARS:
            df[col] = df[col].ffill()
        if "water_level" in df.columns:
            df["water_level"] = df["water_level"].ffill()
    return df


def is_in_dry_season(month, start_month, end_month):
    if start_month <= end_month:
        return start_month <= month <= end_month
    else:
        return month >= start_month or month <= end_month


def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "Variable": name,
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 4),
        "Lags Used": result[2],
        "Observations": result[3],
        "Critical 1%": round(result[4]["1%"], 4),
        "Critical 5%": round(result[4]["5%"], 4),
        "Critical 10%": round(result[4]["10%"], 4),
        "Stationary": "Yes" if result[1] < 0.05 else "No",
    }


def find_best_lag(flowrate, variable, max_lag=6):
    n = len(flowrate)
    correlations = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            corr = np.corrcoef(variable[:n], flowrate[:n])[0, 1]
        else:
            corr = np.corrcoef(variable[:n - lag], flowrate[lag:n])[0, 1]
        correlations.append({"Lag": lag, "Correlation": round(corr, 4)})
    corr_df = pd.DataFrame(correlations)
    best_lag = corr_df.loc[corr_df["Correlation"].abs().idxmax(), "Lag"]
    return int(best_lag), corr_df


def create_lagged_features(df, best_lags, n_lags=6):
    features = pd.DataFrame(index=df.index)
    for var in WEATHER_VARS:
        features[var] = df[var].values
        for i in range(1, n_lags + 1):
            features[f"{var}_lag{i}"] = df[var].shift(i)
    for i in range(1, n_lags + 1):
        features[f"flowrate_lag{i}"] = df["flowrate"].shift(i)
    for var in WEATHER_VARS:
        bl = best_lags.get(var, 0)
        if bl > 0:
            features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
    features["month"] = df["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    return features


def auto_arima_search(train_series, max_p=3, max_d=2, max_q=3):
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            model = ARIMA(train_series, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def auto_sarimax_search(train_series, exog_train, best_arima_order):
    p, d, q = best_arima_order
    best_aic = np.inf
    best_seasonal = (0, 0, 0, 12)
    seasonal_options = [
        (0, 0, 0, 12),
        (1, 0, 0, 12),
        (0, 0, 1, 12),
        (1, 0, 1, 12),
        (1, 1, 0, 12),
        (0, 1, 1, 12),
        (1, 1, 1, 12),
    ]
    for seasonal in seasonal_options:
        try:
            model = SARIMAX(
                train_series,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=200)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_seasonal = seasonal
        except Exception:
            continue
    return best_seasonal, best_aic


def run_forecast(df, model_choice, test_size, forecast_months, future_weather=None, future_weather_sequence=None, target_col="flowrate"):
    n = len(df)
    train_size = n - test_size
    if train_size < 2:
        train_size = 2
        test_size = n - train_size
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    train_target = train_df[target_col]
    test_target = test_df[target_col]

    results = {}

    if model_choice == "SARIMAX":
        best_order, _ = auto_arima_search(train_target)
        results["arima_order"] = best_order

        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl
        results["weather_lags"] = str(best_lags)

        exog_cols = list(WEATHER_VARS)
        exog_df = df[WEATHER_VARS].copy()
        max_lag_val = max(best_lags.values()) if best_lags else 0
        for var in WEATHER_VARS:
            bl = best_lags[var]
            for lag in range(1, bl + 1):
                col_name = f"{var}_lag{lag}"
                exog_df[col_name] = df[var].shift(lag)
                exog_cols.append(col_name)

        valid_start = max_lag_val if max_lag_val > 0 else 0
        df_valid = df.iloc[valid_start:].reset_index(drop=True)
        exog_valid = exog_df.iloc[valid_start:].reset_index(drop=True)
        n_valid = len(df_valid)

        adj_test_size = min(test_size, n_valid - 2)
        if adj_test_size < 1: adj_test_size = 1
        
        train_size_s = n_valid - adj_test_size
        train_target_s = df_valid[target_col].iloc[:train_size_s]
        test_target_s = df_valid[target_col].iloc[train_size_s:]
        train_exog = exog_valid.iloc[:train_size_s].values
        test_exog = exog_valid.iloc[train_size_s:].values

        train_df = df_valid.iloc[:train_size_s]
        test_df = df_valid.iloc[train_size_s:]
        test_target = test_target_s

        best_seasonal, best_aic = auto_sarimax_search(train_target_s, train_exog, best_order)
        results["seasonal_order"] = best_seasonal
        results["aic"] = round(best_aic, 2)

        model = SARIMAX(
            train_target_s, exog=train_exog, order=best_order,
            seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        test_pred = fitted.forecast(steps=adj_test_size, exog=test_exog)

        full_model = SARIMAX(
            df_valid[target_col], exog=exog_valid.values, order=best_order,
            seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False,
        )
        full_fitted = full_model.fit(disp=False, maxiter=200)

        fw = future_weather or {}
        future_exog_rows = []
        for step in range(forecast_months):
            fw_step = future_weather_sequence[step] if (future_weather_sequence and step < len(future_weather_sequence)) else fw
            row = []
            for var in WEATHER_VARS:
                val = fw_step.get(var, df[var].iloc[-1])
                row.append(val)
            for var in WEATHER_VARS:
                bl = best_lags[var]
                recent = list(df[var].values[-(bl):]) if bl > 0 else []
                for lag in range(1, bl + 1):
                    if step - lag >= 0:
                        row.append(fw_step.get(var, df[var].iloc[-1]))
                    elif len(recent) >= (lag - step):
                        row.append(recent[-(lag - step)])
                    else:
                        row.append(fw_step.get(var, df[var].iloc[-1]))
            future_exog_rows.append(row)
        future_exog = np.array(future_exog_rows)

        forecast_result = full_fitted.get_forecast(steps=forecast_months, exog=future_exog)
        forecast_vals = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

    elif model_choice == "ML":
        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl

        features = pd.DataFrame(index=df.index)
        for var in WEATHER_VARS:
            features[var] = df[var].values
            for i in range(1, 7):
                features[f"{var}_lag{i}"] = df[var].shift(i)
        for i in range(1, 7):
            features[f"target_lag{i}"] = df[target_col].shift(i)
        for var in WEATHER_VARS:
            bl = best_lags.get(var, 0)
            if bl > 0:
                features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
        features["month"] = df["date"].dt.month
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        
        target = df[target_col]

        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        offset = n - len(features)
        adj_train = max(train_size - offset, 2)
        adj_test = len(features) - adj_train
        if adj_test < 1:
            adj_train = len(features) - 1
            adj_test = 1
        if len(features) < 2:
             raise ValueError(f"Data tidak cukup untuk model ML pada {target_col}. Butuh minimal 2 baris data valid.")

        feat_train = features.iloc[:adj_train]
        feat_test = features.iloc[adj_train:]
        tgt_train = target.iloc[:adj_train]
        tgt_test = target.iloc[adj_train:]

        scaler = StandardScaler()
        feat_train_sc = scaler.fit_transform(feat_train)
        feat_test_sc = scaler.transform(feat_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.05)
        rf.fit(feat_train_sc, tgt_train)
        gb.fit(feat_train_sc, tgt_train)

        rf_pred = rf.predict(feat_test_sc)
        gb_pred = gb.predict(feat_test_sc)
        test_pred = pd.Series((rf_pred + gb_pred) / 2, index=tgt_test.index)
        test_target = tgt_test

        fw = future_weather or {}
        forecast_vals_list = []
        temp_targets = list(df[target_col].values[-6:])
        temp_weather = {var: list(df[var].values[-6:]) for var in WEATHER_VARS}

        for step in range(forecast_months):
            fw_step = future_weather_sequence[step] if (future_weather_sequence and step < len(future_weather_sequence)) else fw
            new_feat = pd.DataFrame(index=[0])
            for var in WEATHER_VARS:
                current_val = fw_step.get(var, df[var].iloc[-1])
                new_feat[var] = current_val
                for i in range(1, 7):
                    tw = temp_weather[var]
                    new_feat[f"{var}_lag{i}"] = tw[-i] if i <= len(tw) else current_val
            for i in range(1, 7):
                new_feat[f"target_lag{i}"] = temp_targets[-i] if i <= len(temp_targets) else df[target_col].iloc[-1]
            for var in WEATHER_VARS:
                bl = best_lags.get(var, 0)
                if bl > 0:
                    tw = temp_weather[var]
                    new_feat[f"{var}_best_lag{bl}"] = tw[-bl] if bl <= len(tw) else fw_step.get(var, df[var].iloc[-1])
            month_num = (df["date"].iloc[-1].month + step) % 12 + 1
            new_feat["month"] = month_num
            new_feat["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            new_feat["month_cos"] = np.cos(2 * np.pi * month_num / 12)

            new_feat = new_feat[feat_train.columns]
            new_feat_sc = scaler.transform(new_feat)
            pred_val = (rf.predict(new_feat_sc)[0] + gb.predict(new_feat_sc)[0]) / 2
            forecast_vals_list.append(pred_val)

            temp_targets.append(pred_val)
            for var in WEATHER_VARS:
                temp_weather[var].append(fw_step.get(var, df[var].iloc[-1]))

        forecast_vals = pd.Series(forecast_vals_list)
        residuals = test_pred.values - test_target.values
        std_resid = np.std(residuals) if len(residuals) > 1 else 0.1
        lower = forecast_vals - 1.96 * std_resid
        upper = forecast_vals + 1.96 * std_resid
        conf_int = pd.DataFrame({"lower": lower.values, "upper": upper.values})

        results["weather_lags"] = str(best_lags)
        results["models"] = "RF + GBR Ensemble"

    elif model_choice == "XGBoost":
        best_lags = {}
        for var in WEATHER_VARS:
            bl, _ = find_best_lag(df[target_col].values, df[var].values)
            best_lags[var] = bl

        features = pd.DataFrame(index=df.index)
        for var in WEATHER_VARS:
            features[var] = df[var].values
            for i in range(1, 7):
                features[f"{var}_lag{i}"] = df[var].shift(i)
        for i in range(1, 7):
            features[f"target_lag{i}"] = df[target_col].shift(i)
        for var in WEATHER_VARS:
            bl = best_lags.get(var, 0)
            if bl > 0:
                features[f"{var}_best_lag{bl}"] = df[var].shift(bl)
        features["month"] = df["date"].dt.month
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        target = df[target_col]

        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]

        offset = n - len(features)
        adj_train = max(train_size - offset, 2)
        adj_test = len(features) - adj_train
        if adj_test < 1:
            adj_train = len(features) - 1
            adj_test = 1
        if len(features) < 2:
             raise ValueError(f"Data tidak cukup untuk XGBoost pada {target_col}. Butuh minimal 2 baris data valid.")

        feat_train = features.iloc[:adj_train]
        feat_test = features.iloc[adj_train:]
        tgt_train = target.iloc[:adj_train]
        tgt_test = target.iloc[adj_train:]

        scaler = StandardScaler()
        feat_train_sc = scaler.fit_transform(feat_train)
        feat_test_sc = scaler.transform(feat_test)

        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        xgb_model.fit(feat_train_sc, tgt_train)

        test_pred = pd.Series(xgb_model.predict(feat_test_sc), index=tgt_test.index)
        test_target = tgt_test

        fw = future_weather or {}
        forecast_vals_list = []
        temp_targets = list(df[target_col].values[-6:])
        temp_weather = {var: list(df[var].values[-6:]) for var in WEATHER_VARS}

        for step in range(forecast_months):
            fw_step = future_weather_sequence[step] if (future_weather_sequence and step < len(future_weather_sequence)) else fw
            new_feat = pd.DataFrame(index=[0])
            for var in WEATHER_VARS:
                current_val = fw_step.get(var, df[var].iloc[-1])
                new_feat[var] = current_val
                for i in range(1, 7):
                    tw = temp_weather[var]
                    new_feat[f"{var}_lag{i}"] = tw[-i] if i <= len(tw) else current_val
            for i in range(1, 7):
                new_feat[f"target_lag{i}"] = temp_targets[-i] if i <= len(temp_targets) else df[target_col].iloc[-1]
            for var in WEATHER_VARS:
                bl = best_lags.get(var, 0)
                if bl > 0:
                    tw = temp_weather[var]
                    new_feat[f"{var}_best_lag{bl}"] = tw[-bl] if bl <= len(tw) else fw_step.get(var, df[var].iloc[-1])
            month_num = (df["date"].iloc[-1].month + step) % 12 + 1
            new_feat["month"] = month_num
            new_feat["month_sin"] = np.sin(2 * np.pi * month_num / 12)
            new_feat["month_cos"] = np.cos(2 * np.pi * month_num / 12)

            new_feat = new_feat[feat_train.columns]
            new_feat_sc = scaler.transform(new_feat)
            pred_val = xgb_model.predict(new_feat_sc)[0]
            forecast_vals_list.append(pred_val)

            temp_targets.append(pred_val)
            for var in WEATHER_VARS:
                temp_weather[var].append(fw_step.get(var, df[var].iloc[-1]))

        forecast_vals = pd.Series(forecast_vals_list)
        residuals = test_pred.values - test_target.values
        std_resid = np.std(residuals) if len(residuals) > 1 else 0.1
        lower = forecast_vals - 1.96 * std_resid
        upper = forecast_vals + 1.96 * std_resid
        conf_int = pd.DataFrame({"lower": lower.values, "upper": upper.values})

        results["weather_lags"] = str(best_lags)
        results["models"] = "XGBoost"

    mae = mean_absolute_error(test_target, test_pred)
    rmse = np.sqrt(mean_squared_error(test_target, test_pred))
    r2 = r2_score(test_target, test_pred)
    results["mae"] = round(mae, 4)
    results["rmse"] = round(rmse, 4)
    results["r2"] = round(r2, 4)

    last_date = df["date"].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq="MS")

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        f"forecast_{target_col}": forecast_vals.values,
        "lower_95": conf_int.iloc[:, 0].values,
        "upper_95": conf_int.iloc[:, 1].values,
    })

    return results, train_df, test_df, test_pred, forecast_df


site_names = ["Site B", "Site T", "Site L", "Site K", "Site S", "Site KL", "Site M"]

# ── Language toggle (always visible in sidebar) ──────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "ID"

def t(key):
    return TRANSLATIONS[st.session_state.get("lang", "ID")][key]

with st.sidebar:
    st.markdown("**🌐 Language / Bahasa**")
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("🇮🇩 ID", key="btn_lang_id",
                     type="primary" if st.session_state.lang == "ID" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "ID"
            st.rerun()
    with lc2:
        if st.button("🇬🇧 EN", key="btn_lang_en",
                     type="primary" if st.session_state.lang == "EN" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "EN"
            st.rerun()
    st.divider()

selected_site = st.selectbox(
    t("pilih_lokasi"),
    site_names,
    key="selected_site"
)

site_config = SITES[selected_site]
WATERLEVEL_SITES = {"Site T", "Site S", "Site M"}
has_water_level = bool(site_config.get("has_water_level", False)) or (selected_site in WATERLEVEL_SITES)

st.title(f"Forecasting {selected_site}")
st.markdown(f"Analisis dan prediksi {'flowrate & water level' if has_water_level else 'flowrate'} berdasarkan data cuaca dari Open-Meteo (precipitation, evapotranspiration).")

if "added_rows" not in st.session_state or st.session_state.get("_last_site") != selected_site:
    st.session_state.added_rows = load_added_data(site_config["added_data_file"])
    if "forecast_results" in st.session_state:
        del st.session_state.forecast_results
    if "water_forecast_results" in st.session_state:
        del st.session_state.water_forecast_results
    st.session_state._last_site = selected_site

with st.spinner("Mengambil data cuaca dari Open-Meteo..."):
    try:
        df = get_data(site_config)
        weather_loaded = True
    except Exception as e:
        st.error(f"Gagal mengambil data cuaca dari Open-Meteo: {str(e)}. Menggunakan data rainfall dari file.")
        df = load_initial_data(site_config["data_file"])
        if "rainfall" in df.columns:
            df["precipitation"] = df["rainfall"]
        else:
            df["precipitation"] = 0.0
        df["et0"] = 4.0
        weather_loaded = False

TAB_OPTIONS = ["Data & Analisis", "Stationarity Check", "Model & Forecast", "Forecast Accuracy", "Input Data Baru", "Export"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = TAB_OPTIONS[0]

active_tab = st.radio("Navigasi", TAB_OPTIONS, index=TAB_OPTIONS.index(st.session_state.active_tab), horizontal=True, key="active_tab", label_visibility="collapsed", format_func=lambda x: TRANSLATIONS[st.session_state.get("lang", "ID")][x])

if active_tab == "Data & Analisis":
    st.header(t("header_data"))
    col1, col2, col3 = st.columns(3)
    col1.metric(t("total_data"), f"{len(df)} {t('bulan')}")
    col2.metric(t("periode"), f"{df['date'].dt.strftime('%Y-%m').iloc[0]} s/d {df['date'].dt.strftime('%Y-%m').iloc[-1]}")
    col3.metric(t("flowrate_terakhir"), f"{df['flowrate'].iloc[-1]:.2f}")

    if weather_loaded:
        st.success(t("weather_ok"))

    st.subheader(t("subheader_chart"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["date"], y=df["flowrate"], name="Flowrate", line=dict(color="#1f77b4", width=2)), secondary_y=False)
    if has_water_level and "water_level" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["water_level"], name="Water Level", line=dict(color="#2ca02c", width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation"], name="Precipitation", marker_color="rgba(44,160,44,0.3)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["et0"], name="ET0 (mm)", line=dict(color="#9467bd", width=1.5, dash="dash")), secondary_y=False)
    fig.update_layout(title="Flowrate, Water Level, ET0 & Precipitation", height=500, hovermode="x unified")
    fig.update_yaxes(title_text="Flowrate / Water Level / ET0", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    

    st.subheader(t("subheader_lag"))
    lag_results = {}
    for row_start in range(0, len(WEATHER_VARS), 3):
        row_vars = WEATHER_VARS[row_start:row_start + 3]
        cols_lag = st.columns(len(row_vars))
        for i, var in enumerate(row_vars):
            bl, corr_df = find_best_lag(df["flowrate"].values, df[var].values)
            lag_results[var] = {"best_lag": bl, "corr_df": corr_df}
            with cols_lag[i]:
                st.markdown(f"**{WEATHER_LABELS[var]}**")
                st.info(f"{t('best_lag')}: **{bl} {t('bulan')}**")
                fig_corr = go.Figure()
                colors = ["#ff7f0e" if lag == bl else "#1f77b4" for lag in corr_df["Lag"]]
                fig_corr.add_trace(go.Bar(x=corr_df["Lag"], y=corr_df["Correlation"], marker_color=colors))
                fig_corr.update_layout(height=300, xaxis_title="Lag", yaxis_title="Corr", margin=dict(t=10))
                st.plotly_chart(fig_corr, use_container_width=True)

    if has_water_level and "water_level" in df.columns:
        st.subheader(t("subheader_lag_wl"))
        for row_start in range(0, len(WEATHER_VARS), 3):
            row_vars = WEATHER_VARS[row_start:row_start + 3]
            cols_lag_wl = st.columns(len(row_vars))
            for i, var in enumerate(row_vars):
                bl_wl, corr_df_wl = find_best_lag(df["water_level"].values, df[var].values)
                with cols_lag_wl[i]:
                    st.markdown(f"**{WEATHER_LABELS[var]} → Water Level**")
                    st.success(f"{t('best_lag')}: **{bl_wl} {t('bulan')}**")
                    fig_corr_wl = go.Figure()
                    colors_wl = ["#2ca02c" if lag == bl_wl else "#d62728" for lag in corr_df_wl["Lag"]]
                    fig_corr_wl.add_trace(go.Bar(x=corr_df_wl["Lag"], y=corr_df_wl["Correlation"], marker_color=colors_wl))
                    fig_corr_wl.update_layout(height=300, xaxis_title="Lag", yaxis_title="Corr", margin=dict(t=10))
                    st.plotly_chart(fig_corr_wl, use_container_width=True)

    st.subheader(t("subheader_table"))
    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m")
    show_cols = ["date", "flowrate"]
    if has_water_level and "water_level" in display_df.columns:
        show_cols.append("water_level")
    if "rainfall" in display_df.columns:
        show_cols.append("rainfall")
    show_cols += WEATHER_VARS
    st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)

elif active_tab == "Stationarity Check":
    st.header(t("header_stat"))
    st.markdown(t("stat_desc"))

    test_vars = [("Flowrate", df["flowrate"])] + [(WEATHER_LABELS[v], df[v]) for v in WEATHER_VARS]
    for row_start in range(0, len(test_vars), 4):
        row_vars = test_vars[row_start:row_start + 4]
        cols_adf = st.columns(len(row_vars))
        for i, (name, series) in enumerate(row_vars):
            adf_result = adf_test(series, name)
            with cols_adf[i]:
                st.subheader(name.split(" ")[0])
                status = t("stasioner") if adf_result["Stationary"] == "Yes" else t("tidak_stasioner")
                color = "green" if adf_result["Stationary"] == "Yes" else "red"
                st.markdown(f":{color}[**{status}**]")
                st.json(adf_result)

    if adf_test(df["flowrate"], "Flowrate")["Stationary"] == "No":
        st.subheader("Differencing Flowrate (1st order)")
        diff_flow = df["flowrate"].diff().dropna()
        adf_diff = adf_test(diff_flow, "Flowrate (1st diff)")
        status_d = t("stasioner") if adf_diff["Stationary"] == "Yes" else t("tidak_stasioner")
        color_d = "green" if adf_diff["Stationary"] == "Yes" else "red"
        st.markdown(f"{t('setelah_diff')}: :{color_d}[**{status_d}**]")
        st.json(adf_diff)

elif active_tab == "Model & Forecast":
    st.header(t("header_model"))

    MONTH_OPTIONS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    MONTH_MAP = {m: i+1 for i, m in enumerate(MONTH_OPTIONS)}

    with st.expander("Simulasi Musim (Opsional)", expanded=False):
        enable_simulation = st.checkbox("Aktifkan Simulasi Musim", key="sim_enable", value=False)
        if enable_simulation:
            st.markdown("**Musim Kemarau Normal**")
            sc1, sc2 = st.columns(2)
            with sc1:
                sim_dry_start = st.selectbox("Mulai Musim Kemarau Normal", MONTH_OPTIONS, index=5, key="sim_dry_start")
            with sc2:
                sim_dry_end = st.selectbox("Akhir Musim Kemarau Normal", MONTH_OPTIONS, index=8, key="sim_dry_end")

            st.markdown("**Fenomena ENSO (Opsional)**")
            enso_mode = st.radio("Pilih Fenomena ENSO", ["Tidak Ada", "El Niño", "La Niña"], index=0, horizontal=True, key="sim_enso_mode")

            if enso_mode != "Tidak Ada":
                st.markdown(f"**Prediksi Pergeseran Musim Kemarau akibat {enso_mode}**")
                ec1, ec2 = st.columns(2)
                with ec1:
                    sim_enso_dry_start = st.selectbox(f"Prediksi Awal Kemarau ({enso_mode})", MONTH_OPTIONS, index=4, key="sim_enso_dry_start")
                with ec2:
                    sim_enso_dry_end = st.selectbox(f"Prediksi Akhir Kemarau ({enso_mode})", MONTH_OPTIONS, index=9, key="sim_enso_dry_end")

                enso_info = {
                    "El Niño": "Musim kemarau lebih panjang & lebih kering. Presipitasi musim kemarau -40%, musim hujan -10%. ET0 musim kemarau +10%.",
                    "La Niña": "Musim hujan lebih panjang & lebih basah. Presipitasi musim hujan +35%, musim kemarau +10%. ET0 -5%.",
                }
                st.info(enso_info[enso_mode])
        else:
            enso_mode = "Tidak Ada"

    with st.form("model_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            model_choice = st.selectbox(t("pilih_model"), ["XGBoost", "ML (Ensemble RF+GBR)", "SARIMAX"])
        with col2:
            # Pastikan max_value > min_value untuk menghindari StreamlitAPIException
            # Jika data terlalu sedikit, kita turunkan min_value-nya
            calculated_max = len(df) // 3
            if calculated_max < 6:
                slider_min = max(2, calculated_max)
                slider_max = 6
            else:
                slider_min = 6
                slider_max = min(36, calculated_max)
            
            if slider_min >= slider_max:
                slider_max = slider_min + 1
                
            test_size = st.slider(t("test_size"), min_value=slider_min, max_value=slider_max, value=slider_min)
        with col3:
            forecast_months = st.slider(t("forecast_months_label"), min_value=1, max_value=24, value=6)

        st.markdown(f"**{t('asumsi_cuaca')}**")
        cw1, cw2 = st.columns(2)
        with cw1:
            fw_precip = st.number_input("Precipitation (mm/bulan)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        with cw2:
            fw_et0 = st.number_input("ET0 (mm/bulan)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

        run_model = st.form_submit_button(t("btn_run_model"), type="primary", use_container_width=True)

    # Run Model Logic
    if run_model:
        model_map = {
            "SARIMAX": "SARIMAX",
            "ML (Ensemble RF+GBR)": "ML",
            "XGBoost": "XGBoost",
        }
        model_key = model_map.get(model_choice, "XGBoost")
        future_weather_vals = {}
        if fw_precip > 0:
            future_weather_vals["precipitation"] = fw_precip
        if fw_et0 > 0:
            future_weather_vals["et0"] = fw_et0
        if not future_weather_vals:
            future_weather_vals = None

        # Build per-month weather sequence if simulation is active
        future_weather_seq = None
        sim_active = st.session_state.get("sim_enable", False)
        if sim_active:
            ds_start_key = st.session_state.get("sim_dry_start", "Jun")
            ds_end_key = st.session_state.get("sim_dry_end", "Sep")
            enso_key = st.session_state.get("sim_enso_mode", "Tidak Ada")
            ds_start_num = MONTH_MAP.get(ds_start_key, 6)
            ds_end_num = MONTH_MAP.get(ds_end_key, 9)

            if enso_key != "Tidak Ada":
                enso_ds_start_key = st.session_state.get("sim_enso_dry_start", ds_start_key)
                enso_ds_end_key = st.session_state.get("sim_enso_dry_end", ds_end_key)
                active_ds_start = MONTH_MAP.get(enso_ds_start_key, ds_start_num)
                active_ds_end = MONTH_MAP.get(enso_ds_end_key, ds_end_num)
            else:
                active_ds_start = ds_start_num
                active_ds_end = ds_end_num

            if enso_key == "El Niño":
                dry_precip_mult, wet_precip_mult = 0.60, 0.90
                dry_et0_mult, wet_et0_mult = 1.10, 1.00
            elif enso_key == "La Niña":
                dry_precip_mult, wet_precip_mult = 1.10, 1.35
                dry_et0_mult, wet_et0_mult = 0.95, 1.00
            else:
                dry_precip_mult = wet_precip_mult = dry_et0_mult = wet_et0_mult = 1.0

            monthly_avg = df.groupby(df["date"].dt.month)[WEATHER_VARS].mean()
            last_date_seq = df["date"].iloc[-1]
            future_weather_seq = []
            for step in range(forecast_months):
                fdate = last_date_seq + pd.DateOffset(months=step + 1)
                m = fdate.month
                base_p = monthly_avg.loc[m, "precipitation"] if m in monthly_avg.index else df["precipitation"].mean()
                base_e = monthly_avg.loc[m, "et0"] if m in monthly_avg.index else df["et0"].mean()
                if is_in_dry_season(m, active_ds_start, active_ds_end):
                    future_weather_seq.append({"precipitation": round(base_p * dry_precip_mult, 2), "et0": round(base_e * dry_et0_mult, 2)})
                else:
                    future_weather_seq.append({"precipitation": round(base_p * wet_precip_mult, 2), "et0": round(base_e * wet_et0_mult, 2)})

        with st.spinner("Sedang menjalankan model... Mohon tunggu."):
            try:
                # Flowrate forecast
                results, train_df, test_df, test_pred, forecast_df = run_forecast(
                    df, model_key, test_size, forecast_months, future_weather_vals,
                    future_weather_sequence=future_weather_seq, target_col="flowrate"
                )
                st.session_state.forecast_results = {
                    "results": results,
                    "train_df": train_df,
                    "test_df": test_df,
                    "test_pred": test_pred,
                    "forecast_df": forecast_df,
                    "model_choice": model_choice,
                    "simulation_active": sim_active,
                    "enso_mode": st.session_state.get("sim_enso_mode", "Tidak Ada") if sim_active else "Tidak Ada",
                }

                # Water level forecast if applicable
                if has_water_level:
                    w_results, w_train_df, w_test_df, w_test_pred, w_forecast_df = run_forecast(
                        df, model_key, test_size, forecast_months, future_weather_vals,
                        future_weather_sequence=future_weather_seq, target_col="water_level"
                    )
                    st.session_state.water_forecast_results = {
                        "results": w_results,
                        "train_df": w_train_df,
                        "test_df": w_test_df,
                        "test_pred": w_test_pred,
                        "forecast_df": w_forecast_df,
                        "model_choice": model_choice,
                    }
                # Persist forecast to history
                origin_month = df["date"].iloc[-1].strftime("%Y-%m")
                try:
                    save_forecast_run(forecast_df, model_choice, origin_month, selected_site)
                except Exception:
                    pass
                st.success(t("model_success"))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Display Results Logic
    if "forecast_results" in st.session_state:
        fr = st.session_state.forecast_results
        res = fr["results"]
        fc_df = fr["forecast_df"]

        if fr.get("simulation_active"):
            enso_label = fr.get("enso_mode", "Tidak Ada")
            sim_msg = "Simulasi Musim aktif"
            if enso_label != "Tidak Ada":
                sim_msg += f" — Fenomena **{enso_label}**"
            st.info(f"Simulasi: {sim_msg}")

        st.subheader(t("subheader_perf"))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"{res['mae']:.4f}")
        mc2.metric("RMSE", f"{res['rmse']:.4f}")
        mc3.metric("R²", f"{res['r2']:.4f}")

        st.subheader(t("subheader_chart_fc"))
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df["date"], y=df["flowrate"], name="Historical", line=dict(color="#1f77b4", width=2)))
        fig_fc.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_flowrate"], name="Forecast", line=dict(color="#d62728", width=2)))
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
            y=pd.concat([fc_df["upper_95"], fc_df["lower_95"][::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% CI",
        ))
        fig_fc.update_layout(title="Flowrate Forecast", xaxis_title="Date", yaxis_title="Flowrate", height=450)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.subheader(t("subheader_table_fc"))
        cols = ["date", "forecast_flowrate", "lower_95", "upper_95"]
        preview_fc = fc_df[cols].copy()
        preview_fc["date"] = preview_fc["date"].dt.strftime("%Y-%m")
        st.dataframe(preview_fc, use_container_width=True, hide_index=True)
        
        if has_water_level and "water_forecast_results" in st.session_state:
            wr = st.session_state.water_forecast_results
            w_res = wr["results"]
            w_fc_df = wr["forecast_df"]
            
            st.subheader(t("subheader_perf_wl"))
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("MAE", f"{w_res['mae']:.4f}")
            wc2.metric("RMSE", f"{w_res['rmse']:.4f}")
            wc3.metric("R²", f"{w_res['r2']:.4f}")

            st.subheader(t("subheader_chart_wl"))
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=df["date"], y=df["water_level"], name="Historical", line=dict(color="#1f77b4", width=2)))
            fig_w.add_trace(go.Scatter(x=w_fc_df["date"], y=w_fc_df["forecast_water_level"], name="Forecast", line=dict(color="#2ca02c", width=2)))
            fig_w.add_trace(go.Scatter(
                x=pd.concat([w_fc_df["date"], w_fc_df["date"][::-1]]),
                y=pd.concat([w_fc_df["upper_95"], w_fc_df["lower_95"][::-1]]),
                fill="toself", fillcolor="rgba(44,160,44,0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% CI",
            ))
            fig_w.update_layout(title="Water Level Forecast", xaxis_title="Date", yaxis_title="Water Level", height=450)
            st.plotly_chart(fig_w, use_container_width=True)
            st.subheader(t("subheader_table_wl"))
            cols_w = ["date", "forecast_water_level", "lower_95", "upper_95"]
            preview_w = w_fc_df[cols_w].copy()
            preview_w["date"] = preview_w["date"].dt.strftime("%Y-%m")
            st.dataframe(preview_w, use_container_width=True, hide_index=True)


elif active_tab == "Forecast Accuracy":
    st.header(t("header_accuracy"))
    st.markdown(t("acc_desc"))

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        n_eval_months = st.slider(t("slider_n"), min_value=2, max_value=12, value=6, key="fa_n_months")
    with ctrl2:
        forecast_lag = st.slider(t("slider_lag"), min_value=1, max_value=6, value=3,
                                  key="fa_forecast_lag",
                                  help="k = berapa bulan sebelumnya forecast dibuat. "
                                       "Contoh: k=3 → forecast Desember dibuat dari data September.")
    with ctrl3:
        bt_models = st.multiselect(t("multisel_models"), ["XGBoost", "ML", "SARIMAX"],
                                    default=["XGBoost", "ML", "SARIMAX"], key="fa_models")
    with ctrl4:
        st.markdown('<div style="padding-top:28px"></div>', unsafe_allow_html=True)
        run_bt = st.button(t("btn_run_eval"), type="primary", key="fa_run", use_container_width=True)

    # Show which months will be evaluated and their cutoff dates
    all_avail_dates = sorted(df["date"].unique())
    eval_dates_preview = all_avail_dates[-n_eval_months:]
    preview_info = []
    for d in eval_dates_preview:
        eval_ts = pd.Timestamp(d)
        cutoff_ts = eval_ts - pd.DateOffset(months=forecast_lag)
        preview_info.append(f"{eval_ts.strftime('%b %Y')} {t('lag_arrow')} {cutoff_ts.strftime('%b %Y')}")
    st.caption(
        f"**{t('forecast_lag_label')} = {forecast_lag} {t('bulan')}** — "
        f"{' | '.join(preview_info)}"
    )

    st.divider()

    # ── Run / load backtest results ───────────────────────────────────────────
    bt_key = f"bt_result_{selected_site}"
    bt_cfg_key = f"bt_cfg_{selected_site}"

    bt_horizon_key = f"bt_horizon_{selected_site}"
    bt_horizon_model_key = f"bt_horizon_model_{selected_site}"

    if run_bt:
        if not bt_models:
            st.warning(t("warn_model"))
        else:
            with st.spinner(
                f"{forecast_lag}-step ahead backtest "
                f"({n_eval_months} {t('bulan')}, {len(bt_models)} models)..."
            ):
                bt_result = run_rolling_backtest(
                    df, n_months=n_eval_months, forecast_lag=forecast_lag, models=bt_models
                )
            st.session_state[bt_key] = bt_result
            st.session_state[bt_cfg_key] = {
                "n": n_eval_months, "models": bt_models, "lag": forecast_lag
            }

            # ── Horizon analysis: run lags 1–6 for the best model ────────────
            if not bt_result.empty:
                avg_acc_tmp = bt_result.groupby("model_name")["accuracy_pct"].mean()
                best_model_horizon = avg_acc_tmp.idxmax()
                with st.spinner(t("horizon_spinner")):
                    horizon_rows = []
                    for _lag in range(1, 7):
                        lag_res = run_rolling_backtest(
                            df, n_months=n_eval_months,
                            forecast_lag=_lag, models=[best_model_horizon]
                        )
                        if not lag_res.empty:
                            _avg = lag_res["accuracy_pct"].mean()
                            _above = int((lag_res["accuracy_pct"] >= 90).sum())
                            _total = len(lag_res)
                            horizon_rows.append({
                                "lag": _lag,
                                "avg_accuracy": round(_avg, 1),
                                "months_above_90": _above,
                                "total_months": _total,
                            })
                st.session_state[bt_horizon_key] = pd.DataFrame(horizon_rows)
                st.session_state[bt_horizon_model_key] = best_model_horizon

    bt_result = st.session_state.get(bt_key, pd.DataFrame())

    if bt_result.empty:
        st.info(t("info_run"))
    else:
        cfg = st.session_state.get(bt_cfg_key, {})
        st.success(
            f"Backtest — {cfg.get('n', '?')} {t('bulan')}, "
            f"{t('forecast_lag_label')} **{cfg.get('lag', '?')} {t('bulan')}**, "
            f"model: {', '.join(cfg.get('models', []))}"
        )

        # ── Horizon recommendation banner (shown immediately) ─────────────────
        _horizon_df_early = st.session_state.get(bt_horizon_key, pd.DataFrame())
        _best_model_early = st.session_state.get(bt_horizon_model_key, "")
        if not _horizon_df_early.empty and _best_model_early:
            _reliable_early = _horizon_df_early[_horizon_df_early["avg_accuracy"] >= 90]["lag"].tolist()
            _max_consec_early = 0
            for _l in range(1, 7):
                if _l in _reliable_early:
                    _max_consec_early = _l
                else:
                    break
            if _max_consec_early > 0:
                st.info(t("horizon_recommend_ok").format(
                    site=selected_site, model=_best_model_early, n=_max_consec_early
                ))
            elif _reliable_early:
                _lags_str = ", ".join([f"{l} {t('bulan')}" for l in _reliable_early])
                st.warning(t("horizon_recommend_partial").format(
                    site=selected_site, model=_best_model_early, lags=_lags_str
                ))
            else:
                st.error(t("horizon_recommend_none").format(
                    site=selected_site, model=_best_model_early
                ))

        # ── Summary metrics ───────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric(t("avg_acc"), f"{bt_result['accuracy_pct'].mean():.2f}%")
        m2.metric(t("mape"), f"{bt_result['ape'].mean():.2f}%")
        m3.metric(t("mae"), f"{bt_result['abs_error'].mean():.4f}")

        st.divider()

        # ── Chart 1: Akurasi per bulan per model ─────────────────────────────
        model_colors = {"XGBoost": "#1f77b4", "ML": "#ff7f0e", "SARIMAX": "#2ca02c"}

        st.subheader(t("chart1_title"))
        fig_bt = go.Figure()
        for mdl in bt_result["model_name"].unique():
            sub = bt_result[bt_result["model_name"] == mdl].sort_values("target_month")
            fig_bt.add_trace(go.Bar(
                name=mdl,
                x=sub["target_month"],
                y=sub["accuracy_pct"],
                text=[f"{v:.1f}%" for v in sub["accuracy_pct"]],
                textposition="auto",
                marker_color=model_colors.get(mdl, "#999"),
            ))
        fig_bt.add_hline(y=90, line_dash="dot", line_color="green", annotation_text="90%")
        fig_bt.add_hline(y=70, line_dash="dot", line_color="orange", annotation_text="70%")
        fig_bt.update_layout(
            barmode="group",
            yaxis=dict(title=t("akurasi_col"), range=[0, 115]),
            xaxis_title=t("bulan_col"),
            height=400,
            legend_title=t("model_col"),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # ── Chart 2: Prediksi vs Aktual per model ────────────────────────────
        st.subheader(t("chart2_title"))
        fig_vs = go.Figure()
        for mdl in bt_result["model_name"].unique():
            sub = bt_result[bt_result["model_name"] == mdl].sort_values("target_month")
            fig_vs.add_trace(go.Scatter(
                name=f"{mdl} ({t('prediksi').lower()})",
                x=sub["target_month"], y=sub["yhat"],
                mode="lines+markers",
                marker_color=model_colors.get(mdl, "#999"),
                line=dict(dash="dash"),
            ))
        actual_line = bt_result.drop_duplicates("target_month").sort_values("target_month")
        fig_vs.add_trace(go.Scatter(
            name=t("aktual"),
            x=actual_line["target_month"], y=actual_line["actual"],
            mode="lines+markers",
            marker_color="black", line=dict(width=2),
        ))
        fig_vs.update_layout(
            yaxis_title="Flowrate",
            xaxis_title=t("bulan_col"),
            height=380,
            legend_title=t("model_col"),
        )
        st.plotly_chart(fig_vs, use_container_width=True)

        # ── Rekomendasi ───────────────────────────────────────────────────────
        st.divider()
        avg_acc = bt_result.groupby("model_name")["accuracy_pct"].mean().sort_values(ascending=False)
        best_model = avg_acc.index[0]
        st.subheader(t("subheader_rekomendasi"))
        st.success(
            f"**{best_model}** — {t('avg_acc').rstrip(' (%)')}: {avg_acc[best_model]:.1f}% "
            f"({cfg.get('n','?')} {t('bulan')})"
        )

        # ── Horizon Analysis ─────────────────────────────────────────────────
        horizon_df = st.session_state.get(bt_horizon_key, pd.DataFrame())
        best_model_h = st.session_state.get(bt_horizon_model_key, best_model)

        if not horizon_df.empty:
            st.markdown(f"#### {t('horizon_title')}")
            st.markdown(t("horizon_subtitle").format(model=best_model_h))

            # Bar chart: accuracy per lag
            _model_clr = model_colors.get(best_model_h, "#1f77b4")
            bar_colors = [
                _model_clr if v >= 90 else ("#ff7f0e" if v >= 70 else "#d62728")
                for v in horizon_df["avg_accuracy"]
            ]
            fig_hor = go.Figure()
            fig_hor.add_trace(go.Bar(
                x=horizon_df["lag"],
                y=horizon_df["avg_accuracy"],
                text=[f"{v:.1f}%" for v in horizon_df["avg_accuracy"]],
                textposition="outside",
                marker_color=bar_colors,
                width=0.55,
            ))
            fig_hor.add_hline(y=90, line_dash="dot", line_color="green",
                               annotation_text="90%", annotation_position="top right")
            fig_hor.add_hline(y=70, line_dash="dot", line_color="orange",
                               annotation_text="70%", annotation_position="top right")
            fig_hor.update_layout(
                xaxis=dict(title=t("horizon_lag_col"), tickmode="array",
                           tickvals=list(horizon_df["lag"]),
                           ticktext=[f"{l}" for l in horizon_df["lag"]]),
                yaxis=dict(title=t("horizon_avg_col"), range=[0, 115]),
                height=320,
                showlegend=False,
                margin=dict(t=30, b=10),
            )
            st.plotly_chart(fig_hor, use_container_width=True)

            # Detail table
            def _hor_status(v):
                if v >= 90:
                    return t("horizon_ok")
                elif v >= 70:
                    return t("horizon_warn")
                return t("horizon_bad")

            hor_display = horizon_df[["lag", "avg_accuracy"]].copy()
            hor_display[t("horizon_status_col")] = horizon_df["avg_accuracy"].apply(_hor_status)
            hor_display = hor_display.rename(columns={
                "lag": t("horizon_lag_col"),
                "avg_accuracy": t("horizon_avg_col"),
            })
            st.dataframe(hor_display.reset_index(drop=True), use_container_width=True, hide_index=True)



elif active_tab == "Input Data Baru":
    st.header(t("header_input"))
    st.markdown(t("desc_input"))

    last_date = df["date"].iloc[-1]
    next_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq="MS")

    with st.form("input_data_form"):
        st.markdown(t("form_label"))
        input_dates = []
        input_flowrates = []
        input_waterlevels = []
        
        for i in range(6):
            if has_water_level:
                c1, c2, c3 = st.columns([1, 1, 1])
            else:
                c1, c2 = st.columns([1, 1])

            with c1:
                d = st.text_input(
                    f"Date {i+1} (YYYY-MM)",
                    value=next_dates[i].strftime("%Y-%m"),
                    key=f"date_{i}"
                )

            with c2:
                f = st.number_input(
                    f"Flowrate {i+1}",
                    min_value=0.0, max_value=1000.0,
                    value=0.0, step=0.01,
                    key=f"flow_{i}"
                )

            input_dates.append(d)
            input_flowrates.append(f)

            if has_water_level:
                with c3:
                    wl = st.number_input(
                        f"Water Level {i+1}",
                        min_value=0.0, max_value=100.0,
                        value=0.0, step=0.01,
                        key=f"wl_{i}"
                    )
                input_waterlevels.append(wl)


        submitted_data = st.form_submit_button(t("btn_add"), type="primary", use_container_width=True)

    if submitted_data:
        edited_df = pd.DataFrame({
            "date": input_dates,
            "flowrate": input_flowrates
        })

        if has_water_level:
            edited_df["water_level"] = input_waterlevels

        edited_df["flowrate"] = edited_df["flowrate"].replace(0, np.nan)
        if has_water_level:
            edited_df["water_level"] = edited_df["water_level"].replace(0, np.nan)

        # Valid jika minimal salah satu terisi (flowrate atau water_level)
        cols_check = ["flowrate"] + (["water_level"] if has_water_level else [])
        valid_rows = edited_df.dropna(subset=cols_check, how="all").copy()

        if len(valid_rows) == 0:
            st.error("Tidak ada data valid (flowrate &/atau water level masih kosong semua).")
        else:
            existing_dates = df["date"].tolist()
            added_count = 0
            skipped = []
            new_rows_to_add = []
            for _, row in valid_rows.iterrows():
                try:
                    date_dt = pd.to_datetime(row["date"], format="%Y-%m")
                    if date_dt in existing_dates:
                        skipped.append(row["date"])
                    else:
                        new_row = {
                            "date": row["date"],
                            "flowrate": float(row["flowrate"]),
                        }

                        if has_water_level and "water_level" in row:
                            new_row["water_level"] = float(row["water_level"])

                        new_rows_to_add.append(new_row)
                        existing_dates.append(date_dt)
                        added_count += 1
                except Exception:
                    skipped.append(str(row["date"]))

            if added_count > 0:
                weather_fetched = False
                try:
                    dates_to_fetch = [pd.to_datetime(r["date"], format="%Y-%m") for r in new_rows_to_add]
                    min_date = min(dates_to_fetch).strftime("%Y-%m-%d")
                    max_date = (max(dates_to_fetch) + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    weather_new = fetch_openmeteo_monthly(min_date, max_date, site_config["latitude"], site_config["longitude"])
                    weather_dict = {}
                    for _, wr in weather_new.iterrows():
                        key = wr["date"].strftime("%Y-%m")
                        w_entry = {}
                        for wv in WEATHER_VARS:
                            w_entry[wv] = wr[wv]
                        weather_dict[key] = w_entry
                    for nr in new_rows_to_add:
                        if nr["date"] in weather_dict:
                            nr.update(weather_dict[nr["date"]])
                            weather_fetched = True
                        else:
                            for wv in WEATHER_VARS:
                                nr[wv] = df[wv].iloc[-1]
                except Exception:
                    for nr in new_rows_to_add:
                        for wv in WEATHER_VARS:
                            nr[wv] = df[wv].iloc[-1]
                    st.warning("Gagal mengambil data cuaca dari Open-Meteo. Menggunakan data cuaca terakhir sebagai pengganti.")

                for nr in new_rows_to_add:
                    st.session_state.added_rows.append(nr)
                save_added_data(st.session_state.added_rows, site_config["added_data_file"])

                if "forecast_results" in st.session_state:
                    del st.session_state.forecast_results
                load_data_with_weather.clear()

                if weather_fetched:
                    st.success(f"**{added_count}** {t('desc_input').split('.')[0].lower()} ✓ (Open-Meteo)")
                else:
                    st.success(f"**{added_count} data** ✓")
            if skipped:
                st.warning(f"Data berikut dilewati (sudah ada/format salah): {', '.join(skipped)}")
            if added_count > 0:
                st.rerun()

    if len(st.session_state.added_rows) > 0:
        st.subheader(t("subheader_added"))

        # Per-row delete — show each row with its own Hapus button
        # Only show meaningful columns; exclude legacy 'rainfall' (same as precipitation)
        display_keys = ["date", "flowrate"] + (["water_level"] if has_water_level else []) + ["precipitation", "et0"]
        # Header row
        hdr_cols = st.columns([2, 2, 2, 2, 1])
        for ci, key in enumerate(display_keys[:4]):
            hdr_cols[ci].markdown(f"<small><b>{key}</b></small>", unsafe_allow_html=True)
        hdr_cols[4].markdown(f"<small><b>{t('col_aksi')}</b></small>", unsafe_allow_html=True)

        delete_idx = None
        for i, row in enumerate(st.session_state.added_rows):
            cols_row = st.columns([2, 2, 2, 2, 1])
            for ci, key in enumerate(display_keys[:4]):
                val = row.get(key, "—")
                if isinstance(val, float):
                    cols_row[ci].markdown(f"{val:.4f}")
                else:
                    cols_row[ci].markdown(str(val))
            if cols_row[4].button(t("btn_hapus"), key=f"del_row_{i}", help=f"{t('col_aksi')} {row.get('date','')}", type="secondary"):
                delete_idx = i

        if delete_idx is not None:
            st.session_state.added_rows.pop(delete_idx)
            save_added_data(st.session_state.added_rows, site_config["added_data_file"])
            if "forecast_results" in st.session_state:
                del st.session_state.forecast_results
            if "water_forecast_results" in st.session_state:
                del st.session_state.water_forecast_results
            load_data_with_weather.clear()
            st.rerun()

        st.divider()
        if st.button(t("btn_reset_all"), type="secondary"):
            st.session_state.added_rows = []
            save_added_data([], site_config["added_data_file"])
            if "forecast_results" in st.session_state:
                del st.session_state.forecast_results
            if "water_forecast_results" in st.session_state:
                del st.session_state.water_forecast_results
            load_data_with_weather.clear()
            st.rerun()

elif active_tab == "Export":
    st.header(t("header_export"))

    if "forecast_results" in st.session_state:
        forecast_df = st.session_state.forecast_results["forecast_df"]

        st.subheader(t("subheader_export_fc"))
        export_df = forecast_df.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m")
        csv = export_df.to_csv(index=False)
        st.download_button(label=t("btn_download_fc"), data=csv, file_name="forecast_flowrate.csv", mime="text/csv", type="primary", use_container_width=True)

        st.subheader(t("subheader_preview"))
        st.dataframe(export_df, use_container_width=True, hide_index=True)

        st.subheader(t("subheader_export_full"))
        full_export = df.copy()
        full_export["date"] = full_export["date"].dt.strftime("%Y-%m")
        full_export["type"] = "historical"

        fc_export = forecast_df.copy()
        fc_export["date"] = fc_export["date"].dt.strftime("%Y-%m")
        fc_export = fc_export.rename(columns={"forecast_flowrate": "flowrate"})
        fc_export["type"] = "forecast"

        base_cols = ["date", "flowrate", "type"]
        combined = pd.concat([full_export[base_cols], fc_export[base_cols]], ignore_index=True)
        csv_full = combined.to_csv(index=False)
        st.download_button(label=t("btn_download_full"), data=csv_full, file_name="full_data_with_forecast.csv", mime="text/csv", use_container_width=True)
    else:
        st.info(t("info_no_model"))
