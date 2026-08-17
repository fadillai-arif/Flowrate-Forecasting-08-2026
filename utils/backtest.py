"""
Rolling walk-forward backtest with configurable forecast lag (k-step ahead).

For each evaluation month T:
  - Training cutoff : df[date <= T - k months]
  - Iteratively forecast k steps ahead using actual weather for each step
  - Compare the k-th step prediction (= month T) against actual flowrate[T]

This lets you measure accuracy as it would have been if you had made the
forecast k months before the evaluation month (e.g. k=3 → Sep forecast
evaluated against Dec actual).
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

warnings.filterwarnings("ignore")

WEATHER_VARS = ["precipitation", "et0"]
EPSILON = 1e-6
MIN_TRAIN_ROWS = 12
MIN_TRAIN_SARIMAX = 24


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_best_lag(target, feature, max_lag=6):
    correlations = []
    for lag in range(1, max_lag + 1):
        shifted = pd.Series(feature).shift(lag)
        valid = pd.DataFrame({"t": target, "f": shifted}).dropna()
        if len(valid) > 5:
            corr = valid["t"].corr(valid["f"])
            correlations.append({"Lag": lag, "Correlation": corr})
    if not correlations:
        return 1
    df_c = pd.DataFrame(correlations)
    return int(df_c.loc[df_c["Correlation"].abs().idxmax(), "Lag"])


def _build_ml_features(df, best_lags, target_col="flowrate"):
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
    return features


def _build_pred_row(df, best_lags, target_month_weather, next_month_num, feat_cols, target_col="flowrate"):
    new_feat = pd.DataFrame(index=[0])
    for var in WEATHER_VARS:
        current_val = target_month_weather.get(var, float(df[var].iloc[-1]))
        new_feat[var] = current_val
        for i in range(1, 7):
            vals = df[var].values
            new_feat[f"{var}_lag{i}"] = vals[-i] if i <= len(vals) else current_val
    for i in range(1, 7):
        vals = df[target_col].values
        new_feat[f"target_lag{i}"] = vals[-i] if i <= len(vals) else float(df[target_col].iloc[-1])
    for var in WEATHER_VARS:
        bl = best_lags.get(var, 0)
        if bl > 0:
            vals = df[var].values
            new_feat[f"{var}_best_lag{bl}"] = (
                vals[-bl] if bl <= len(vals)
                else target_month_weather.get(var, float(df[var].iloc[-1]))
            )
    new_feat["month"] = next_month_num
    new_feat["month_sin"] = np.sin(2 * np.pi * next_month_num / 12)
    new_feat["month_cos"] = np.cos(2 * np.pi * next_month_num / 12)
    return new_feat[feat_cols]


# ── k-step ahead predictors ───────────────────────────────────────────────────

def _k_step_xgboost(train_df, future_weather_list, target_col="flowrate"):
    """
    Iteratively forecast k steps ahead for XGBoost.
    future_weather_list: list of dicts, one per step, keys = WEATHER_VARS.
                         len = k (number of steps to forecast).
    Returns the final step's prediction (float) or None on failure.
    """
    df = train_df.copy().reset_index(drop=True)
    if len(df) < MIN_TRAIN_ROWS:
        return None

    best_lags = {var: _find_best_lag(df[target_col].values, df[var].values) for var in WEATHER_VARS}
    features = _build_ml_features(df, best_lags, target_col)
    target = df[target_col]

    valid_idx = features.dropna().index
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    if len(features) < 8:
        return None

    scaler = StandardScaler()
    feat_sc = scaler.fit_transform(features)

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    model.fit(feat_sc, target)

    # Iteratively forecast k steps
    working_df = train_df.copy().reset_index(drop=True)
    last_date = pd.Timestamp(working_df["date"].iloc[-1])
    yhat = None

    for step_weather in future_weather_list:
        next_date = last_date + pd.DateOffset(months=1)
        next_month_num = next_date.month

        pred_row = _build_pred_row(
            working_df, best_lags, step_weather, next_month_num, features.columns, target_col
        )
        pred_sc = scaler.transform(pred_row)
        yhat = float(model.predict(pred_sc)[0])

        # Append predicted row to working_df for next iteration
        new_row = {col: working_df[col].iloc[-1] for col in working_df.columns}
        new_row["date"] = next_date
        new_row[target_col] = yhat
        for var in WEATHER_VARS:
            new_row[var] = step_weather.get(var, float(working_df[var].iloc[-1]))
        working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return yhat


def _k_step_ml(train_df, future_weather_list, target_col="flowrate"):
    """
    Iteratively forecast k steps ahead for the RF+GBR ensemble.
    """
    df = train_df.copy().reset_index(drop=True)
    if len(df) < MIN_TRAIN_ROWS:
        return None

    best_lags = {var: _find_best_lag(df[target_col].values, df[var].values) for var in WEATHER_VARS}
    features = _build_ml_features(df, best_lags, target_col)
    target = df[target_col]

    valid_idx = features.dropna().index
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    if len(features) < 8:
        return None

    scaler = StandardScaler()
    feat_sc = scaler.fit_transform(features)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.05)
    rf.fit(feat_sc, target)
    gb.fit(feat_sc, target)

    working_df = train_df.copy().reset_index(drop=True)
    last_date = pd.Timestamp(working_df["date"].iloc[-1])
    yhat = None

    for step_weather in future_weather_list:
        next_date = last_date + pd.DateOffset(months=1)
        next_month_num = next_date.month

        pred_row = _build_pred_row(
            working_df, best_lags, step_weather, next_month_num, features.columns, target_col
        )
        pred_sc = scaler.transform(pred_row)
        yhat = float((rf.predict(pred_sc)[0] + gb.predict(pred_sc)[0]) / 2)

        new_row = {col: working_df[col].iloc[-1] for col in working_df.columns}
        new_row["date"] = next_date
        new_row[target_col] = yhat
        for var in WEATHER_VARS:
            new_row[var] = step_weather.get(var, float(working_df[var].iloc[-1]))
        working_df = pd.concat([working_df, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return yhat


def _k_step_sarimax(train_df, future_weather_list, target_col="flowrate"):
    """
    Forecast k steps ahead for SARIMAX.
    future_weather_list: list of dicts, len = k.
    """
    df = train_df.copy().reset_index(drop=True)
    if len(df) < MIN_TRAIN_SARIMAX:
        return None

    k = len(future_weather_list)
    best_lags = {var: _find_best_lag(df[target_col].values, df[var].values) for var in WEATHER_VARS}
    max_lag_val = max(best_lags.values()) if best_lags else 0

    exog_df = df[WEATHER_VARS].copy()
    exog_cols = list(WEATHER_VARS)
    for var in WEATHER_VARS:
        bl = best_lags[var]
        for lag in range(1, bl + 1):
            col_name = f"{var}_lag{lag}"
            exog_df[col_name] = df[var].shift(lag)
            exog_cols.append(col_name)

    valid_start = max_lag_val if max_lag_val > 0 else 0
    df_valid = df.iloc[valid_start:].reset_index(drop=True)
    exog_valid = exog_df.iloc[valid_start:].reset_index(drop=True)[exog_cols]
    exog_arr = exog_valid.ffill().bfill().values

    train_series = df_valid[target_col]

    order = (1, 1, 1)
    seasonal = (1, 0, 1, 12)

    try:
        model = SARIMAX(
            train_series, exog=exog_arr, order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=100)
    except Exception:
        return None

    # Build future exog for all k steps
    future_exog_rows = []
    for step_idx, step_weather in enumerate(future_weather_list):
        row = [step_weather.get(var, float(df[var].iloc[-1])) for var in WEATHER_VARS]
        for var in WEATHER_VARS:
            bl = best_lags[var]
            for lag in range(1, bl + 1):
                # For step_idx=0 (first future month), lag-1 = last training value
                # For step_idx>0, use previous future weather as the lagged value
                actual_lag_idx = step_idx - lag
                if actual_lag_idx < 0:
                    # Look back into training data
                    vals = df[var].values
                    train_idx = len(vals) + actual_lag_idx
                    row.append(vals[train_idx] if 0 <= train_idx < len(vals) else float(df[var].iloc[-1]))
                else:
                    row.append(future_weather_list[actual_lag_idx].get(var, float(df[var].iloc[-1])))
        future_exog_rows.append(row)

    try:
        forecasts = fitted.forecast(steps=k, exog=np.array(future_exog_rows))
        return float(forecasts.iloc[-1])
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def run_rolling_backtest(df, n_months=3, forecast_lag=1, models=None, target_col="flowrate"):
    """
    Walk-forward backtest over the last `n_months` months with actual data,
    using a `forecast_lag`-step ahead approach.

    For each evaluation month T:
      - Training cutoff : T - forecast_lag months
      - Forecast        : iterative k-step ahead using actual weather[T-k+1..T]
      - Target          : actual flowrate[T]

    forecast_lag=1 → classic 1-step ahead (train on T-1, predict T)
    forecast_lag=3 → train on T-3, predict 3 steps to reach T

    Returns a DataFrame with columns:
      target_month, model_name, yhat, actual, abs_error, ape, accuracy_pct
    """
    if models is None:
        models = ["XGBoost", "ML", "SARIMAX"]

    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    all_dates = sorted(df["date"].unique())

    # We need at least forecast_lag months before each eval date
    eval_dates = all_dates[-n_months:]

    rows = []
    for eval_date in eval_dates:
        eval_ts = pd.Timestamp(eval_date)
        target_month_str = eval_ts.strftime("%Y-%m")

        # Cutoff = forecast_lag months before eval_date
        cutoff_ts = eval_ts - pd.DateOffset(months=forecast_lag)

        # Training data: strictly up to cutoff (inclusive)
        train_df = df[df["date"] <= cutoff_ts].copy()
        if len(train_df) < MIN_TRAIN_ROWS:
            continue

        # Actual value for eval month
        actual_rows = df.loc[df["date"] == eval_ts, target_col].values
        if len(actual_rows) == 0:
            continue
        actual = float(actual_rows[0])

        # Build future_weather_list: actual weather for the k intermediate months
        # These are months: cutoff+1, cutoff+2, ..., eval_date (k months total)
        future_dates = pd.date_range(
            start=cutoff_ts + pd.DateOffset(months=1),
            end=eval_ts,
            freq="MS"
        )
        future_weather_list = []
        for fdate in future_dates:
            wrow = df[df["date"] == fdate][WEATHER_VARS]
            if not wrow.empty:
                future_weather_list.append(wrow.iloc[0].to_dict())
            else:
                # fallback to last known weather
                future_weather_list.append({var: float(train_df[var].iloc[-1]) for var in WEATHER_VARS})

        if len(future_weather_list) == 0:
            continue

        for model_name in models:
            try:
                if model_name == "XGBoost":
                    yhat = _k_step_xgboost(train_df, future_weather_list, target_col)
                elif model_name == "ML":
                    yhat = _k_step_ml(train_df, future_weather_list, target_col)
                elif model_name == "SARIMAX":
                    yhat = _k_step_sarimax(train_df, future_weather_list, target_col)
                else:
                    continue

                if yhat is None:
                    continue

                abs_error = abs(actual - yhat)
                eps = max(abs(actual), EPSILON)
                accuracy = max(0.0, 100.0 * (1.0 - abs_error / eps))

                rows.append({
                    "target_month": target_month_str,
                    "model_name": model_name,
                    "yhat": round(yhat, 4),
                    "actual": round(actual, 4),
                    "abs_error": round(abs_error, 4),
                    "ape": round(abs_error / eps * 100, 2),
                    "accuracy_pct": round(accuracy, 2),
                })
            except Exception:
                continue

    return pd.DataFrame(rows)
