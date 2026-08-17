import pandas as pd
import numpy as np

EPSILON = 1e-6


def compute_accuracy_pct(actual, yhat, epsilon=EPSILON):
    abs_error = abs(actual - yhat)
    ape = abs_error / max(abs(actual), epsilon)
    return max(0.0, 100.0 * (1.0 - ape))


def select_latest_forecasts(history_df):
    """
    Take the latest forecast run per (target_month, model_name, output_type).
    No one-step-ahead constraint — any saved forecast for a month counts.
    """
    if history_df.empty:
        return pd.DataFrame()
    df = history_df.copy()
    df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], errors="coerce")
    df = df.sort_values("run_timestamp", ascending=False)
    df = df.drop_duplicates(
        subset=["site_name", "target_month", "model_name", "output_type"], keep="first"
    )
    return df.reset_index(drop=True)


def build_accuracy_table(one_step_df, actual_series, include_ci=True):
    """
    LEFT JOIN forecast → actual.
    Rows with actual → status='Completed', metrics computed.
    Rows without actual → status='Waiting actual', metrics are NaN.
    """
    if one_step_df.empty:
        return pd.DataFrame()

    output_types = ["forecast_flowrate"]
    if include_ci:
        output_types += ["lower_95", "upper_95"]

    df = one_step_df[one_step_df["output_type"].isin(output_types)].copy()
    if df.empty:
        return pd.DataFrame()

    actual_df = actual_series.rename("actual").reset_index()
    actual_df.columns = ["date_str", "actual"]

    # LEFT JOIN: keep all forecasts, attach actual where available
    merged = df.merge(actual_df, left_on="target_month", right_on="date_str", how="left")

    rows = []
    for _, row in merged.iterrows():
        yhat = float(row["yhat"])
        has_actual = pd.notna(row.get("actual"))

        base = {
            "target_month": row["target_month"],
            "model_name": row["model_name"],
            "output_type": row["output_type"],
            "yhat": round(yhat, 4),
        }

        if has_actual:
            actual = float(row["actual"])
            abs_error = abs(actual - yhat)
            ape = abs_error / max(abs(actual), EPSILON)
            acc_pct = compute_accuracy_pct(actual, yhat)
            base.update({
                "actual": round(actual, 4),
                "abs_error": round(abs_error, 4),
                "ape": round(ape * 100, 2),
                "accuracy_pct": round(acc_pct, 2),
                "status": "Completed",
            })
        else:
            base.update({
                "actual": None,
                "abs_error": None,
                "ape": None,
                "accuracy_pct": None,
                "status": "Waiting actual",
            })

        rows.append(base)

    return pd.DataFrame(rows)


_WINNER_COLS = ["target_month", "model_name", "output_type", "yhat", "actual",
                "abs_error", "ape", "accuracy_pct", "status"]


def _empty_winners():
    return pd.DataFrame(columns=_WINNER_COLS)


def pick_winner_per_month(accuracy_df):
    """Only pick winners for Completed months."""
    if accuracy_df.empty:
        return _empty_winners()
    completed = accuracy_df[accuracy_df["status"] == "Completed"].copy()
    if completed.empty:
        return _empty_winners()
    idx = completed.groupby("target_month")["accuracy_pct"].idxmax()
    winners = completed.loc[idx].copy().reset_index(drop=True)
    # Ensure target_month column always present
    if "target_month" not in winners.columns:
        return _empty_winners()
    return winners.sort_values("target_month").reset_index(drop=True)


def get_recommendation(accuracy_df):
    """Based only on Completed months."""
    if accuracy_df.empty:
        return None, None
    completed = accuracy_df[accuracy_df["status"] == "Completed"]
    if completed.empty:
        return None, None
    agg = (
        completed.groupby(["model_name", "output_type"])["accuracy_pct"]
        .mean()
        .reset_index()
    )
    best = agg.loc[agg["accuracy_pct"].idxmax()]
    return best["model_name"], best["output_type"]


def window_summary_metrics(accuracy_df):
    """Based only on Completed months."""
    if accuracy_df.empty:
        return {}
    completed = accuracy_df[accuracy_df["status"] == "Completed"]
    if completed.empty:
        return {}
    return {
        "MAE": round(completed["abs_error"].mean(), 4),
        "MAPE": round(completed["ape"].mean(), 2),
        "Mean Accuracy": round(completed["accuracy_pct"].mean(), 2),
    }
