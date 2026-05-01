"""
Time Series Forecasting
Primary dataset: M4 Monthly competition series (open GitHub mirror).
Fallback dataset: atmospheric CO2 (statsmodels) if network/download is unavailable.

Builds and compares:
1) Seasonal Naive
2) Holt-Winters Exponential Smoothing
3) XGBoost with lag/calendar features

Artifacts are serialized for instant Streamlit startup.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from statsmodels.datasets import co2
from statsmodels.tsa.holtwinters import ExponentialSmoothing


M4_TRAIN_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv"
M4_TEST_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv"


def load_co2_series() -> tuple[pd.Series, pd.Series, dict]:
    data = co2.load_pandas().data.copy()
    s = data["co2"].asfreq("W-SAT")
    s = s.interpolate().resample("MS").mean().dropna()
    s.name = "value"
    horizon = 12
    return s.iloc[:-horizon], s.iloc[-horizon:], {
        "dataset_name": "Atmospheric CO2 (statsmodels)",
        "dataset_key": "co2_fallback",
        "series_id": "co2",
        "horizon": horizon,
        "source": "fallback",
    }


def _row_to_series(row: pd.Series) -> pd.Series:
    vals = pd.to_numeric(row.iloc[1:], errors="coerce").dropna().values
    return pd.Series(vals, dtype=float)


def load_m4_series() -> tuple[pd.Series, pd.Series, dict]:
    series_id = os.getenv("M4_SERIES_ID", "M1").strip()
    train_df = pd.read_csv(M4_TRAIN_URL)
    test_df = pd.read_csv(M4_TEST_URL)

    if series_id not in set(train_df.iloc[:, 0].astype(str)):
        series_id = str(train_df.iloc[0, 0])

    train_row = train_df[train_df.iloc[:, 0].astype(str) == series_id].iloc[0]
    test_row = test_df[test_df.iloc[:, 0].astype(str) == series_id].iloc[0]

    train_vals = _row_to_series(train_row)
    test_vals = _row_to_series(test_row)

    start = pd.Timestamp("2000-01-01")
    train_idx = pd.date_range(start=start, periods=len(train_vals), freq="MS")
    test_idx = pd.date_range(start=train_idx[-1] + pd.offsets.MonthBegin(1), periods=len(test_vals), freq="MS")

    train_series = pd.Series(train_vals.values, index=train_idx, name="value")
    test_series = pd.Series(test_vals.values, index=test_idx, name="value")

    meta = {
        "dataset_name": "M4 Monthly (competition dataset)",
        "dataset_key": "m4_monthly",
        "series_id": series_id,
        "horizon": int(len(test_series)),
        "source": "m4",
    }
    return train_series, test_series, meta


def load_series() -> tuple[pd.Series, pd.Series, dict]:
    try:
        return load_m4_series()
    except Exception as exc:
        print(f"M4 download unavailable ({exc}); using CO2 fallback")
        return load_co2_series()


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100


def make_features(series: pd.Series, lags=(1, 2, 3, 6, 12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    idx = df.index
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["trend"] = np.arange(len(df))
    df = df.dropna()
    return df


def forecast_seasonal_naive(train: pd.Series, steps: int = 1, season: int = 12):
    if len(train) < season:
        return np.repeat(train.iloc[-1], steps)
    tail = train.iloc[-season:].values
    reps = int(np.ceil(steps / season))
    return np.tile(tail, reps)[:steps]


def evaluate_on_holdout(train: pd.Series, test: pd.Series):
    horizon = len(test)
    preds_naive = []
    preds_hw = []
    preds_xgb = []
    y_true = test.values

    for i in range(horizon):
        hist = pd.concat([train, test.iloc[:i]])

        # Seasonal Naive
        p_naive = forecast_seasonal_naive(hist, steps=1, season=12)[0]

        # Holt-Winters
        hw = ExponentialSmoothing(
            hist,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        p_hw = hw.forecast(1).iloc[0]

        # XGBoost with lag features
        feat_df = make_features(hist)
        X_train = feat_df.drop(columns=["y"])
        y_train = feat_df["y"]
        xgb = XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb.fit(X_train, y_train)

        pred_idx = hist.index[-1] + pd.offsets.MonthBegin(1)
        row = {"month": pred_idx.month, "quarter": pred_idx.quarter, "trend": len(hist)}
        for lag in (1, 2, 3, 6, 12):
            row[f"lag_{lag}"] = hist.iloc[-lag]
        X_one = pd.DataFrame([row])[X_train.columns]
        p_xgb = float(xgb.predict(X_one)[0])

        preds_naive.append(p_naive)
        preds_hw.append(p_hw)
        preds_xgb.append(p_xgb)

    preds = {
        "Seasonal Naive": np.array(preds_naive),
        "Holt-Winters": np.array(preds_hw),
        "XGBoost": np.array(preds_xgb),
    }

    metrics = {}
    for name, p in preds.items():
        metrics[name] = {
            "MAE": round(mean_absolute_error(y_true, p), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_true, p)), 3),
            "sMAPE": round(smape(y_true, p), 3),
        }

    return test.index, y_true, preds, metrics


def fit_final_models_and_forecast(full_series: pd.Series, steps: int = 24):
    # Seasonal Naive forecast
    fc_naive = forecast_seasonal_naive(full_series, steps=steps, season=12)

    # Holt-Winters
    hw = ExponentialSmoothing(
        full_series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    fc_hw = hw.forecast(steps).values

    # XGBoost recursive multi-step
    feat_df = make_features(full_series)
    X_train = feat_df.drop(columns=["y"])
    y_train = feat_df["y"]

    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    xgb.fit(X_train, y_train)

    history = full_series.copy()
    preds_xgb = []
    for _ in range(steps):
        pred_idx = history.index[-1] + pd.offsets.MonthBegin(1)
        row = {"month": pred_idx.month, "quarter": pred_idx.quarter, "trend": len(history)}
        for lag in (1, 2, 3, 6, 12):
            row[f"lag_{lag}"] = history.iloc[-lag]
        X_one = pd.DataFrame([row])[X_train.columns]
        pred = float(xgb.predict(X_one)[0])
        preds_xgb.append(pred)
        history.loc[pred_idx] = pred

    future_idx = pd.date_range(full_series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "future_index": future_idx,
        "forecast_naive": fc_naive,
        "forecast_hw": fc_hw,
        "forecast_xgb": np.array(preds_xgb),
        "xgb_feature_importance": feature_importance,
    }


def main():
    print("Loading series")
    train_series, test_series, meta = load_series()
    full_series = pd.concat([train_series, test_series])
    print(f"Dataset: {meta['dataset_name']} ({meta['series_id']})")
    print(f"Train length: {len(train_series)}  |  Holdout: {len(test_series)}")

    print("Evaluating on holdout")
    test_idx, y_true, preds, metrics = evaluate_on_holdout(train_series, test_series)
    print(metrics)

    print("Fitting final models and generating 24-month forecast")
    final = fit_final_models_and_forecast(full_series, steps=24)

    os.makedirs("models", exist_ok=True)
    artifacts = {
        "series": full_series,
        "train_series": train_series,
        "test_series": test_series,
        "test_index": test_idx,
        "y_true": y_true,
        "preds": preds,
        "metrics": metrics,
        "dataset_meta": meta,
        **final,
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    print("Saved models/artifacts.pkl")


if __name__ == "__main__":
    main()
