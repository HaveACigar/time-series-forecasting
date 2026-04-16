"""
Time Series Forecasting Project
Dataset: Atmospheric CO2 concentration (statsmodels built-in dataset)

Builds and compares three forecasting approaches with walk-forward validation:
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


def load_series() -> pd.Series:
    data = co2.load_pandas().data.copy()
    s = data["co2"].asfreq("W-SAT")
    s = s.interpolate().resample("MS").mean().dropna()
    s.name = "co2"
    return s


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


def walk_forward_evaluate(series: pd.Series, horizon: int = 12):
    split_idx = len(series) - horizon
    train_init = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    preds_naive = []
    preds_hw = []
    preds_xgb = []

    y_true = []

    for i in range(horizon):
        train = series.iloc[:split_idx + i]
        actual = series.iloc[split_idx + i]

        # Seasonal Naive
        p_naive = forecast_seasonal_naive(train, steps=1, season=12)[0]

        # Holt-Winters
        hw = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        p_hw = hw.forecast(1).iloc[0]

        # XGBoost with lag features
        feat_df = make_features(train)
        X_train = feat_df.drop(columns=["y"])
        y_train = feat_df["y"]
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        xgb.fit(X_train, y_train)

        # One-step recursive feature row
        pred_idx = train.index[-1] + pd.offsets.MonthBegin(1)
        row = {"month": pred_idx.month, "quarter": pred_idx.quarter, "trend": len(train)}
        for lag in (1, 2, 3, 6, 12):
            row[f"lag_{lag}"] = train.iloc[-lag]
        X_one = pd.DataFrame([row])[X_train.columns]
        p_xgb = float(xgb.predict(X_one)[0])

        preds_naive.append(p_naive)
        preds_hw.append(p_hw)
        preds_xgb.append(p_xgb)
        y_true.append(actual)

    y_true = np.array(y_true)
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


def fit_final_models_and_forecast(series: pd.Series, steps: int = 24):
    # Seasonal Naive forecast
    fc_naive = forecast_seasonal_naive(series, steps=steps, season=12)

    # Holt-Winters
    hw = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)
    fc_hw = hw.forecast(steps).values

    # XGBoost recursive multi-step
    feat_df = make_features(series)
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

    history = series.copy()
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

    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")

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
    series = load_series()
    print(f"Length: {len(series)} months")

    print("Running walk-forward evaluation")
    test_idx, y_true, preds, metrics = walk_forward_evaluate(series, horizon=12)
    print(metrics)

    print("Fitting final models and generating 24-month forecast")
    final = fit_final_models_and_forecast(series, steps=24)

    os.makedirs("models", exist_ok=True)
    artifacts = {
        "series": series,
        "test_index": test_idx,
        "y_true": y_true,
        "preds": preds,
        "metrics": metrics,
        **final,
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    print("Saved models/artifacts.pkl")


if __name__ == "__main__":
    main()
