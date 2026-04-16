import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TEMPLATE = "plotly_dark"


@st.cache_resource
def load_artifacts():
    return joblib.load("models/artifacts.pkl")


def render_overview(series, metrics):
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "Atmospheric CO2")
    c2.metric("Frequency", "Monthly")
    c3.metric("History Length", f"{len(series)} months")
    best_model = min(metrics.keys(), key=lambda k: metrics[k]["RMSE"])
    c4.metric("Best RMSE", f"{best_model} ({metrics[best_model]['RMSE']})")

    fig = px.line(
        x=series.index,
        y=series.values,
        labels={"x": "Date", "y": "CO2"},
        title="Historical CO2 Time Series",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(metrics):
    st.subheader("Walk-Forward Backtesting (12 months)")
    df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "sMAPE": "{:.3f}%"}), use_container_width=True, hide_index=True)

    melted = df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(
        melted,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Comparison Metrics",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_backtest_chart(test_index, y_true, preds):
    st.subheader("Backtest: Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=y_true, mode="lines+markers", name="Actual", line=dict(width=3)))
    for name, p in preds.items():
        fig.add_trace(go.Scatter(x=test_index, y=p, mode="lines+markers", name=name))
    fig.update_layout(
        title="12-Month Holdout Predictions",
        xaxis_title="Date",
        yaxis_title="CO2",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecast(series, future_index, fc_naive, fc_hw, fc_xgb):
    st.subheader("24-Month Forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name="Historical",
        line=dict(width=3, color="#90caf9"),
    ))

    fig.add_trace(go.Scatter(x=future_index, y=fc_naive, mode="lines", name="Seasonal Naive"))
    fig.add_trace(go.Scatter(x=future_index, y=fc_hw, mode="lines", name="Holt-Winters"))
    fig.add_trace(go.Scatter(x=future_index, y=fc_xgb, mode="lines", name="XGBoost"))

    fig.update_layout(
        title="Future Forecasts by Model",
        xaxis_title="Date",
        yaxis_title="CO2",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_xgb_importance(importance_df):
    st.subheader("XGBoost Feature Importance")
    top = importance_df.head(15)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        template=TEMPLATE,
        title="Top Lag/Calendar Features",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)



def main():
    st.title("📈 Time Series Forecasting")
    st.markdown(
        "**Pipeline:** monthly CO2 forecasting with three model families: "
        "Seasonal Naive, Holt-Winters, and XGBoost with lag/calendar features. "
        "Includes walk-forward backtesting and multi-horizon forecast visualization."
    )

    arts = load_artifacts()

    tabs = st.tabs([
        "Overview",
        "Model Comparison",
        "Backtest",
        "Forecast",
        "Feature Importance",
    ])

    with tabs[0]:
        render_overview(arts["series"], arts["metrics"])
    with tabs[1]:
        render_model_comparison(arts["metrics"])
    with tabs[2]:
        render_backtest_chart(arts["test_index"], arts["y_true"], arts["preds"])
    with tabs[3]:
        render_forecast(
            arts["series"],
            arts["future_index"],
            arts["forecast_naive"],
            arts["forecast_hw"],
            arts["forecast_xgb"],
        )
    with tabs[4]:
        render_xgb_importance(arts["xgb_feature_importance"])


if __name__ == "__main__":
    main()
