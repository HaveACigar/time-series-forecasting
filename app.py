import joblib
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

TAB_TAKEAWAYS = {
    "overview": "The time-series baseline highlights demand volatility and trend direction, shaping planning assumptions before capital is committed.",
    "model_comparison": "Model error comparison quantifies forecast risk so leadership can choose planning buffers with confidence.",
    "backtest": "Holdout performance reveals where forecast misses are likely, helping finance and operations pre-position mitigation actions.",
    "forecast": "Scenario trajectories provide a forward-looking view of demand range, supporting budget, staffing, and inventory decisions.",
    "importance": "Feature importance shows which temporal drivers matter most, informing where additional data collection improves decision quality.",
}


def render_takeaway(key: str) -> None:
    st.info(f"Shareholder Takeaway: {TAB_TAKEAWAYS[key]}")


@st.cache_resource
def load_artifacts():
    return joblib.load("models/artifacts.pkl")


def render_overview(series, metrics, meta):
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", meta.get("dataset_name", "Unknown"))
    c2.metric("Series", meta.get("series_id", "N/A"))
    c3.metric("History Length", f"{len(series)} months")
    best_model = min(metrics.keys(), key=lambda k: metrics[k]["RMSE"])
    c4.metric("Best RMSE", f"{best_model} ({metrics[best_model]['RMSE']})")

    fig = px.line(
        x=series.index,
        y=series.values,
        labels={"x": "Date", "y": "Value"},
        title="Historical Time Series",
        template=TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison(metrics, meta):
    st.subheader("Holdout Evaluation")
    st.caption(f"Holdout horizon: {meta.get('horizon', '-') } months")
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
    st.subheader("Holdout: Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=y_true, mode="lines+markers", name="Actual", line=dict(width=3)))
    for name, p in preds.items():
        fig.add_trace(go.Scatter(x=test_index, y=p, mode="lines+markers", name=name))
    fig.update_layout(
        title="Holdout Predictions",
        xaxis_title="Date",
        yaxis_title="Value",
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
        yaxis_title="Value",
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

    arts = load_artifacts()
    meta = arts.get("dataset_meta", {})
    st.markdown(
        f"Dataset source: {meta.get('dataset_name', 'Unknown')} | "
        f"Series: {meta.get('series_id', 'N/A')} | "
        f"Mode: {meta.get('source', 'unknown')}"
    )

    tabs = st.tabs([
        "Overview",
        "Model Comparison",
        "Backtest",
        "Forecast",
        "Feature Importance",
    ])

    with tabs[0]:
        render_takeaway("overview")
        render_overview(arts["series"], arts["metrics"], meta)
    with tabs[1]:
        render_takeaway("model_comparison")
        render_model_comparison(arts["metrics"], meta)
    with tabs[2]:
        render_takeaway("backtest")
        render_backtest_chart(arts["test_index"], arts["y_true"], arts["preds"])
    with tabs[3]:
        render_takeaway("forecast")
        render_forecast(
            arts["series"],
            arts["future_index"],
            arts["forecast_naive"],
            arts["forecast_hw"],
            arts["forecast_xgb"],
        )
    with tabs[4]:
        render_takeaway("importance")
        render_xgb_importance(arts["xgb_feature_importance"])


if __name__ == "__main__":
    main()
