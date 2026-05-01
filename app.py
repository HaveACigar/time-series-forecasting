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

def build_takeaways(metrics: dict, meta: dict) -> dict:
    results_df = (
        pd.DataFrame(metrics)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("RMSE", ascending=True)
    )
    best = results_df.iloc[0]
    worst = results_df.iloc[-1]
    rmse_gap = float(worst["RMSE"] - best["RMSE"]) if len(results_df) > 1 else 0.0
    horizon = int(meta.get("horizon", 0) or 0)
    dataset_name = meta.get("dataset_name", "the selected dataset")

    return {
        "overview": f"Using {dataset_name}, this baseline defines the demand trajectory leadership should anchor planning assumptions on.",
        "model_comparison": f"Best model is {best['Model']} at RMSE {best['RMSE']:.3f}; model choice changes error by {rmse_gap:.3f} RMSE versus the weakest option.",
        "backtest": f"Holdout diagnostics across {horizon} periods highlight where misses are likely so finance and operations can prepare contingency actions.",
        "forecast": "Forward trajectories expose planning range and uncertainty bands, helping teams set budget and capacity with clearer risk awareness.",
        "importance": "Temporal feature ranking identifies which lag and seasonality signals drive projections, guiding where additional data will improve forecast reliability.",
    }


def render_takeaway(key: str, takeaways: dict) -> None:
    st.info(f"Shareholder Takeaway: {takeaways[key]}")


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
    takeaways = build_takeaways(arts["metrics"], meta)
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
        render_takeaway("overview", takeaways)
        render_overview(arts["series"], arts["metrics"], meta)
    with tabs[1]:
        render_takeaway("model_comparison", takeaways)
        render_model_comparison(arts["metrics"], meta)
    with tabs[2]:
        render_takeaway("backtest", takeaways)
        render_backtest_chart(arts["test_index"], arts["y_true"], arts["preds"])
    with tabs[3]:
        render_takeaway("forecast", takeaways)
        render_forecast(
            arts["series"],
            arts["future_index"],
            arts["forecast_naive"],
            arts["forecast_hw"],
            arts["forecast_xgb"],
        )
    with tabs[4]:
        render_takeaway("importance", takeaways)
        render_xgb_importance(arts["xgb_feature_importance"])


if __name__ == "__main__":
    main()
