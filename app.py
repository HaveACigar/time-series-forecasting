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


def render_summary_cards(cards: list[dict]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div style="
                    padding: 0.2rem 0.1rem 0.75rem 0.1rem;
                    min-height: 120px;
                ">
                  <div style="
                      font-size: 0.95rem;
                      color: rgba(250,250,250,0.82);
                      margin-bottom: 0.5rem;
                  ">{card['label']}</div>
                  <div style="
                      font-size: 2.0rem;
                      font-weight: 700;
                      line-height: 1.08;
                      word-break: break-word;
                      overflow-wrap: anywhere;
                  ">{card['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_lag_explainer(meta: dict) -> None:
    dataset_name = meta.get("dataset_name", "this dataset")
    with st.expander("What lag features mean in this forecast", expanded=False):
        st.markdown(
            f"""
For {dataset_name}, a lag feature means "the value from an earlier month" used as an input for the next prediction.

- `lag_1`: the previous month's value
- `lag_2`: the value from two months ago
- `lag_3`: the value from three months ago
- `lag_6`: the value from six months ago
- `lag_12`: the value from the same month one year earlier

In this app, lags help XGBoost learn short-term momentum and yearly seasonality. For a monthly series like M4 Monthly, `lag_12` is especially useful because it gives the model the last observation from the same season one year ago.
            """
        )

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
    best_model = min(metrics.keys(), key=lambda k: metrics[k]["RMSE"])
    render_summary_cards([
        {"label": "Dataset", "value": meta.get("dataset_name", "Unknown")},
        {"label": "Series", "value": meta.get("series_id", "N/A")},
        {"label": "History Length", "value": f"{len(series)} months"},
        {"label": "Best RMSE", "value": f"{best_model} ({metrics[best_model]['RMSE']:.3f})"},
    ])

    render_lag_explainer(meta)

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
    st.caption("Lag features use prior monthly values to help the model recognize momentum and recurring yearly patterns.")
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
