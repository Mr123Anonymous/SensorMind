"""Model comparison dashboard page."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is importable for `app` and `src` module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import load_error_analysis, load_phase3_summary

st.set_page_config(page_title="Model Comparison", page_icon=":bar_chart:", layout="wide")
st.title("Model Comparison Dashboard")
st.caption("Cross-task metrics for forecasting, anomaly detection, and NLP.")

summary = load_phase3_summary()
if not summary:
	st.error("Missing phase3_summary.json. Run scripts/run_phase3.py first.")
	st.stop()

forecast_df = pd.DataFrame(summary["forecast"]["metrics"]).T.reset_index().rename(columns={"index": "model"})
anomaly_df = pd.DataFrame(summary["anomaly"]["metrics"]).T.reset_index().rename(columns={"index": "model"})
nlp_df = pd.DataFrame(summary["nlp"]["metrics"]).T.reset_index().rename(columns={"index": "model"})

st.subheader("Forecasting Models")
st.dataframe(forecast_df, use_container_width=True)
fig_f = px.bar(forecast_df, x="model", y=["RMSE", "MAE", "MAPE"], barmode="group", title="Forecast Metrics")
st.plotly_chart(fig_f, use_container_width=True)

st.subheader("Anomaly Models")
st.dataframe(anomaly_df, use_container_width=True)
fig_a = px.bar(anomaly_df, x="model", y=["F1", "Precision", "Recall", "Accuracy"], barmode="group", title="Anomaly Metrics")
st.plotly_chart(fig_a, use_container_width=True)

st.subheader("NLP Models")
st.dataframe(nlp_df, use_container_width=True)
if {"Accuracy", "F1", "Precision", "Recall"}.issubset(set(nlp_df.columns)):
	fig_n = px.bar(nlp_df, x="model", y=["Accuracy", "F1", "Precision", "Recall"], barmode="group", title="NLP Metrics")
	st.plotly_chart(fig_n, use_container_width=True)

analysis = load_error_analysis()
if analysis:
	st.subheader("Error Analysis Highlights")
	h1, h2, h3 = st.columns(3)
	h1.metric("Forecast P90 Abs Error", f"{analysis['forecast']['analysis']['p90_abs_error']:.4f}")
	h2.metric("Anomaly False Positives", int(analysis['anomaly']['analysis']['false_positives']))
	h3.metric("Anomaly False Negatives", int(analysis['anomaly']['analysis']['false_negatives']))
