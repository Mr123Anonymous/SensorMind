"""Anomaly detection dashboard page."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is importable for `app` and `src` module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import load_anomaly_model, load_error_analysis, load_phase3_summary, load_split
from src.models.predict import predict_anomaly

st.set_page_config(page_title="Anomaly Detection", page_icon=":rotating_light:", layout="wide")
st.title("Anomaly Detection Dashboard")
st.caption("Interactive anomaly scoring and threshold analysis.")

summary = load_phase3_summary()
if not summary:
	st.error("Missing phase3_summary.json. Run scripts/run_phase3.py first.")
	st.stop()

anomaly_meta = summary.get("anomaly", {})
available_models = sorted(list((anomaly_meta.get("metrics") or {}).keys()))
if not available_models:
	st.error("No anomaly model metadata found.")
	st.stop()

best_model = anomaly_meta.get("best_model", available_models[0])
selected_model = st.sidebar.selectbox("Model", available_models, index=available_models.index(best_model))

_, _, X_test, _, _, y_test = load_split("anomaly")
model, default_threshold = load_anomaly_model(selected_model)
threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=5.0, value=float(default_threshold), step=0.0005)

output = predict_anomaly(model, X_test, threshold=threshold)
y_pred = output["is_anomaly"].astype(int)
scores = output["anomaly_score"]

if len(scores) == 0:
	st.error("No anomaly scores were produced for the selected model.")
	st.stop()

fp = int(np.sum((y_test == 0) & (y_pred == 1)))
fn = int(np.sum((y_test == 1) & (y_pred == 0)))
tp = int(np.sum((y_test == 1) & (y_pred == 1)))
tn = int(np.sum((y_test == 0) & (y_pred == 0)))

col1, col2, col3, col4 = st.columns(4)
metrics = anomaly_meta["metrics"][selected_model]
col1.metric("Model", selected_model)
col2.metric("F1", f"{metrics['F1']:.4f}")
col3.metric("Precision", f"{metrics['Precision']:.4f}")
col4.metric("Recall", f"{metrics['Recall']:.4f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("TP", tp)
col6.metric("FP", fp)
col7.metric("FN", fn)
col8.metric("TN", tn)

max_points = min(2000, len(scores))
min_points = 1 if len(scores) < 100 else 100
default_points = min(600, max_points)

if min_points >= max_points:
	sample_size = max_points
	st.sidebar.caption(f"Points to display: {sample_size}")
else:
	sample_size = st.sidebar.slider(
		"Points to display",
		min_value=min_points,
		max_value=max_points,
		value=default_points,
	)

max_start = max(0, len(scores) - sample_size)
if max_start == 0:
	start_idx = 0
	st.sidebar.caption("Start index: 0")
else:
	start_idx = st.sidebar.slider("Start index", min_value=0, max_value=max_start, value=0)
end_idx = start_idx + sample_size

indices = np.arange(start_idx, end_idx)
scores_view = scores[start_idx:end_idx]
y_view = y_test[start_idx:end_idx]
pred_view = y_pred[start_idx:end_idx]

fig = go.Figure()
fig.add_trace(go.Scatter(x=indices, y=scores_view, mode="lines", name="Anomaly Score"))
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")

anom_idx = indices[pred_view == 1]
anom_scores = scores_view[pred_view == 1]
fig.add_trace(go.Scatter(x=anom_idx, y=anom_scores, mode="markers", name="Predicted Anomaly", marker=dict(color="red", size=7)))

fig.update_layout(title="Anomaly Scores", xaxis_title="Index", yaxis_title="Score", height=450)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Sample")
view_df = pd.DataFrame({
	"index": indices,
	"score": scores_view,
	"actual": y_view,
	"predicted": pred_view,
})
st.dataframe(view_df.head(40), use_container_width=True)

analysis = load_error_analysis()
if analysis.get("anomaly"):
	st.subheader("Phase 4 Error Analysis Snapshot")
	st.json(analysis["anomaly"]["analysis"])
