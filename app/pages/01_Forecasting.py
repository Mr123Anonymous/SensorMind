"""Forecasting dashboard page."""

from __future__ import annotations

from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
st = importlib.import_module("streamlit")

# Ensure project root is importable for `app` and `src` module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import load_error_analysis, load_forecast_model, load_phase3_summary, load_split
from src.models.predict import predict_forecast

st.set_page_config(page_title="Forecasting", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Forecasting Dashboard")
st.caption("Phase 5 deployment view for time-series model inference and diagnostics.")

summary = load_phase3_summary()
if not summary:
	st.error("Missing phase3_summary.json. Run scripts/run_phase3.py first.")
	st.stop()

forecast_meta = summary.get("forecast", {})
available_models = sorted(list((forecast_meta.get("metrics") or {}).keys()))
if not available_models:
	st.error("No forecasting model metadata found.")
	st.stop()

best_model = forecast_meta.get("best_model", available_models[0])
selected_model = st.sidebar.selectbox("Model", available_models, index=available_models.index(best_model))
window = st.sidebar.slider("Display window size", min_value=50, max_value=600, value=300, step=10)
start_idx = st.sidebar.slider("Window start index", min_value=0, max_value=500, value=0, step=10)

_, _, X_test, _, _, y_test = load_split("forecast")
model = load_forecast_model(selected_model)
pred = predict_forecast(model, X_test)

end_idx = min(start_idx + window, len(y_test))
idx = np.arange(start_idx, end_idx)
actual = y_test[start_idx:end_idx]
pred_view = pred[start_idx:end_idx]

col1, col2, col3, col4 = st.columns(4)
metrics = forecast_meta["metrics"][selected_model]
col1.metric("Model", selected_model)
col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
col3.metric("MAE", f"{metrics['MAE']:.4f}")
col4.metric("MAPE", f"{metrics['MAPE']:.4f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=idx, y=actual, mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=idx, y=pred_view, mode="lines", name="Predicted"))
fig.update_layout(
	title=f"Forecast vs Actual ({selected_model})",
	xaxis_title="Test Index",
	yaxis_title="Target",
	height=450,
)
st.plotly_chart(fig, use_container_width=True)

abs_err = np.abs(y_test - pred)
error_df = pd.DataFrame({
	"index": np.arange(len(abs_err)),
	"actual": y_test,
	"predicted": pred,
	"abs_error": abs_err,
}).sort_values("abs_error", ascending=False)

st.subheader("Top Error Samples")
st.dataframe(error_df.head(15), use_container_width=True)

analysis = load_error_analysis()
if analysis.get("forecast"):
	st.subheader("Phase 4 Error Analysis Snapshot")
	st.json(analysis["forecast"]["analysis"])
