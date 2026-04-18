"""About page for project and deployment info."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is importable for `app` and `src` module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import has_phase5_artifacts, load_phase3_summary

st.set_page_config(page_title="About", page_icon=":information_source:", layout="wide")
st.title("About This Deployment")

st.markdown(
	"""
This Streamlit App is the Phase 5 deployment layer for the SensorMind ML Portfolio.

It provides:
- Forecasting inference dashboard
- Anomaly detection threshold tuning and diagnostics
- NLP sentiment inference with probabilities
- Cross-model comparison and error-analysis insights
"""
)

phase3 = load_phase3_summary()
ready = has_phase5_artifacts()

col1, col2 = st.columns(2)
with col1:
	st.subheader("Deployment Readiness")
	if ready:
		st.success("Artifacts available")
	else:
		st.error("Missing required artifacts")
	st.write("Best forecast model:", phase3.get("forecast", {}).get("best_model", "n/a"))
	st.write("Best anomaly model:", phase3.get("anomaly", {}).get("best_model", "n/a"))
	st.write("Best NLP model:", phase3.get("nlp", {}).get("best_model", "n/a"))

with col2:
	st.subheader("Run Commands")
	st.code(
		"""python scripts/run_phase2.py
python scripts/run_phase3.py
python scripts/run_phase4.py
streamlit run app/Home.py""",
		language="bash",
	)

st.subheader("Architecture")
st.code(
	"""data/raw -> phase2 preprocessing -> data/processed/*.npy
					   |
					   v
				 phase3 training -> models/* + phase3_summary.json
					   |
					   v
			phase4 error analysis -> reports/error_analysis.json
					   |
					   v
			 phase5 Streamlit dashboards (this app)"""
)

