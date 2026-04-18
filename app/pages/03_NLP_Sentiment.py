"""NLP sentiment page."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is importable for `app` and `src` module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import load_error_analysis, load_nlp_model, load_phase3_summary
from src.models.predict import predict_sentiment

st.set_page_config(page_title="NLP Sentiment", page_icon=":speech_balloon:", layout="wide")
st.title("NLP Sentiment Inference")
st.caption("TF-IDF + Logistic Regression baseline with probability outputs.")

summary = load_phase3_summary()
if not summary:
	st.error("Missing phase3_summary.json. Run scripts/run_phase3.py first.")
	st.stop()

nlp_meta = summary.get("nlp", {})
model_name = nlp_meta.get("best_model", "tfidf_logistic_regression")
model = load_nlp_model()

col1, col2, col3 = st.columns(3)
metrics = nlp_meta.get("metrics", {}).get(model_name, {})
col1.metric("Model", model_name)
col2.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
col3.metric("F1", f"{metrics.get('F1', 0):.4f}")

st.subheader("Single/Batched Text Inference")
default_text = "great output from the model\npoor deployment experience"
user_text = st.text_area("Enter one text per line", value=default_text, height=150)
lines = [line.strip() for line in user_text.splitlines() if line.strip()]

if st.button("Run Sentiment Prediction", type="primary"):
	if not lines:
		st.warning("Please enter at least one text line.")
	else:
		result = predict_sentiment(model, lines)
		probs = result.get("probabilities")
		out_df = pd.DataFrame({
			"text": lines,
			"prediction": result["predictions"],
		})
		if probs is not None:
			out_df["prob_negative"] = probs[:, 0]
			out_df["prob_positive"] = probs[:, 1]
		st.dataframe(out_df, use_container_width=True)

analysis = load_error_analysis()
if analysis.get("nlp"):
	st.subheader("Phase 4 NLP Risk Note")
	st.info(analysis["nlp"]["analysis"]["note"])
	st.warning(analysis["nlp"]["analysis"]["risk"])
