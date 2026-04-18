"""SensorMind Dashboard - Home Page."""

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

# Ensure project root is importable when launched via `streamlit run app/Home.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app._shared import has_phase5_artifacts, load_error_analysis, load_phase3_summary

# Page configuration
st.set_page_config(
    page_title="SensorMind ML Portfolio",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("SensorMind ML Portfolio")
st.markdown("### End-to-End Machine Learning Project: Time-Series, Anomaly Detection & NLP")

# Introduction
st.markdown("""
Welcome to the **SensorMind ML Portfolio** project! 

This comprehensive ML project showcases:
- **Time-Series Forecasting** (PGCB Grid Energy)
- **Anomaly Detection** (Sensor Fault Detection)
- **NLP with Modern AI** (Sentiment Analysis + LLMs)
- **Production Deployment** (Streamlit + MLflow + Docker)

---
""")

# Navigation
st.markdown("## Quick Navigation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Forecasting", use_container_width=True):
        st.switch_page("pages/01_Forecasting.py")

with col2:
    if st.button("Anomalies", use_container_width=True):
        st.switch_page("pages/02_Anomaly_Detection.py")

with col3:
    if st.button("NLP/Sentiment", use_container_width=True):
        st.switch_page("pages/03_NLP_Sentiment.py")

with col4:
    if st.button("Comparison", use_container_width=True):
        st.switch_page("pages/04_Model_Comparison.py")

with col5:
    if st.button("About", use_container_width=True):
        st.switch_page("pages/05_About.py")

st.markdown("## Project Overview")

summary = load_phase3_summary()
analysis = load_error_analysis()
ready = has_phase5_artifacts()

overview_data = {
    "Task": ["Time-Series Forecasting", "Anomaly Detection", "NLP Sentiment"],
    "Dataset": ["PGCB Grid", "Yahoo Time-Series", "Kaggle News"],
    "Models Trained": [
        "Ridge, Random Forest, MLP",
        "Isolation Forest, Autoencoder",
        "TF-IDF + Logistic Regression",
    ],
    "Best Model": [
        summary.get("forecast", {}).get("best_model", "n/a"),
        summary.get("anomaly", {}).get("best_model", "n/a"),
        summary.get("nlp", {}).get("best_model", "n/a"),
    ],
    "Status": ["Deployed", "Deployed", "Deployed"],
}

st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

st.markdown("## Key Metrics")

col1, col2, col3, col4 = st.columns(4)
trained_model_count = 0
if summary:
    trained_model_count = (
        len(summary.get("forecast", {}).get("metrics", {}))
        + len(summary.get("anomaly", {}).get("metrics", {}))
        + len(summary.get("nlp", {}).get("metrics", {}))
    )

col1.metric("Models Trained", str(trained_model_count), delta="Phase 3 complete")
col2.metric("Test Datasets", "3", delta="All included")
col3.metric("Deployment", "Streamlit", delta="Phase 5 ready" if ready else "Artifacts missing")
col4.metric("Error Analysis", "Available" if analysis else "Missing", delta="Phase 4")

st.markdown("## Architecture")

st.code("""
SensorMind-ml-portfolio/
|- src/
|  |- data/          # Loaders & preprocessing
|  |- features/      # Feature engineering
|  |- models/        # Training & inference
|  '- utils/         # Helpers & metrics
|- notebooks/        # EDA & experiments
|- app/              # Streamlit pages
|- models/           # Trained artifacts
|- tests/            # Unit tests
'- reports/          # Generated figures
""")

st.markdown("## Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ML/DL")
    st.markdown("""
    - scikit-learn
    - XGBoost, LightGBM
    - PyTorch, TensorFlow
    - Prophet, ARIMA
    """)

with col2:
    st.markdown("### Modern AI")
    st.markdown("""
    - HuggingFace Transformers
    - Sentence-Transformers
    - LangChain
    - OpenAI API
    """)

with col3:
    st.markdown("### Production")
    st.markdown("""
    - Streamlit (UI)
    - MLflow (Tracking)
    - Docker (Containers)
    - GitHub Actions (CI/CD)
    """)

st.markdown("---")
st.markdown("""
### Get Started
1. Navigate to a project using the buttons above
2. Explore model predictions & performance
3. Check the About page for technical details

### Resources
""")

error_report_path = PROJECT_ROOT / "reports" / "error_analysis.md"
if error_report_path.exists():
    report_text = error_report_path.read_text(encoding="utf-8")
    col_open, col_download = st.columns([1, 1])
    with col_open:
        if st.button("Open Error Analysis Report", use_container_width=True):
            st.session_state["show_error_report"] = True
    with col_download:
        st.download_button(
            "Download Error Analysis Report",
            data=report_text,
            file_name="error_analysis.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.session_state.get("show_error_report", False):
        st.markdown("#### Error Analysis Report")
        st.markdown(report_text)
else:
    st.info("Error analysis report not found yet. Run `python scripts/run_phase4.py` first.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    SensorMind ML Portfolio | Phase 5 Deployment Layer | Built with Streamlit
</div>
""", unsafe_allow_html=True)

