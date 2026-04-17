"""
Streamlit App - Home Page
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Faclon ML Portfolio",
    page_icon="📊",
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

# Header
st.title("📊 Faclon ML Portfolio")
st.markdown("### End-to-End Machine Learning Project: Time-Series, Anomaly Detection & NLP")

# Introduction
st.markdown("""
Welcome to the **Faclon Labs Data Science Internship Portfolio** project! 

This comprehensive ML project showcases:
- 🔮 **Time-Series Forecasting** (PGCB Grid Energy)
- 🚨 **Anomaly Detection** (Sensor Fault Detection)
- 🤖 **NLP with Modern AI** (Sentiment Analysis + LLMs)
- 📦 **Production Deployment** (Streamlit + MLflow + Docker)

---
""")

# Navigation
st.markdown("## 🚀 Quick Navigation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📈 Forecasting", use_container_width=True):
        st.switch_page("pages/01_Forecasting.py")

with col2:
    if st.button("🚨 Anomalies", use_container_width=True):
        st.switch_page("pages/02_Anomaly_Detection.py")

with col3:
    if st.button("💬 NLP/Sentiment", use_container_width=True):
        st.switch_page("pages/03_NLP_Sentiment.py")

with col4:
    if st.button("📊 Comparison", use_container_width=True):
        st.switch_page("pages/04_Model_Comparison.py")

with col5:
    if st.button("ℹ️ About", use_container_width=True):
        st.switch_page("pages/05_About.py")

# Project Overview
st.markdown("## 📋 Project Overview")

overview_data = {
    "Task": ["Time-Series Forecasting", "Anomaly Detection", "NLP Sentiment"],
    "Dataset": ["PGCB Grid", "Yahoo Time-Series", "Kaggle News"],
    "Models Trained": ["ARIMA, Prophet, LSTM, XGBoost", "Isolation Forest, Autoencoder", "BERT, LLM"],
    "Status": ["🔄 In Progress", "🔄 In Progress", "⏱️ Planned"],
}

st.dataframe(pd.DataFrame(overview_data), use_container_width=True)

# Key Metrics
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Models Trained", "0/12", delta="Coming soon")
col2.metric("Test Datasets", "3", delta="All included")
col3.metric("Deployment", "Streamlit", delta="Ready")
col4.metric("Experiments Logged", "0", delta="MLflow tracking")

# Project Structure
st.markdown("## 📁 Architecture")

st.code("""
faclon-ml-portfolio/
├── src/
│   ├── data/          # Loaders & preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Training & inference
│   └── utils/         # Helpers & metrics
├── notebooks/         # EDA & experiments
├── app/              # Streamlit pages
├── models/           # Trained artifacts
├── tests/            # Unit tests
└── reports/          # Generated figures
""")

# Tech Stack
st.markdown("## 🛠️ Technology Stack")

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

# Call to Action
st.markdown("---")
st.markdown("""
### 🎯 Get Started
1. Navigate to a project using the buttons above
2. Explore model predictions & performance
3. Check the About page for technical details

### 📚 Resources
- [GitHub Repository](https://github.com/yourusername/faclon-ml-portfolio)
- [Faclon Labs](https://faclon.io)
- [Project Documentation](reports/README.md)
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Faclon ML Portfolio • Phase 1 Complete • Built with Streamlit
</div>
""", unsafe_allow_html=True)
