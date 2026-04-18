# Deployment Guide

## Prerequisites
- Python 3.11+ recommended
- pip and virtual environment support
- Optional: Docker Desktop

## Local Deployment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_phase2.py
python scripts/run_phase3.py
python scripts/run_phase4.py
streamlit run app/Home.py
```

## Docker Deployment
```bash
docker-compose up -d
```
- Streamlit App: http://localhost:8501
- Jupyter: http://localhost:8888

## Streamlit Community Cloud Deployment
1. Push repository to your remote host
2. Open Streamlit Community Cloud dashboard
3. Create a new app from this repo
4. Set entrypoint to app/Home.py
5. Deploy

## Environment Variables (Optional)
- API_HOST (default: 0.0.0.0)
- API_PORT (default: 8000)

## Post-Deployment Validation
1. Home page loads without errors
2. Forecasting page renders predictions
3. Anomaly page renders score graph and threshold controls
4. NLP page returns predictions and probabilities
5. Model comparison renders all charts
6. About page shows artifact readiness

## Troubleshooting
- Missing artifacts: run Phase 2-4 scripts first
- Import errors: ensure project root is active working directory
- Empty dashboards: verify data/processed and models contain generated files
