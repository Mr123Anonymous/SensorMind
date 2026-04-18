# Architecture

## System Overview
The project follows an end-to-end ML pipeline with six phases:
1. Data ingestion and preprocessing
2. Exploratory analysis and feature engineering
3. Baseline model training
4. Error analysis and experiment tracking
5. Interactive Streamlit deployment
6. Documentation and release readiness

## Data and Model Flow
```text
data/raw
  -> scripts/run_phase2.py
  -> data/processed/*.npy
  -> scripts/run_phase3.py
  -> models/* + data/processed/phase3_summary.json
  -> scripts/run_phase4.py
  -> reports/error_analysis.json + reports/error_analysis.md
  -> app/Home.py (Phase 5 dashboards)
```

## Runtime Components
- Data layer: src/data/loaders.py, src/data/preprocessing.py
- Feature layer: src/features/engineering.py
- Model layer: src/models/train.py, src/models/predict.py, src/models/evaluate.py
- Analysis layer: src/analysis/error_analysis.py
- UI layer: app/Home.py + app/pages/*
- CI/CD layer: .github/workflows/ci-cd.yml

## Model Families
- Forecasting: Ridge, Random Forest, PyTorch MLP
- Anomaly detection: Isolation Forest, PyTorch Autoencoder
- NLP sentiment: TF-IDF + Logistic Regression

## Observability and Tracking
- Metrics are persisted to JSON in models/ and data/processed/
- Error-analysis outputs are persisted in reports/
- MLflow experiment tracking is integrated in src/models/train.py

## Deployment Targets
- Local Streamlit: streamlit run app/Home.py
- Docker: docker-compose up -d
- Streamlit Community Cloud: via app/Home.py entrypoint
