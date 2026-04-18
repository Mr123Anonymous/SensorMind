# Code Structure

## Root
- README.md: project overview and run instructions
- requirements.txt: dependencies
- Makefile: command shortcuts
- Dockerfile, docker-compose.yml: container execution

## src/
- config.py: central constants and model/data configuration
- data/
  - loaders.py: dataset loading and synthetic fallback
  - preprocessing.py: cleaning, feature creation, splits
- features/
  - engineering.py: feature generation utilities
- models/
  - train.py: model training and artifact persistence
  - predict.py: inference wrappers
  - evaluate.py: metric utilities
- analysis/
  - error_analysis.py: phase 4 diagnostics reports
- utils/
  - helpers.py: logging, metrics, IO helpers

## scripts/
- run_phase2.py: data pipeline execution
- run_phase3.py: model training execution
- run_phase4.py: tests/error analysis execution
- run_phase6.py: release readiness checks

## app/
- Home.py: deployment home page
- _shared.py: cached model/data loaders for pages
- pages/
  - 01_Forecasting.py
  - 02_Anomaly_Detection.py
  - 03_NLP_Sentiment.py
  - 04_Model_Comparison.py
  - 05_About.py

## tests/
- test_preprocessing.py
- test_phase2_pipeline.py
- test_phase3_models.py
- test_predict_and_evaluate.py
- test_error_analysis.py

## reports/
- error_analysis.json
- error_analysis.md
- figures/
