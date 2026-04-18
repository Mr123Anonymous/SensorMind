# Phase 4 Error Analysis

## Forecasting
- Best model: random_forest
- MAE: 308.6512
- P90 absolute error: 700.0890
- Residual mean: 287.4414
- Residual std: 321.7330
- Worst sample indices: [3902, 1299, 6577, 2168, 6647, 1430, 8218, 3163, 7962, 7437]

## Anomaly Detection
- Best model: autoencoder
- Threshold: 0.000771
- TP: 46, FP: 152, FN: 0, TN: 552
- False-positive examples: [0, 2, 3, 4, 6, 7, 30, 31, 52, 58]
- False-negative examples: []

## NLP
- Best model: tfidf_logistic_regression
- Accuracy: 1.0000
- F1: 1.0000
- Risk note: Potential overfitting to templated synthetic phrases.
