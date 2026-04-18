# SensorMind ML Portfolio - End-to-End Data Science Project

**Status**: Production-ready portfolio project with complete docs, deployment paths, and validation checks

A comprehensive machine learning portfolio project showcasing **time-series forecasting**, **anomaly detection**, and **NLP** with **LLM integration**. Built as a standalone resume project.

---

##  Project Overview

This project demonstrates the modern AI/ML stack across multiple domains:

| Task | Dataset | Models | Skills |
|------|---------|--------|--------|
| **Time-Series Forecasting** | PGCB Grid (UCI) | ARIMA, Prophet, LSTM, XGBoost | Regression, DL, Classical ML |
| **Anomaly Detection** | Yahoo (HuggingFace) | Isolation Forest, Autoencoder, LSTM | Classification, DL, Unsupervised |
| **NLP Sentiment** (optional) | Kaggle/Custom | TF-IDF, BERT, LLM Prompting | NLP, Embeddings, Modern AI |
| **Deployment** | All Models | Streamlit + MLflow | DevOps, ML Ops, APIs |

---

##  Key Skills Demonstrated

-  **ML Fundamentals**: Regression, classification, clustering, feature engineering
-  **Deep Learning**: LSTM, autoencoders, transformer embeddings
-  **Modern AI Stack**: LLM integration, prompt engineering, embeddings
-  **Production ML**: Experiment tracking (MLflow), model versioning, deployment
-  **Software Engineering**: Modular code, testing, CI/CD, Docker
-  **Data Science Workflow**: EDA -> Preprocessing -> Modeling -> Evaluation -> Deployment

---

##  Project Structure

```
SensorMind-ml-portfolio/
 README.md                    # This file
 requirements.txt             # All dependencies
 setup.py                     # Package installation
 Dockerfile & docker-compose.yml
 Makefile                     # Common commands

 src/
    config.py               # Centralized configuration
    data/
       loaders.py          # Load PGCB, Yahoo, synthetic data
       preprocessing.py    # Clean, normalize, feature engineering
    features/
       engineering.py      # Create lag, rolling, cyclical features
    models/
       train.py            # Training loops
       predict.py          # Inference
       evaluate.py         # Metrics
    utils/
        helpers.py          # Logging, metrics, utilities

 notebooks/
    01_pgcb_exploration.ipynb       # EDA for forecasting
    01_anomaly_exploration.ipynb    # EDA for anomaly detection

 app/
   Home.py                         # Main app entry point
    pages/
        01_Forecasting.py           # PGCB prediction demo
        02_Anomaly_Detection.py    # Sensor anomaly demo
        03_NLP_Sentiment.py        # Sentiment analysis (optional)
        04_Model_Comparison.py     # Performance dashboard
        05_About.py                 # Project info

 models/                     # Trained model artifacts
 data/
    raw/                    # Original datasets
    processed/              # Cleaned datasets
 tests/                      # Unit tests
 reports/                    # Generated reports & figures
 .github/workflows/          # CI/CD pipeline
```

---

##  Quick Start

> **Python version**: Use Python **3.11** for this pinned dependency set.

### Option 1: Local Setup

```bash
# Clone the repository and enter the project directory
git clone https://github.com/Mr123Anonymous/SensorMind.git
cd SensorMind

# Create virtual environment
py -3.11 -m venv .venv  # Windows (recommended)
python -m venv .venv  # Linux/macOS (ensure this python is 3.11)

# Activate the virtual environment
.\\.venv\\Scripts\\Activate.ps1  # Windows PowerShell
source .venv/bin/activate  # Linux/macOS

# Verify interpreter version inside the venv
python --version  # Should print Python 3.11.x

# Install dependencies
pip install -r requirements.txt

# Initiate Phase 2 data preparation
python scripts/run_phase2.py

# Initiate Phase 3 model training
python scripts/run_phase3.py

# Initiate Phase 4 (error analysis only)
python scripts/run_phase4.py

# Launch Phase 5 deployment app
streamlit run app/Home.py

# Run Phase 6 release-readiness checks
python scripts/run_phase6.py
```

### Option 2: Docker Setup

```bash
# Build and run containers
docker-compose up -d

# App runs at http://localhost:8501
# Jupyter Lab at http://localhost:8888
```

### Option 3: Cloud Deployment

```bash
# Deploy to Streamlit Cloud (free tier)
# Connect your repository at https://share.streamlit.io
```

---

##  Datasets

### 1. PGCB Grid Forecasting (Time-Series)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/1175)
- **Size**: 92.65K hourly records
- **Task**: Predict electricity generation
- **Relevance**: Direct application to SensorMind's IoT/infrastructure focus

### 2. Yahoo Anomaly Detection
- **Source**: [HuggingFace](https://huggingface.co/datasets/YahooResearch/ydata-labeled-time-series-anomalies-v1_0)
- **Size**: 367 time-series with labeled anomalies
- **Task**: Detect sensor/system anomalies
- **Relevance**: Real-world labeled data for grid monitoring

### 3. NLP Sentiment Classification (Optional)
- **Source**: Kaggle News/Twitter
- **Task**: Text classification with embeddings & LLMs
- **Relevance**: Demonstrates modern AI stack

---

##  Commands

```bash
# Install dependencies
make install

# Run tests (with coverage)
make test

# Lint code
make lint

# Format code with black
make format

# Clean cache/artifacts
make clean

# Start Docker containers
make docker-up

# Stop Docker containers
make docker-down

# Run Streamlit App
make streamlit

# Train all models
make train

# Download datasets
make download-data

# Initiate Phase 2 pipeline
python scripts/run_phase2.py

# Initiate Phase 3 pipeline
python scripts/run_phase3.py

# Initiate Phase 4 pipeline
python scripts/run_phase4.py

# Launch Phase 5 app
streamlit run app/Home.py

# Run Phase 6 checks
python scripts/run_phase6.py
```

---

##  Model Performance

Model metrics are generated by Phase 3 and written to [data/processed/phase3_summary.json](data/processed/phase3_summary.json). Phase 4 consumes those artifacts to produce [reports/error_analysis.json](reports/error_analysis.json) and [reports/error_analysis.md](reports/error_analysis.md).

The shipped baselines are:

- Forecasting: Ridge, Random Forest, and a PyTorch MLP
- Anomaly detection: Isolation Forest and a PyTorch autoencoder
- NLP sentiment: TF-IDF + Logistic Regression

---

##  Learning Outcomes

By completing this project, you'll understand:

1. **Data Science Workflow**: EDA -> Feature Engineering -> Modeling -> Evaluation
2. **Classical ML**: Regression (ARIMA), Classification (Logistic, Tree-based)
3. **Deep Learning**: LSTM for sequences, Autoencoders for unsupervised learning
4. **Modern AI**: LLM integration, embeddings, prompt engineering
5. **Production ML**: Experiment tracking, model versioning, APIs
6. **DevOps**: Docker, CI/CD workflows, cloud deployment
7. **Software Engineering**: Modular design, testing, documentation

---

##  Tech Stack

### Core ML
- **scikit-learn**: Classical ML algorithms
- **pandas/numpy**: Data manipulation
- **statsmodels**: ARIMA, time-series analysis

### Deep Learning
- **PyTorch**: LSTM, Autoencoders
- **TensorFlow**: Alternative DL framework
- **Transformers**: BERT, sentence-transformers for NLP

### Modern AI
- **LangChain**: LLM orchestration
- **OpenAI API**: Claude/GPT integration
- **HuggingFace**: Pre-trained models

### Deployment
- **Streamlit**: Interactive dashboard
- **MLflow**: Experiment tracking
- **FastAPI**: REST API (optional)
- **Docker**: Containerization

---

##  Development Roadmap

### Phase 1  (Days 1-3)
- [x] Project structure & setup
- [x] Docker configuration
- [x] Configuration management

### Phase 2  (Days 4-10)
- [x] EDA notebooks (PGCB + anomaly)
- [x] Data loaders & preprocessing
- [x] Feature engineering

### Phase 3 (Days 11-28)
- [x] Time-series forecasting baseline models
- [x] Anomaly detection baseline models
- [x] Optional: NLP sentiment baseline models

### Phase 4 (Days 29-32)
- [x] Unit tests
- [x] Error analysis
- [x] MLflow experiment tracking

### Phase 5 (Days 33-40)
- [x] Streamlit App (5 pages)
- [x] Model API/inference integration
- [x] Local testing

### Phase 6 (Days 41-42)
- [x] Cloud deployment
- [x] Comprehensive documentation
- [x] Repository finalization checklist

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

##  Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design & data flow
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Cloud deployment guide
- [API.md](docs/API.md) - API endpoints & examples
- [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) - Codebase navigation
- [RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) - Final release checklist

---

##  Contributing

Contributions welcome! Please:
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Write tests & docstrings
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  Resources

- **Project Repository**: Source code and documentation in this repo
- **Dataset Links**:
  - [PGCB Dataset](https://archive.ics.uci.edu/dataset/1175)
  - [Yahoo Anomaly](https://huggingface.co/datasets/YahooResearch/...)

---

##  License

This project is MIT licensed. See LICENSE file for details.

---

##  Getting Help

- **Issues**: Open a repository issue for bugs/questions
- **Discussions**: Use repository discussions for ideas

---

**Happy Learning!** Build amazing things with SensorMind!


