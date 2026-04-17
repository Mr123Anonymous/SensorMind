# Faclon ML Portfolio - End-to-End Data Science Project

**Status**: 🚀 Phase 1 Complete - Project Setup Done

A comprehensive machine learning portfolio project showcasing **time-series forecasting**, **anomaly detection**, and **NLP** with **LLM integration**. Built for the Faclon Labs Data Science Internship (3-6 months).

---

## 📋 Project Overview

This project demonstrates the modern AI/ML stack across multiple domains:

| Task | Dataset | Models | Skills |
|------|---------|--------|--------|
| **Time-Series Forecasting** | PGCB Grid (UCI) | ARIMA, Prophet, LSTM, XGBoost | Regression, DL, Classical ML |
| **Anomaly Detection** | Yahoo (HuggingFace) | Isolation Forest, Autoencoder, LSTM | Classification, DL, Unsupervised |
| **NLP Sentiment** (optional) | Kaggle/Custom | TF-IDF, BERT, LLM Prompting | NLP, Embeddings, Modern AI |
| **Deployment** | All Models | Streamlit + MLflow | DevOps, ML Ops, APIs |

---

## 🎯 What You'll Learn

- ✅ **ML Fundamentals**: Regression, classification, clustering, feature engineering
- ✅ **Deep Learning**: LSTM, autoencoders, transformer embeddings
- ✅ **Modern AI Stack**: LLM integration, prompt engineering, embeddings
- ✅ **Production ML**: Experiment tracking (MLflow), model versioning, deployment
- ✅ **Software Engineering**: Modular code, testing, CI/CD, Docker
- ✅ **Data Science Workflow**: EDA → Preprocessing → Modeling → Evaluation → Deployment

---

## 📁 Project Structure

```
faclon-ml-portfolio/
├── README.md                    # This file
├── requirements.txt             # All dependencies
├── setup.py                     # Package installation
├── Dockerfile & docker-compose.yml
├── Makefile                     # Common commands
│
├── src/
│   ├── config.py               # Centralized configuration
│   ├── data/
│   │   ├── loaders.py          # Load PGCB, Yahoo, synthetic data
│   │   └── preprocessing.py    # Clean, normalize, feature engineering
│   ├── features/
│   │   └── engineering.py      # Create lag, rolling, cyclical features
│   ├── models/
│   │   ├── train.py            # Training loops
│   │   ├── predict.py          # Inference
│   │   └── evaluate.py         # Metrics
│   └── utils/
│       └── helpers.py          # Logging, metrics, utilities
│
├── notebooks/
│   ├── 01_pgcb_exploration.ipynb       # EDA for forecasting
│   ├── 01_anomaly_exploration.ipynb    # EDA for anomaly detection
│   ├── 03_pgcb_modeling.ipynb          # Model experiments
│   └── 03_anomaly_modeling.ipynb       # Anomaly detection experiments
│
├── app/
│   ├── streamlit_app.py                # Main app entry point
│   └── pages/
│       ├── 01_Forecasting.py           # PGCB prediction demo
│       ├── 02_Anomaly_Detection.py    # Sensor anomaly demo
│       ├── 03_NLP_Sentiment.py        # Sentiment analysis (optional)
│       ├── 04_Model_Comparison.py     # Performance dashboard
│       └── 05_About.py                 # Project info
│
├── models/                     # Trained model artifacts
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned datasets
├── tests/                      # Unit tests
├── reports/                    # Generated reports & figures
└── .github/workflows/          # CI/CD pipeline
```

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/yourusername/faclon-ml-portfolio.git
cd faclon-ml-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/streamlit_app.py
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
# Connect your GitHub repo at https://share.streamlit.io
```

---

## 📊 Datasets

### 1. PGCB Grid Forecasting (Time-Series)
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/1175)
- **Size**: 92.65K hourly records
- **Task**: Predict electricity generation
- **Relevance**: Direct application to Faclon's IoT/infrastructure focus

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

## 🔧 Commands

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

# Run Streamlit app
make streamlit

# Train all models
make train

# Download datasets
make download-data
```

---

## 📈 Model Performance

### Time-Series Forecasting (PGCB)
| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| ARIMA | TBD | TBD | TBD | TBD |
| Prophet | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |
| **XGBoost** | **TBD** | **TBD** | **TBD** | **TBD** |

### Anomaly Detection
| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Isolation Forest | TBD | TBD | TBD | TBD |
| **Autoencoder** | **TBD** | **TBD** | **TBD** | **TBD** |
| One-Class SVM | TBD | TBD | TBD | TBD |

---

## 🎓 Learning Outcomes

By completing this project, you'll understand:

1. **Data Science Workflow**: EDA → Feature Engineering → Modeling → Evaluation
2. **Classical ML**: Regression (ARIMA), Classification (Logistic, Tree-based)
3. **Deep Learning**: LSTM for sequences, Autoencoders for unsupervised learning
4. **Modern AI**: LLM integration, embeddings, prompt engineering
5. **Production ML**: Experiment tracking, model versioning, APIs
6. **DevOps**: Docker, GitHub Actions, cloud deployment
7. **Software Engineering**: Modular design, testing, documentation

---

## 🛠️ Tech Stack

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

## 📝 Development Roadmap

### Phase 1 ✅ (Days 1-3)
- [x] Project structure & setup
- [x] Docker configuration
- [x] Configuration management

### Phase 2 🔄 (Days 4-10)
- [ ] EDA notebooks (3 datasets)
- [ ] Data loaders & preprocessing
- [ ] Feature engineering

### Phase 3 (Days 11-28)
- [ ] Time-series forecasting models
- [ ] Anomaly detection models
- [ ] Optional: NLP sentiment models

### Phase 4 (Days 29-32)
- [ ] Unit tests
- [ ] Error analysis
- [ ] MLflow experiment tracking

### Phase 5 (Days 33-40)
- [ ] Streamlit app (5 pages)
- [ ] Model API
- [ ] Local testing

### Phase 6 (Days 41-42)
- [ ] Cloud deployment
- [ ] Comprehensive documentation
- [ ] GitHub repo finalization

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design & data flow
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Cloud deployment guide
- [API.md](docs/API.md) - API endpoints & examples
- [CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) - Codebase navigation

---

## 🤝 Contributing

Contributions welcome! Please:
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Write tests & docstrings
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Resources

- **Faclon Labs**: [Website](https://faclon.io)
- **Internship Info**: See original job posting
- **Dataset Links**:
  - [PGCB Dataset](https://archive.ics.uci.edu/dataset/1175)
  - [Yahoo Anomaly](https://huggingface.co/datasets/YahooResearch/...)

---

## 📄 License

This project is MIT licensed. See LICENSE file for details.

---

## 🎉 Getting Help

- **Issues**: Open a GitHub issue for bugs/questions
- **Discussions**: Join GitHub Discussions for ideas
- **Email**: your.email@example.com

---

**Happy Learning! 🚀** Build amazing things with Faclon!
