.PHONY: help install test lint format clean docker-up docker-down streamlit train

help:
	@echo "Faclon ML Portfolio - Available Commands"
	@echo "========================================"
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run pytest"
	@echo "  make lint            - Run flake8 and mypy"
	@echo "  make format          - Format code with black"
	@echo "  make clean           - Remove cache, build artifacts"
	@echo "  make docker-up       - Start Docker containers"
	@echo "  make docker-down     - Stop Docker containers"
	@echo "  make streamlit       - Run Streamlit app locally"
	@echo "  make train           - Train all models"
	@echo "  make download-data   - Download datasets"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	flake8 src tests --max-line-length=100
	mypy src --ignore-missing-imports

format:
	black src tests notebooks/

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage .mypy_cache .pytest_cache
	rm -rf mlruns artifacts

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

streamlit:
	streamlit run app/streamlit_app.py

train:
	python -m src.models.train

download-data:
	python scripts/download_data.py
