from setuptools import setup, find_packages

setup(
    name="faclon-ml-portfolio",
    version="0.1.0",
    description="End-to-end ML portfolio project: Time-series forecasting, anomaly detection, and NLP deployment",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourname/faclon-ml-portfolio",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "torch>=2.1.0",
        "streamlit>=1.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
        ]
    },
)
