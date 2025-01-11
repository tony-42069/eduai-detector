from setuptools import setup, find_packages

setup(
    name="eduai-detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "setuptools>=65.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
        ]
    },
    author="Tony",
    author_email="your.email@example.com",
    description="An AI text detection tool for educational use",
    keywords="ai detection, education, nlp",
    python_requires=">=3.8",
)