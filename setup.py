from setuptools import setup, find_packages

setup(
    name="cribbage-rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "streamlit>=1.29.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train=run:train_agent",
            "play=run:run_streamlit",
        ],
    },
)