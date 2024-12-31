#!/bin/bash

# Install poetry if not already installed
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Create necessary directories
mkdir -p models logs

echo "Setup completed! You can now:"
echo "1. Train the DQN agent: poetry run python run.py train"
echo "2. Play via Streamlit: poetry run python run.py play"
echo "3. Play via CLI: poetry run python ui/cli/play_game.py --player 0"