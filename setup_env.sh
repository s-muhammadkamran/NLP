#!/bin/bash

# Create Virtual Environment
if [ -d ".venv" ]; then
    echo ".venv already exists"
else
    python -m venv .venv
    echo "Virtual environment created in .venv"
fi

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements with SSL bypass (only if needed; use with caution)
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

echo
echo "Environment setup complete."