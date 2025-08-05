#!/bin/bash

# Create Virtual Environment
python -m venv .venv

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