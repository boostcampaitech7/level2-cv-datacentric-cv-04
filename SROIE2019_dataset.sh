#!/bin/bash

# Exit immediately if any command fails
set -e

# Variables
PYTHON_SCRIPT="download_with_kagglehub.py"

# Step 1: Install kagglehub if not already installed
echo "Installing kagglehub..."
pip install kagglehub

# Step 2: Run the Python script to download the dataset
echo "Running download script..."
python "$PYTHON_SCRIPT"

echo "Dataset download complete!"