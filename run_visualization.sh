#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the visualization
echo "Running visualization..."
python dimension_transition.py

echo "Visualization complete! The HTML file should have opened in your browser."
echo "If not, manually open 'dimension_transition.html' in your browser."
