#!/bin/bash
# Setup script for local development

set -e

echo "=== LLM Agent Pathway Validator Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create cache directories
mkdir -p hf_cache

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run Stage 1 tests:"
echo "  python tests/test_stage1.py"
echo ""
echo "Note: First run will download models and may take some time."
