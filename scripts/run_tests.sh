#!/bin/bash
# Run Stage 1 tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Set HuggingFace cache
export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"

echo "=== Running Stage 1 Tests ==="
echo "Cache directory: $HF_HOME"
echo ""

python tests/test_stage1.py "$@"
