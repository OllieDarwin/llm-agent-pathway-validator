"""Configuration for the pipeline."""

import os
from pathlib import Path
from huggingface_hub import login

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AGENTS_CSV = DATA_DIR / "all-agents.csv"
PATHWAYS_CSV = DATA_DIR / "all-tsnc-pathways.csv"

# Model cache directory (for Hartree offline mode)
MODEL_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# HuggingFace authentication
HF_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ Logged in to HuggingFace using HUGGINGFACE_ACCESS_TOKEN")

# Model configuration
REASONING_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Biomedical reasoning (Step 1)
PARSER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # JSON extraction (Step 2)

# Quantization settings (for large models like 70B)
USE_4BIT_QUANTIZATION = False  # Set to True for 70B models to reduce VRAM ~140GB -> ~35GB
USE_8BIT_QUANTIZATION = False  # Alternative: 8-bit uses ~70GB

# All models to download for offline use
MODELS = [REASONING_MODEL, PARSER_MODEL]