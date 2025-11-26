"""Configuration for the pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AGENTS_CSV = DATA_DIR / "all-agents.csv"
PATHWAYS_CSV = DATA_DIR / "all-tsnc-pathways.csv"

# Model cache directory (for Hartree offline mode)
MODEL_CACHE_DIR = PROJECT_ROOT / "hf_cache"

# Model configuration
REASONING_MODEL = "microsoft/MediPhi-PubMed"  # Biomedical reasoning (Step 1)
PARSER_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # JSON extraction (Step 2)

# All models to download for offline use
MODELS = [REASONING_MODEL, PARSER_MODEL]