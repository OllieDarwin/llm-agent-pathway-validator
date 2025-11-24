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
MEDIPHI_MODEL = "microsoft/MediPhi-PubMed"  # Biomedical reasoning
PARSER_MODEL = "osmosis-ai/Osmosis-Structure-0.6B"  # JSON extraction

# All models to download for offline use
MODELS = [MEDIPHI_MODEL, PARSER_MODEL]

# Legacy alias
MODEL_NAME = MEDIPHI_MODEL

# Stage 1 filtering thresholds
MAX_CANCER_TYPES_PER_PAIR = 3
MAX_PATHWAYS_CHEMOTHERAPY = 3
MAX_PATHWAYS_TARGETED = 2
MAX_PATHWAYS_IMMUNOTHERAPY = 1

# Literature search (Stage 2)
EXA_API_KEY = None  # Set via environment variable
PUBLICATION_YEAR_MIN = 2015
PUBLICATION_YEAR_MAX = 2025
MAX_PUBLICATIONS_PER_QUERY = 10

# Confidence thresholds
MIN_CONFIDENCE_FOR_PASS = 70
