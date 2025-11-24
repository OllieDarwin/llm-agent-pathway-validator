#!/usr/bin/env python3
"""Pre-download models for offline use (Hartree deployment)."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODELS, MODEL_CACHE_DIR


def download_model(model_name: str, cache_dir: Path) -> None:
    """Download model and tokenizer to cache."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"{'='*60}")

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    print(f"Tokenizer saved to {cache_dir}")

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    print(f"Model saved to {cache_dir}")

    # Clean up memory
    del model
    del tokenizer


def main():
    print("=== Model Download Script ===")
    print(f"Cache directory: {MODEL_CACHE_DIR}")
    print(f"Models to download: {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m}")

    # Create cache directory
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        try:
            download_model(model_name, MODEL_CACHE_DIR)
        except Exception as e:
            print(f"ERROR downloading {model_name}: {e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("All models downloaded successfully!")
    print(f"Cache location: {MODEL_CACHE_DIR}")
    print("\nFor Hartree offline use, copy hf_cache/ to the compute node.")
    print("="*60)


if __name__ == "__main__":
    main()
