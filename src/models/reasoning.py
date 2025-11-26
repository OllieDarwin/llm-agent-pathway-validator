"""Reasoning model wrapper for biomedical text generation."""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ReasoningModel:
    """Wrapper for biomedical reasoning models (e.g., microsoft/MediPhi-PubMed)."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load model and tokenizer. Call this before inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = True,
    ) -> str:
        """Generate text completion for the given prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Debug: check input token count
        input_token_count = inputs["input_ids"].shape[1]
        logger.debug(f"Input tokens: {input_token_count}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        output_token_count = outputs[0].shape[0]
        new_token_count = output_token_count - inputs["input_ids"].shape[1]
        logger.debug(f"Output tokens generated: {new_token_count}")

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        logger.debug(f"Response length: {len(response)} chars")
        return response.strip()

