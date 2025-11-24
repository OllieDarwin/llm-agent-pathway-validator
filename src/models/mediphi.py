"""MediPhi-PubMed model wrapper for biomedical text generation."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class MediPhiModel:
    """Wrapper for microsoft/MediPhi-PubMed model."""

    def __init__(
        self,
        model_name: str = "microsoft/MediPhi-PubMed",
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
        print(f"[MEDIPHI DEBUG] Input tokens: {input_token_count}")

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
        print(f"[MEDIPHI DEBUG] Output tokens generated: {new_token_count}")

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"[MEDIPHI DEBUG] Response length before truncation: {len(response)} chars")

        # Stop at conclusion - prevent over-generation
        response = self._truncate_at_conclusion(response)
        print(f"[MEDIPHI DEBUG] Response length after truncation: {len(response)} chars")
        return response.strip()

    def _truncate_at_conclusion(self, text: str) -> str:
        """Truncate response after the conclusion section."""
        # Look for end of conclusion section
        markers = [
            "\n\nAs a clinical",  # Start of new prompt
            "\n\n###",  # New section marker
            "\n\n---",  # Separator
            "\nBegin your analysis:",  # Prompt leak
        ]

        original_length = len(text)
        for marker in markers:
            if marker in text:
                print(f"[TRUNCATE DEBUG] Found marker '{marker[:20]}...' at position {text.index(marker)}")
                text = text.split(marker)[0]
                print(f"[TRUNCATE DEBUG] Truncated from {original_length} to {len(text)} chars")
                break

        return text

    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> dict | list | None:
        """Generate and parse JSON response."""
        response = self.generate(prompt, max_new_tokens, temperature)

        # Try to extract JSON from response
        try:
            # Handle cases where model outputs extra text
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to find JSON-like content
            start = response.find("[") if "[" in response else response.find("{")
            end = response.rfind("]") + 1 if "]" in response else response.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass
            return None
