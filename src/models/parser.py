"""Parser for extracting structured data from plaintext using LLMs.

Uses a schema-driven approach compatible with any pipeline stage.
"""

import json
import logging
import re
from typing import Any, Type

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


class Parser:
    """Parser for extracting structured data from plaintext using LLMs.

    Can be used across all pipeline stages with different schemas and prompts.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        """Initialize parser with a specific model.

        Args:
            model_name: HuggingFace model identifier (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')
            cache_dir: Directory for caching model files (for Hartree offline mode)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or MODEL_CACHE_DIR

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load the parser model into memory."""
        logger.info(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def parse(
        self,
        text: str,
        schema: Type[BaseModel],
        context: dict[str, Any],
        prompt_template: str,
        max_new_tokens: int = 400,
        temperature: float = 0.1,
    ) -> list[dict]:
        """Parse plaintext into structured data according to a Pydantic schema.

        Args:
            text: The plaintext to parse (e.g., biomedical reasoning from reasoning model)
            schema: Pydantic model class defining the output structure
            context: Dictionary of variables to format into the prompt template
            prompt_template: Prompt template with placeholders (must include {plaintext})
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            List of validated dictionaries matching the schema, or empty list if parsing fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Add the text to context
        context["plaintext"] = text

        # Format the prompt
        prompt = prompt_template.format(**context)

        # Generate with the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        logger.debug(f"Parsing {len(text)} chars of text with {schema.__name__}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        logger.debug(f"Raw parser output length: {len(response)} chars")

        # Extract and validate JSON
        result = self._extract_and_validate_json(response, schema)
        logger.info(f"Extracted {len(result)} valid {schema.__name__} objects")

        return result

    def _extract_and_validate_json(
        self,
        response: str,
        schema: Type[BaseModel],
    ) -> list[dict]:
        """Extract JSON from response and validate against schema.

        Tries multiple strategies to extract JSON from model output.
        """
        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(response.strip())
            return self._validate_data(data, schema)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        if "```" in response:
            match = re.search(r"```(?:json)?\\s*([\\s\\S]*?)```", response)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    return self._validate_data(data, schema)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find array brackets (most common for Mistral)
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(response[start:end])
                return self._validate_data(data, schema)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try single object and wrap in array
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(response[start:end])
                return self._validate_data([obj], schema)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to extract valid JSON for {schema.__name__}")
        return []

    def _validate_data(
        self,
        data: Any,
        schema: Type[BaseModel],
    ) -> list[dict]:
        """Validate data against Pydantic schema.

        Args:
            data: Raw data from JSON parsing (should be list or dict)
            schema: Pydantic model class to validate against

        Returns:
            List of validated dictionaries
        """
        # Handle empty response
        if not data:
            return []

        # Ensure we have a list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            logger.warning(f"Expected list or dict, got {type(data)}")
            return []

        # Validate each item
        validated = []
        for item in data:
            if not isinstance(item, dict):
                logger.debug(f"Skipping non-dict item: {type(item)}")
                continue

            try:
                # Validate with Pydantic
                validated_obj = schema(**item)
                # Convert back to dict for consistency
                validated.append(validated_obj.model_dump())
            except ValidationError as e:
                logger.debug(f"Validation failed for item: {e}")
                continue

        return validated
