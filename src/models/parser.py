"""Parser model for extracting structured JSON from plaintext LLM output.

Uses Mistral-7B-Instruct for reliable JSON extraction from biomedical reasoning.
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import PARSER_MODEL, MODEL_CACHE_DIR


class ResponseParser:
    """Parses plaintext biomedical reasoning into structured JSON using Mistral."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name or PARSER_MODEL
        self.cache_dir = cache_dir or MODEL_CACHE_DIR

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load parser model."""
        print(f"[PARSER] Loading {self.model_name}...")
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

        print(f"[PARSER] Model loaded on {self.device}")

    def parse_interaction(
        self,
        plaintext: str,
        agent_name: str,
        pathway_name: str,
    ) -> list[dict]:
        """
        Parse MediPhi's plaintext reasoning into structured interaction JSON.

        Returns list of interaction dicts, or empty list if no valid interactions.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Quick check: if analysis explicitly says NO, skip parsing
        plaintext_lower = plaintext.lower()
        if any(phrase in plaintext_lower for phrase in [
            "no valid interaction",
            "conclusion: no",
            "does not directly target",
            "lacks phase iii",
            "no phase iii data",
            "no completed phase iii",
        ]):
            print(f"[PARSER] Analysis indicates NO interaction, skipping extraction")
            return []

        # Mistral-style instruction format
        prompt = f"""[INST] You are a JSON extraction assistant. Extract structured data from biomedical analysis.

ANALYSIS:
{plaintext}

TASK: Extract valid agent-pathway-cancer interactions ONLY if the analysis concludes YES.

IMPORTANT: The pathway name must be "{pathway_name}" exactly. If the analysis discusses a different pathway (e.g., "PD-1 pathway" when asked about "Tumor Antigen"), return [].

For each valid interaction, extract this exact JSON structure:
- agentName: "{agent_name}" (must be exact)
- pathwayName: "{pathway_name}" (must be exact)
- agentEffect: "inhibits", "activates", or "modulates"
- primaryTarget: the molecular target (protein/gene name)
- cancerType: the specific cancer type
- targetStatus: "overexpressed", "overactive", "present", "mutated", or "lost"
- mechanismType: must be "direct"

RULES:
- If analysis says "NO VALID INTERACTION" or "NO" in conclusion, return: []
- If analysis mentions Phase III data is lacking, return: []
- If analysis discusses wrong pathway name, return: []
- If mechanism is "downstream" or "indirect", return: []
- Maximum 3 cancer types per agent-pathway pair
- Return ONLY valid JSON array, nothing else

Output (JSON only): [/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        print(f"[PARSER] Extracting JSON from {len(plaintext)} chars...")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        print(f"[PARSER] Raw output length: {len(response)} chars")

        result = self._extract_json(response, agent_name, pathway_name)
        print(f"[PARSER] Extracted {len(result)} valid interactions")

        return result

    def _extract_json(
        self,
        response: str,
        agent_name: str,
        pathway_name: str,
    ) -> list[dict]:
        """Extract and validate JSON from response."""
        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(response.strip())
            if isinstance(data, list):
                return self._validate_interactions(data, agent_name, pathway_name)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, list):
                        return self._validate_interactions(data, agent_name, pathway_name)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Find array brackets (most common for Mistral)
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(response[start:end])
                if isinstance(data, list):
                    return self._validate_interactions(data, agent_name, pathway_name)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try single object and wrap in array
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            try:
                obj = json.loads(response[start:end])
                if isinstance(obj, dict):
                    return self._validate_interactions([obj], agent_name, pathway_name)
            except json.JSONDecodeError:
                pass

        print(f"[PARSER] Failed to extract valid JSON from response")
        return []

    def _validate_interactions(
        self,
        data: list,
        agent_name: str,
        pathway_name: str,
    ) -> list[dict]:
        """Validate and clean interaction data."""
        valid = []
        required_fields = [
            "agentName", "pathwayName", "agentEffect",
            "primaryTarget", "cancerType", "targetStatus", "mechanismType"
        ]

        for item in data:
            if not isinstance(item, dict):
                continue

            # Check required fields
            if not all(field in item for field in required_fields):
                print(f"[PARSER] Skipping item with missing fields: {list(item.keys())}")
                continue

            # Enforce exact name matching
            item["agentName"] = agent_name
            item["pathwayName"] = pathway_name

            # Only accept direct mechanisms
            if item.get("mechanismType") != "direct":
                print(f"[PARSER] Skipping non-direct mechanism: {item.get('mechanismType')}")
                continue

            # Validate enum values
            if item["agentEffect"] not in ["inhibits", "activates", "modulates"]:
                print(f"[PARSER] Invalid agentEffect: {item['agentEffect']}")
                continue

            # Handle compound targetStatus (e.g., "overexpressed and mutated")
            # Extract first valid status if compound
            target_status = item["targetStatus"].lower()
            valid_statuses = ["overexpressed", "overactive", "present", "mutated", "lost"]

            # Check if it's already a valid single status
            if target_status not in valid_statuses:
                # Try to extract first valid status from compound
                found_status = None
                for status in valid_statuses:
                    if status in target_status:
                        found_status = status
                        break

                if found_status:
                    item["targetStatus"] = found_status
                    print(f"[PARSER] Normalized compound status '{target_status}' to '{found_status}'")
                else:
                    print(f"[PARSER] Invalid targetStatus: {item['targetStatus']}")
                    continue

            valid.append(item)

        # Max 3 interactions
        return valid[:3]
