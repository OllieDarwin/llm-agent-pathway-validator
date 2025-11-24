"""Parser model for extracting structured JSON from plaintext LLM output."""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ResponseParser:
    """Parses plaintext biomedical reasoning into structured JSON."""

    def __init__(
        self,
        model_name: str = "osmosis-ai/Osmosis-Structure-0.6B",
        cache_dir: str | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Load parser model."""
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

        prompt = f"""<s>[INST] Extract structured data from this biomedical analysis.

ANALYSIS:
{plaintext}

TASK: Extract any valid agent-pathway-cancer interactions mentioned.
Agent name must be exactly: {agent_name}
Pathway name must be exactly: {pathway_name}

For each valid interaction found, extract:
- agentEffect: "inhibits", "activates", or "modulates"
- primaryTarget: the molecular target (protein/gene)
- cancerType: the cancer type
- targetStatus: "overexpressed", "overactive", "present", "mutated", or "lost"

RULES:
- Only extract interactions explicitly supported in the analysis
- If the analysis concludes NO valid interaction exists, return empty array
- If mechanism is "downstream" or "indirect", do NOT include it
- Maximum 3 cancer types

Return ONLY a valid JSON array, nothing else:
[{{"agentName": "...", "pathwayName": "...", "agentEffect": "...", "primaryTarget": "...", "cancerType": "...", "targetStatus": "...", "mechanismType": "direct"}}]

Or if no valid interactions: []
[/INST]"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return self._extract_json(response, agent_name, pathway_name)

    def _extract_json(
        self,
        response: str,
        agent_name: str,
        pathway_name: str,
    ) -> list[dict]:
        """Extract and validate JSON from response."""
        # Try to find JSON array in response
        try:
            # Direct parse
            data = json.loads(response)
            if isinstance(data, list):
                return self._validate_interactions(data, agent_name, pathway_name)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if "```" in response:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if isinstance(data, list):
                        return self._validate_interactions(data, agent_name, pathway_name)
                except json.JSONDecodeError:
                    pass

        # Try to find array brackets
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(response[start:end])
                if isinstance(data, list):
                    return self._validate_interactions(data, agent_name, pathway_name)
            except json.JSONDecodeError:
                pass

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
                continue

            # Enforce exact name matching
            item["agentName"] = agent_name
            item["pathwayName"] = pathway_name

            # Only accept direct mechanisms
            if item.get("mechanismType") != "direct":
                continue

            # Validate enum values
            if item["agentEffect"] not in ["inhibits", "activates", "modulates"]:
                continue
            if item["targetStatus"] not in ["overexpressed", "overactive", "present", "mutated", "lost"]:
                continue

            valid.append(item)

        # Max 3 interactions
        return valid[:3]


class LightweightParser:
    """
    Regex-based parser for simpler/faster extraction.
    Use when parser model is unavailable or for quick validation.
    """

    # Patterns to detect rejection
    REJECTION_PATTERNS = [
        r"no\s+(?:valid|direct)\s+interaction",
        r"empty\s+array",
        r"return(?:s|ing)?\s*\[\s*\]",
        r"does\s+not\s+(?:directly\s+)?(?:target|interact|affect)",
        r"not\s+a\s+(?:core\s+)?component",
        r"no\s+fda\s+approval",
        r"no\s+phase\s+(?:iii|3)",
        r"indirect\s+(?:mechanism|effect)",
        r"downstream\s+effect",
    ]

    # Patterns to extract cancer types
    CANCER_PATTERNS = [
        r"fda[- ]approved\s+(?:for|in)\s+([A-Za-z\s]+(?:cancer|carcinoma|leukemia|lymphoma|melanoma|myeloma))",
        r"(?:indicated|approved)\s+for\s+(?:treating\s+)?([A-Za-z\s]+(?:cancer|carcinoma|leukemia|lymphoma|melanoma|myeloma))",
        r"(?:cml|aml|all|nsclc|sclc|hcc|rcc|crc)",
    ]

    @classmethod
    def is_rejection(cls, text: str) -> bool:
        """Check if the text indicates no valid interaction."""
        text_lower = text.lower()
        for pattern in cls.REJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    @classmethod
    def extract_cancer_types(cls, text: str) -> list[str]:
        """Extract mentioned cancer types from text."""
        cancers = []
        text_lower = text.lower()

        for pattern in cls.CANCER_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            cancers.extend(matches)

        # Deduplicate and clean
        seen = set()
        clean = []
        for c in cancers:
            c_clean = c.strip().title()
            if c_clean not in seen:
                seen.add(c_clean)
                clean.append(c_clean)

        return clean[:3]
