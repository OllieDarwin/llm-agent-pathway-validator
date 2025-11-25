"""Parser model for extracting structured JSON from plaintext LLM output.

Uses Mistral-7B-Instruct for reliable JSON extraction from biomedical reasoning.
"""

import json
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import PARSER_MODEL, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


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

        # Quick check: if analysis explicitly says NO, skip parsing (NEW: added Phase I check)
        plaintext_lower = plaintext.lower()
        if any(phrase in plaintext_lower for phrase in [
            "no valid interaction",
            "conclusion: no",
            "does not directly target",
            "lacks phase iii",
            "no phase iii data",
            "no completed phase iii",
            "no phase i data",  # NEW: natural compounds
            "lacks phase i",  # NEW: natural compounds
            "insufficient evidence",
            "merely present",  # NEW: dysregulation requirement
            "not dysregulated",  # NEW: dysregulation requirement
        ]):
            logger.info("Analysis indicates NO interaction, skipping extraction")
            return []

        # Mistral-style instruction format
        prompt = f"""
You are a **biomedical JSON-extraction assistant**.
Your job is to extract structured agent–pathway–cancer interaction data from the following analysis.

**ANALYSIS:**
{plaintext}

---

## **TASK**

Extract interactions **ONLY if the analysis concludes a valid, direct interaction** between the agent and the exact pathway **"{pathway_name}"**.

If the analysis concludes **NO**, **NO VALID INTERACTION**, or indicates:

* wrong pathway
* missing Phase I data (natural compounds) OR Phase III/FDA (other drugs)  **NEW**
* target merely "present" (not dysregulated)  **NEW**
* indirect/downstream mechanism
* regulatory relationships or pathway crosstalk  **NEW**
  then you must output:

```
[]
```

---

## **STRICT PATHWAY MATCHING**

The pathway must match **"{pathway_name}"** exactly.
If the analysis references a different pathway name (even similar), return `[]`.

---

## **NEW: DYSREGULATION REQUIREMENT**

**CRITICAL**: Only extract if target is DYSREGULATED:
* overexpressed, OR
* overactive, OR
* mutated, OR
* lost

**DO NOT extract if target is only "present"** - this violates dysregulation requirement.

---

## **OUTPUT FORMAT**

Return ONLY a **JSON array** where each element adheres to the **exact object shape** below:

```json
{{
    "agentName": "",
    "pathwayName": "",
    "agentEffect": "",
    "primaryTarget": "",
    "cancerType": "",
    "targetStatus": "",
    "mechanismType": ""
}}
```

### **Requirements**

* `agentName` must be exactly **"{agent_name}"**
* `pathwayName` must be exactly **"{pathway_name}"**
* `agentEffect` ∈ **["inhibits", "activates", "modulates"]**
* `targetStatus` ∈ **["overexpressed", "overactive", "mutated", "lost"]**  **NEW: removed "present"**
* `mechanismType` must be **"direct"**
* Maximum **3 cancer types**
* If any required field is missing → output `[]`
* If target is only "present" → output `[]`  **NEW: dysregulation requirement**
* Output **JSON only**, no explanations

---

## **FINAL INSTRUCTION**

**Return JSON only. No text before or after.**
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        logger.info(f"Extracting JSON from {len(plaintext)} chars of reasoning")

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

        logger.debug(f"Raw parser output length: {len(response)} chars")

        result = self._extract_json(response, agent_name, pathway_name)
        logger.info(f"Extracted {len(result)} valid interactions")

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

        logger.warning("Failed to extract valid JSON from parser response")
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
                logger.debug(f"Skipping item with missing fields: {list(item.keys())}")
                continue

            # Enforce exact name matching
            item["agentName"] = agent_name
            item["pathwayName"] = pathway_name

            # Only accept direct mechanisms
            if item.get("mechanismType") != "direct":
                logger.debug(f"Skipping non-direct mechanism: {item.get('mechanismType')}")
                continue

            # Validate enum values
            if item["agentEffect"] not in ["inhibits", "activates", "modulates"]:
                logger.debug(f"Invalid agentEffect: {item['agentEffect']}")
                continue

            # Handle compound targetStatus (e.g., "overexpressed and mutated")
            # Extract first valid status if compound
            # NEW: "present" is no longer valid - must be dysregulated
            target_status = item["targetStatus"].lower()
            valid_statuses = ["overexpressed", "overactive", "mutated", "lost"]  # NEW: removed "present"

            # NEW: Explicitly reject "present" status
            if target_status == "present":
                logger.info("Rejecting targetStatus 'present' - dysregulation required")
                continue

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
                    logger.debug(f"Normalized compound status '{target_status}' to '{found_status}'")
                else:
                    logger.debug(f"Invalid targetStatus: {item['targetStatus']}")
                    continue

            valid.append(item)

        # Max 3 interactions
        return valid[:3]
