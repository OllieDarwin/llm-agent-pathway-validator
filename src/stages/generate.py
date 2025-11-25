"""Stage 1: generateInteraction() - Initial biological filter.

SIMPLIFIED ARCHITECTURE:
- Single MediPhi model call that outputs JSON directly
- No separate parser model needed
- Clear YES/NO decision structure
- Robust JSON extraction with fallback parsing
"""

import json
import re
from dataclasses import dataclass
from enum import Enum

from src.data.loader import Agent, Pathway
from src.models.mediphi import MediPhiModel


class AgentEffect(str, Enum):
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    MODULATES = "modulates"


class TargetStatus(str, Enum):
    OVEREXPRESSED = "overexpressed"
    OVERACTIVE = "overactive"
    PRESENT = "present"
    MUTATED = "mutated"
    LOST = "lost"


class MechanismType(str, Enum):
    DIRECT = "direct"
    DOWNSTREAM = "downstream"


@dataclass
class Interaction:
    """A validated agent-pathway-cancer interaction."""
    agent_name: str
    pathway_name: str
    agent_effect: AgentEffect
    primary_target: str
    cancer_type: str
    target_status: TargetStatus
    mechanism_type: MechanismType

    @classmethod
    def from_dict(cls, data: dict) -> "Interaction":
        return cls(
            agent_name=data["agentName"],
            pathway_name=data["pathwayName"],
            agent_effect=AgentEffect(data["agentEffect"]),
            primary_target=data["primaryTarget"],
            cancer_type=data["cancerType"],
            target_status=TargetStatus(data["targetStatus"]),
            mechanism_type=MechanismType(data["mechanismType"]),
        )


# Simplified JSON-focused prompt for MediPhi
MEDIPHI_PROMPT = """You are a clinical oncology specialist evaluating therapeutic agent-pathway interactions.

AGENT: {agent_name}
CATEGORY: {agent_category}
PATHWAY: {pathway_name}

TASK: Determine if this agent DIRECTLY targets a core component of this pathway.

EVALUATION CRITERIA:
1. Is the agent's PRIMARY molecular target a core component of this pathway?
2. Does the agent have FDA approval OR positive Phase III data?
3. Is this pathway the agent's PRIMARY mechanism (not downstream/indirect)?

STRICT EXCLUSIONS:
- Natural compounds without Phase III data (Curcumin, Resveratrol, etc.) → return []
- Downstream effects (e.g., PD-1 affecting tumor antigens) → return []
- Indirect mechanisms or pathway crosstalk → return []
- Secondary/off-label mechanisms → return []

OUTPUT FORMAT:
Return ONLY a JSON array. If NO valid interaction, return empty array [].
If valid interaction exists, return 1-3 cancer types with this structure:

[
  {{
    "agentName": "{agent_name}",
    "pathwayName": "{pathway_name}",
    "agentEffect": "inhibits|activates|modulates",
    "primaryTarget": "target protein/gene name",
    "cancerType": "specific cancer type",
    "targetStatus": "overexpressed|overactive|mutated|present|lost",
    "mechanismType": "direct"
  }}
]

Return JSON only, no explanation:"""


def _extract_json_from_response(response: str) -> list[dict]:
    """
    Extract JSON array from model response with multiple fallback strategies.

    Returns empty list if no valid JSON found.
    """
    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(response.strip())
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    if "```" in response:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find array brackets
    start = response.find("[")
    end = response.rfind("]") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(response[start:end])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 4: Look for single object and wrap in array
    start = response.find("{")
    end = response.rfind("}") + 1
    if start != -1 and end > start:
        try:
            obj = json.loads(response[start:end])
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    return []


def _validate_interaction_dict(data: dict, agent_name: str, pathway_name: str) -> bool:
    """Validate that interaction dict has required fields and valid values."""
    required_fields = [
        "agentName", "pathwayName", "agentEffect",
        "primaryTarget", "cancerType", "targetStatus", "mechanismType"
    ]

    # Check all required fields present
    if not all(field in data for field in required_fields):
        return False

    # Validate mechanism type is direct
    if data.get("mechanismType") != "direct":
        return False

    # Validate enum values
    if data["agentEffect"] not in ["inhibits", "activates", "modulates"]:
        return False
    if data["targetStatus"] not in ["overexpressed", "overactive", "present", "mutated", "lost"]:
        return False

    # Ensure names match
    data["agentName"] = agent_name
    data["pathwayName"] = pathway_name

    return True


def generate_interaction(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser=None,  # Kept for backward compatibility but unused
) -> list[Interaction]:
    """
    Stage 1: Generate validated agent-pathway-cancer interactions.

    SIMPLIFIED: Single MediPhi call outputs JSON directly, no separate parser.

    Returns empty list if no valid interaction exists.
    Returns 1-3 Interaction objects for valid combinations.
    """
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Generating interaction for {agent.name} + {pathway.name}")

    # Generate JSON directly from MediPhi
    response = model.generate(prompt, max_new_tokens=400, temperature=0.1)

    print(f"[STAGE1] Response length: {len(response)} chars")

    # Extract JSON from response
    interaction_dicts = _extract_json_from_response(response)

    if not interaction_dicts:
        print(f"[STAGE1] No JSON found in response")
        return []

    print(f"[STAGE1] Extracted {len(interaction_dicts)} potential interactions")

    # Validate and convert to Interaction objects
    interactions = []
    for item in interaction_dicts:
        if not _validate_interaction_dict(item, agent.name, pathway.name):
            print(f"[STAGE1] Skipping invalid interaction: {item}")
            continue

        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            print(f"[STAGE1] ✓ Valid: {interaction.cancer_type} ({interaction.agent_effect.value} {interaction.primary_target})")
        except (KeyError, ValueError) as e:
            print(f"[STAGE1] Failed to parse interaction: {e}")
            continue

    return interactions[:3]


def generate_interaction_with_reasoning(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser=None,  # Kept for backward compatibility but unused
) -> tuple[list[Interaction], str]:
    """
    Same as generate_interaction but also returns the raw response.

    Useful for debugging and logging.
    """
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Generating interaction for {agent.name} + {pathway.name}")

    response = model.generate(prompt, max_new_tokens=400, temperature=0.1)

    print(f"[STAGE1] Response length: {len(response)} chars")

    interaction_dicts = _extract_json_from_response(response)

    if not interaction_dicts:
        print(f"[STAGE1] No JSON found in response")
        return [], response

    print(f"[STAGE1] Extracted {len(interaction_dicts)} potential interactions")

    interactions = []
    for item in interaction_dicts:
        if not _validate_interaction_dict(item, agent.name, pathway.name):
            print(f"[STAGE1] Skipping invalid interaction: {item}")
            continue

        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            print(f"[STAGE1] ✓ Valid: {interaction.cancer_type} ({interaction.agent_effect.value} {interaction.primary_target})")
        except (KeyError, ValueError) as e:
            print(f"[STAGE1] Failed to parse interaction: {e}")
            continue

    return interactions[:3], response


def generate_interactions_batch(
    combinations: list[tuple[Agent, Pathway]],
    model: MediPhiModel,
    parser=None,  # Kept for backward compatibility but unused
    progress_callback: callable = None,
) -> dict[tuple[str, str], list[Interaction]]:
    """
    Process multiple agent-pathway combinations.

    Returns dict mapping (agent_name, pathway_name) to list of interactions.
    """
    results = {}

    for i, (agent, pathway) in enumerate(combinations):
        key = (agent.name, pathway.name)
        results[key] = generate_interaction(agent, pathway, model)

        if progress_callback:
            progress_callback(i + 1, len(combinations), agent.name, pathway.name)

    return results
