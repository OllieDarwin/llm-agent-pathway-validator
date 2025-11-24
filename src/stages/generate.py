"""Stage 1: generateInteraction() - Initial biological filter."""

from dataclasses import dataclass
from enum import Enum

from src.data.loader import Agent, Pathway
from src.models.mediphi import MediPhiModel
from src.models.parser import ResponseParser


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


# Plaintext prompt for MediPhi - optimized for natural language generation
MEDIPHI_PROMPT = """You are a clinical oncology pharmacist. Analyze whether this therapeutic agent directly targets the specified biological pathway.

Agent: {agent_name}
Agent Category: {agent_category}
Pathway: {pathway_name}

Provide your analysis in the following structure:

1. PRIMARY MOLECULAR TARGET: What is {agent_name}'s primary molecular target?

2. PATHWAY COMPONENTS: What are the core components of {pathway_name}?

3. DIRECT INTERACTION CHECK: Is the agent's primary target a core component of this pathway? (YES/NO)

4. CLINICAL EVIDENCE: Does {agent_name} have FDA approval or positive Phase III trial data for any cancer type where this pathway is the PRIMARY mechanism?

5. CONCLUSION: Based on the above analysis, state whether there is a valid direct interaction.
   - If YES: List the cancer type(s) (max 3), the agent's effect (inhibits/activates/modulates), and the target status in the cancer.
   - If NO: Explain why (e.g., "No direct interaction - target is not a pathway component" or "Indirect/downstream mechanism" or "No Phase III clinical evidence").

STRICT RULES:
- Natural compounds (Curcumin, Resveratrol, EGCG, etc.) require Phase III data - most should have NO valid interaction
- Only include DIRECT mechanisms where the agent binds/targets a pathway component
- Exclude downstream effects, pathway crosstalk, and indirect associations
- When uncertain, conclude NO valid interaction

Begin your analysis:"""


def generate_interaction(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: ResponseParser | None = None,
) -> list[Interaction]:
    """
    Stage 1: Generate validated agent-pathway-cancer interactions.

    MediPhi generates plaintext reasoning, then parser extracts structured JSON.

    Returns empty list if no valid interaction exists.
    Returns 1-3 Interaction objects for valid combinations.
    """
    # Generate plaintext analysis from MediPhi
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Calling MediPhi model (max_tokens=512)...")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)
    print(f"[STAGE1] MediPhi returned {len(plaintext)} chars")

    # Parse plaintext into structured JSON
    if parser is not None:
        interaction_dicts = parser.parse_interaction(
            plaintext=plaintext,
            agent_name=agent.name,
            pathway_name=pathway.name,
        )
    else:
        # Fallback: no parser available, return empty
        # In production, parser should always be provided
        return []

    # Convert to Interaction objects
    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
        except (KeyError, ValueError):
            continue

    return interactions[:3]


def generate_interaction_with_reasoning(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: ResponseParser | None = None,
) -> tuple[list[Interaction], str]:
    """
    Same as generate_interaction but also returns the raw reasoning.

    Useful for debugging and logging.
    """
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Calling MediPhi model (max_tokens=512)...")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)
    print(f"[STAGE1] MediPhi returned {len(plaintext)} chars")

    if parser is not None:
        interaction_dicts = parser.parse_interaction(
            plaintext=plaintext,
            agent_name=agent.name,
            pathway_name=pathway.name,
        )
    else:
        return [], plaintext

    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
        except (KeyError, ValueError):
            continue

    return interactions[:3], plaintext


def generate_interactions_batch(
    combinations: list[tuple[Agent, Pathway]],
    model: MediPhiModel,
    parser: ResponseParser,
    progress_callback: callable = None,
) -> dict[tuple[str, str], list[Interaction]]:
    """
    Process multiple agent-pathway combinations.

    Returns dict mapping (agent_name, pathway_name) to list of interactions.
    """
    results = {}

    for i, (agent, pathway) in enumerate(combinations):
        key = (agent.name, pathway.name)
        results[key] = generate_interaction(agent, pathway, model, parser)

        if progress_callback:
            progress_callback(i + 1, len(combinations), agent.name, pathway.name)

    return results
