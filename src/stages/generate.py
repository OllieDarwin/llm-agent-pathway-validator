"""Stage 1: generateInteraction() - Initial biological filter.

TWO-STEP ARCHITECTURE:
- Step 1: MediPhi generates plaintext biomedical reasoning
- Step 2: Mistral parser extracts structured JSON from reasoning
- Separation of concerns: domain knowledge vs. structured output
"""

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


# Plaintext reasoning prompt for MediPhi (Step 1)
MEDIPHI_PROMPT = """You are a clinical oncology pharmacist analyzing therapeutic agent-pathway interactions.

AGENT: {agent_name}
CATEGORY: {agent_category}
PATHWAY: {pathway_name}

TASK: Evaluate whether this agent DIRECTLY targets a core component of this pathway.

Provide a structured analysis covering:

1. MOLECULAR TARGET: What is the agent's primary molecular target?

2. PATHWAY COMPONENTS: What are the core components of this pathway?

3. DIRECT INTERACTION: Is the agent's target a core component of this pathway? (YES/NO)

4. CLINICAL EVIDENCE: Does the agent have FDA approval or positive Phase III data?

5. PRIMARY MECHANISM: Is this pathway the agent's PRIMARY mechanism of action?

6. CONCLUSION:
   - If YES: List up to 3 cancer types where this interaction is clinically relevant
   - For each: specify the agent effect (inhibits/activates/modulates), primary target, and target status (overexpressed/overactive/mutated/present/lost)
   - If NO: Explain why (e.g., no Phase III data, indirect mechanism, target not in pathway)

STRICT CRITERIA:
- Natural compounds require Phase III clinical trial data
- Only DIRECT mechanisms (exclude downstream effects)
- Target must be a CORE pathway component
- Must be PRIMARY mechanism, not secondary/off-label

Begin analysis:"""


def generate_interaction(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: ResponseParser,
) -> list[Interaction]:
    """
    Stage 1: Generate validated agent-pathway-cancer interactions.

    TWO-STEP PROCESS:
    1. MediPhi generates plaintext biomedical reasoning
    2. Mistral parser extracts structured JSON from reasoning

    Returns empty list if no valid interaction exists.
    Returns 1-3 Interaction objects for valid combinations.
    """
    # Step 1: Generate plaintext reasoning from MediPhi
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Step 1: MediPhi analyzing {agent.name} + {pathway.name}")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)
    print(f"[STAGE1] MediPhi generated {len(plaintext)} chars of reasoning")

    # Step 2: Parse plaintext into structured JSON
    print(f"[STAGE1] Step 2: Mistral parsing to JSON")
    interaction_dicts = parser.parse_interaction(
        plaintext=plaintext,
        agent_name=agent.name,
        pathway_name=pathway.name,
    )

    # Convert to Interaction objects
    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            print(f"[STAGE1] ✓ Valid: {interaction.cancer_type} ({interaction.agent_effect.value} {interaction.primary_target})")
        except (KeyError, ValueError) as e:
            print(f"[STAGE1] Failed to create Interaction: {e}")
            continue

    return interactions[:3]


def generate_interaction_with_reasoning(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: ResponseParser,
) -> tuple[list[Interaction], str]:
    """
    Same as generate_interaction but also returns the plaintext reasoning.

    Useful for debugging and logging.
    """
    # Step 1: Generate plaintext reasoning
    prompt = MEDIPHI_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    print(f"[STAGE1] Step 1: MediPhi analyzing {agent.name} + {pathway.name}")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)
    print(f"[STAGE1] MediPhi generated {len(plaintext)} chars of reasoning")

    # Step 2: Parse to JSON
    print(f"[STAGE1] Step 2: Mistral parsing to JSON")
    interaction_dicts = parser.parse_interaction(
        plaintext=plaintext,
        agent_name=agent.name,
        pathway_name=pathway.name,
    )

    # Convert to Interaction objects
    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            print(f"[STAGE1] ✓ Valid: {interaction.cancer_type} ({interaction.agent_effect.value} {interaction.primary_target})")
        except (KeyError, ValueError) as e:
            print(f"[STAGE1] Failed to create Interaction: {e}")
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
