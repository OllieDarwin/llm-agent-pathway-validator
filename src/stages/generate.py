"""Stage 1: generateInteraction() - Initial biological filter.

SIMPLE TWO-STEP ARCHITECTURE:
1. MediPhi generates plaintext biomedical reasoning
2. Mistral parser extracts structured JSON
3. Filter to keep only hasInteraction=True responses
"""

import logging
from dataclasses import dataclass

from src.data.loader import Agent, Pathway
from src.models.mediphi import MediPhiModel
from src.models.parser import Parser
from src.schemas.stage1 import AgentEffect, TargetStatus, MechanismType, InteractionSchema
from src.prompts.stage1 import REASONING_PROMPT, PARSING_PROMPT

logger = logging.getLogger(__name__)


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
        """Create Interaction from parser dict (assumes hasInteraction=True)."""
        return cls(
            agent_name=data["agentName"],
            pathway_name=data["pathwayName"],
            agent_effect=AgentEffect(data["agentEffect"]),
            primary_target=data["primaryTarget"],
            cancer_type=data["cancerType"],
            target_status=TargetStatus(data["targetStatus"]),
            mechanism_type=MechanismType(data["mechanismType"]),
        )


def generate_interaction(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: Parser,
) -> list[Interaction]:
    """Generate validated agent-pathway-cancer interactions.

    Simple flow:
    1. MediPhi analyzes agent + pathway
    2. Mistral parses reasoning to JSON
    3. Filter for hasInteraction=True
    4. Convert to Interaction objects

    Returns:
        List of Interaction objects (empty if no valid interaction).
    """
    # Step 1: Generate plaintext reasoning
    prompt = REASONING_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    logger.info(f"Analyzing: {agent.name} + {pathway.name}")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)

    # Step 2: Parse to JSON and filter
    context = {"agent_name": agent.name, "pathway_name": pathway.name}
    all_results = parser.parse(
        text=plaintext,
        schema=InteractionSchema,
        context=context,
        prompt_template=PARSING_PROMPT,
    )

    # Filter for hasInteraction=True
    interaction_dicts = [r for r in all_results if r.get("hasInteraction", False)]

    # Step 3: Convert to Interaction objects
    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            logger.info(
                f"✓ Interaction found: {interaction.cancer_type} "
                f"({interaction.agent_effect.value} {interaction.primary_target})"
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to create Interaction from {item}: {e}")
            continue

    if not interactions:
        logger.info(f"✗ No interaction: {agent.name} + {pathway.name}")

    return interactions


def generate_interaction_with_reasoning(
    agent: Agent,
    pathway: Pathway,
    model: MediPhiModel,
    parser: Parser,
) -> tuple[list[Interaction], str]:
    """Same as generate_interaction but also returns plaintext reasoning.

    Useful for debugging and logging.
    """
    prompt = REASONING_PROMPT.format(
        agent_name=agent.name,
        agent_category=agent.category,
        pathway_name=pathway.name,
    )

    logger.info(f"Analyzing: {agent.name} + {pathway.name}")
    plaintext = model.generate(prompt, max_new_tokens=512, temperature=0.3)

    context = {"agent_name": agent.name, "pathway_name": pathway.name}
    all_results = parser.parse(
        text=plaintext,
        schema=InteractionSchema,
        context=context,
        prompt_template=PARSING_PROMPT,
    )

    interaction_dicts = [r for r in all_results if r.get("hasInteraction", False)]

    interactions = []
    for item in interaction_dicts:
        try:
            interaction = Interaction.from_dict(item)
            interactions.append(interaction)
            logger.info(
                f"✓ Interaction found: {interaction.cancer_type} "
                f"({interaction.agent_effect.value} {interaction.primary_target})"
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to create Interaction from {item}: {e}")
            continue

    if not interactions:
        logger.info(f"✗ No interaction: {agent.name} + {pathway.name}")

    return interactions, plaintext


def generate_interactions_batch(
    combinations: list[tuple[Agent, Pathway]],
    model: MediPhiModel,
    parser: Parser,
    progress_callback: callable = None,
) -> dict[tuple[str, str], list[Interaction]]:
    """Process multiple agent-pathway combinations.

    Args:
        combinations: List of (Agent, Pathway) tuples
        model: Loaded MediPhiModel
        parser: Loaded GenericParser
        progress_callback: Optional callback(current, total, agent_name, pathway_name)

    Returns:
        Dict mapping (agent_name, pathway_name) to list of Interactions.
    """
    results = {}

    for i, (agent, pathway) in enumerate(combinations):
        key = (agent.name, pathway.name)
        results[key] = generate_interaction(agent, pathway, model, parser)

        if progress_callback:
            progress_callback(i + 1, len(combinations), agent.name, pathway.name)

    return results
