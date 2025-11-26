"""Stage 1: generateInteraction() - Initial biological filter.

SIMPLE TWO-STEP ARCHITECTURE:
1. Reasoning model generates plaintext biomedical reasoning
2. Parser extracts structured JSON
3. Filter to keep only hasInteraction=True responses
"""

import logging
from dataclasses import dataclass

from data.loader import Agent, Pathway
from models.reasoning import ReasoningModel
from models.parser import Parser
from schemas.stage1 import AgentEffect, TargetStatus, MechanismType, InteractionSchema
from prompts.stage1 import REASONING_PROMPT, PARSING_PROMPT

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
    model: ReasoningModel,
    parser: Parser,
) -> list[Interaction]:
    """Generate validated agent-pathway-cancer interactions.

    Simple flow:
    1. Reasoning model analyzes agent + pathway
    2. Parser extracts JSON
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

    # Log reasoning for debugging
    logger.debug(f"Reasoning output ({len(plaintext)} chars):\n{plaintext[:500]}...")

    # Step 2: Parse to JSON and filter
    context = {"agent_name": agent.name, "pathway_name": pathway.name}

    # Stage-specific instruction to enforce exact name matching
    stage_instructions = f"""CRITICAL: The agentName field MUST be exactly "{agent.name}" and pathwayName MUST be exactly "{pathway.name}".
If the analysis mentions other agents, ignore them. Only extract data about {agent.name}."""

    all_results = parser.parse(
        text=plaintext,
        schema=InteractionSchema,
        context=context,
        prompt_template=PARSING_PROMPT,
        stage_instructions=stage_instructions,
    )

    # Filter for hasInteraction=True
    interaction_dicts = [r for r in all_results if r.get("hasInteraction", False)]

    # Step 3: Convert to Interaction objects and validate agent name
    interactions = []
    for item in interaction_dicts:
        # CRITICAL: Reject if agent name doesn't match (model hallucination)
        if item.get("agentName") != agent.name:
            logger.warning(
                f"✗ REJECTED: Parser returned wrong agent '{item.get('agentName')}' "
                f"instead of '{agent.name}' - model hallucination detected"
            )
            continue

        # Also validate pathway name
        if item.get("pathwayName") != pathway.name:
            logger.warning(
                f"✗ REJECTED: Parser returned wrong pathway '{item.get('pathwayName')}' "
                f"instead of '{pathway.name}'"
            )
            continue

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
    model: ReasoningModel,
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

    stage_instructions = f"""CRITICAL: The agentName field MUST be exactly "{agent.name}" and pathwayName MUST be exactly "{pathway.name}".
If the analysis mentions other agents, ignore them. Only extract data about {agent.name}."""

    all_results = parser.parse(
        text=plaintext,
        schema=InteractionSchema,
        context=context,
        prompt_template=PARSING_PROMPT,
        stage_instructions=stage_instructions,
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
    model: ReasoningModel,
    parser: Parser,
    progress_callback: callable = None,
) -> dict[tuple[str, str], list[Interaction]]:
    """Process multiple agent-pathway combinations.

    Args:
        combinations: List of (Agent, Pathway) tuples
        model: Loaded ReasoningModel
        parser: Loaded Parser
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
