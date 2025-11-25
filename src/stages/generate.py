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
    """NEW: Removed 'present' - target must be DYSREGULATED per logic-flow.md."""
    OVEREXPRESSED = "overexpressed"
    OVERACTIVE = "overactive"
    MUTATED = "mutated"
    LOST = "lost"
    # PRESENT = "present"  # REMOVED: NEW constraint requires dysregulation


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


# Plaintext reasoning prompt for MediPhi (Step 1) - Incorporates NEW constraints from logic-flow.md
MEDIPHI_PROMPT = """You are a clinical oncology pharmacist analyzing therapeutic agent-pathway interactions with STRICT criteria.

**AGENT:** {agent_name}
**CATEGORY:** {agent_category}
**PATHWAY:** {pathway_name}

## PRE-CHECK REQUIREMENT

Before proceeding, MUST identify:
1. {agent_name}'s PRIMARY molecular target
2. Whether that target is a CORE COMPONENT of "{pathway_name}"
3. If NO → immediately conclude "NO VALID INTERACTION"

## ANALYSIS SECTIONS

### 1. MOLECULAR TARGET
State {agent_name}'s PRIMARY molecular target (specific protein/gene/receptor). One sentence.

### 2. PATHWAY COMPONENTS
List 3-5 core components of pathway "{pathway_name}" (exact name).

### 3. DIRECT INTERACTION CHECK
YES/NO: Is target from Section 1 a component in Section 2?
If NO → skip to Conclusion.

### 4. CLINICAL EVIDENCE
**NEW RULES:**
- **Natural compounds** (Curcumin, Resveratrol, EGCG, Artemisinin, Quercetin, Ascorbic Acid, Melatonin, Alpha Lipoic Acid, Methylene Blue, NTC, Sodium Phenyl Butyrate, Semaglutide):
  * Require Phase I clinical trial data MINIMUM
  * Most lack this → return NO
- **FDA-approved drugs**: Verify FDA approval OR Phase III data
- **Chemotherapy**: Limit to 2-3 pathways, primary mechanism only
- **Immunotherapy**: Limit to 1 pathway, exclude downstream effects

State: "YES — valid evidence" or "NO — insufficient evidence"

### 5. PRIMARY MECHANISM
Is "{pathway_name}" the PRIMARY mechanism (not downstream/secondary)? YES/NO

### 6. DYSREGULATION REQUIREMENT (NEW)
**CRITICAL**: For each cancer type, the target must be:
- Overexpressed, OR
- Overactive, OR
- Mutated, OR
- Lost (for tumor suppressors)

**NOT merely "present"** - must be DYSREGULATED.

### 7. EXCLUSION CHECKS
Verify NONE of these apply:
- Regulatory relationships (e.g., PD-L1 affecting Tumor Antigen outcomes)
- Downstream effects (e.g., checkpoint inhibition enhancing antigen response)
- Pathway substitution (if discussing different pathway than "{pathway_name}")
- Semantic association without direct molecular interaction

## CONCLUSION

**If ALL criteria met:**
List up to 3 cancer types where:
- Target is DYSREGULATED (overexpressed/overactive/mutated/lost, NOT "present")
- Agent has required clinical evidence
- Mechanism is PRIMARY and DIRECT

For each: agentEffect (inhibits/activates/modulates), primaryTarget, targetStatus (must be overexpressed/overactive/mutated/lost)

**If ANY criterion fails:**
State: "NO VALID INTERACTION — [reason]"
Common reasons:
- No Phase I data (natural compounds)
- No Phase III/FDA approval (other drugs)
- Target not in pathway components
- Pathway name mismatch
- Downstream/indirect mechanism
- Target merely present, not dysregulated
- Exceeds pathway limit (chemo: 2-3, immuno: 1)

## SPECIAL RULES BY CATEGORY
- **Natural compounds**: Phase I minimum (most → NO)
- **Chemotherapy**: Max 2-3 pathways
- **Immunotherapy**: Max 1 pathway, no downstream

## EXAMPLES
**INCLUDE**: Imatinib + BCR-ABL signaling + CML (BCR-ABL overactive, FDA approved)
**EXCLUDE**: Pembrolizumab + Tumor Antigen (PD-1 not in Tumor Antigen components)
**EXCLUDE**: Curcumin + NF-κB (no Phase I data)
**EXCLUDE**: Target "present" only (must be dysregulated)

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
