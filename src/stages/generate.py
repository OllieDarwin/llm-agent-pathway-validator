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
MEDIPHI_PROMPT = """
You are a **clinical oncology pharmacist** analyzing **therapeutic agent–pathway interactions** with strict criteria.

Input variables:
**AGENT:** {agent_name}
**CATEGORY:** {agent_category}
**PATHWAY:** {pathway_name}

---

## **TASK**

Determine whether **this exact agent** directly targets **this exact pathway**.
The pathway name must match **verbatim**. No substitutions, parent pathways, sub-pathways, or related signaling families are allowed.

Provide analysis using ONLY the following labelled sections. Keep all sections concise.

---

## **1. MOLECULAR TARGET**

State the **primary molecular target** (specific protein/gene/receptor) of {agent_name}.
*One sentence only.*

---

## **2. PATHWAY COMPONENTS**

List **3–5 core components** (proteins/genes) of the pathway **named exactly** "{pathway_name}".
If pathway name is ambiguous, incomplete, or non-standard → mark as “Not a valid pathway name”.

---

## **3. DIRECT INTERACTION CHECK**

Answer **YES/NO** for both:

* Whether the target from Section 1 is within the components from Section 2.
* Whether the agent binds/inhibits a component with the **exact same name** as a pathway element.

If NO for either → stop further justification and proceed to Conclusion.

---

## **4. CLINICAL EVIDENCE**

Rules:

* **Natural compounds** (e.g., Curcumin, Resveratrol, EGCG, Green Tea, Turmeric): must have **published Phase III clinical trial results**. If not → “NO”.
* **FDA-approved oncology drugs:** verify approval status.
  State ONLY: “YES — valid evidence” or “NO — insufficient evidence”.

---

## **5. PRIMARY MECHANISM**

State whether the pathway "{pathway_name}" is the **primary mechanism**, or only **downstream/secondary**.

---

## **6. CONCLUSION**

If **all** criteria are met, provide:

* up to **3 cancer types**,
* each with **agentEffect** (“inhibits”, “activates”, or “modulates”),
* the agent’s **primaryTarget**,
* the **targetStatus** (“overexpressed”, “overactive”, “present”, “mutated”, or “lost”),
* mechanismType = **“direct”**.

If **any** criterion fails → output:
**“NO VALID INTERACTION — {brief reason}”**
(Use reasons such as: no Phase III data, target not in pathway, pathway mismatch, indirect mechanism.)

---

### **STRICT RULES**

* Natural compounds almost always → **NO** (no Phase III).
* Pathway name must match **exactly**.
* Only **direct binding** qualifies.
* Keep outputs short.
* Do not repeat information across sections.

---

**Begin analysis:**
"""


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
