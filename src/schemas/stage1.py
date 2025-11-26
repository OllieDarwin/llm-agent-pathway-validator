"""Pydantic schemas for Stage 1: generateInteraction().

Defines the structure for agent-pathway-cancer interactions.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class AgentEffect(str, Enum):
    """How the agent affects the pathway."""
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    MODULATES = "modulates"


class TargetStatus(str, Enum):
    """Target dysregulation status."""
    OVEREXPRESSED = "overexpressed"
    OVERACTIVE = "overactive"
    MUTATED = "mutated"
    LOST = "lost"


class MechanismType(str, Enum):
    """Type of molecular mechanism."""
    DIRECT = "direct"
    DOWNSTREAM = "downstream"


class InteractionSchema(BaseModel):
    """Response from Stage 1 analysis.

    Can be either:
    - hasInteraction=True with interaction details
    - hasInteraction=False with empty interaction fields
    """

    agentName: str = Field(..., description="Name of the therapeutic agent")
    pathwayName: str = Field(..., description="Name of the biological pathway")
    hasInteraction: bool = Field(..., description="Whether a valid interaction exists")

    # Optional fields - only required if hasInteraction=True
    agentEffect: Optional[AgentEffect] = Field(None, description="How the agent affects the pathway")
    primaryTarget: Optional[str] = Field(None, description="Primary molecular target of the agent")
    cancerType: Optional[str] = Field(None, description="Specific cancer type for this interaction")
    targetStatus: Optional[TargetStatus] = Field(None, description="Dysregulation status of the target")
    mechanismType: Optional[MechanismType] = Field(None, description="Type of molecular mechanism")

    class Config:
        use_enum_values = True
