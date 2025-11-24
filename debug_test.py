"""Quick debug test for first positive control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MEDIPHI_MODEL, MODEL_CACHE_DIR
from src.data.loader import Agent, Pathway
from src.models.mediphi import MediPhiModel
from src.models.parser import ResponseParser
from src.stages.generate import generate_interaction_with_reasoning

print("Loading MediPhi...")
model = MediPhiModel(model_name=MEDIPHI_MODEL, cache_dir=MODEL_CACHE_DIR)
model.load()
print("MediPhi loaded.\n")

print("Loading parser...")
parser = ResponseParser()
parser.load()
print("Parser loaded.\n")

# First positive control
agent = Agent(name="Trastuzumab", category="immunotherapy")
pathway = Pathway(name="ERBB2 Signaling")

print(f"Testing: {agent.name} + {pathway.name}")
print("=" * 60)

interactions, reasoning = generate_interaction_with_reasoning(agent, pathway, model, parser)

print(f"\n\nFinal reasoning length: {len(reasoning)} chars")
print(f"Number of interactions: {len(interactions)}")
print(f"\nFirst 500 chars of reasoning:\n{reasoning[:500]}")
