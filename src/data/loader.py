"""Data loader for agents and pathways CSV files."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Agent:
    name: str
    category: str  # chemotherapy, immunotherapy, natural_agent


@dataclass
class Pathway:
    name: str


def load_agents(filepath: Path | str) -> list[Agent]:
    """Load agents from CSV file."""
    agents = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            agents.append(Agent(name=row["name"], category=row["category"]))
    return agents


def load_pathways(filepath: Path | str) -> list[Pathway]:
    """Load pathways from CSV file."""
    pathways = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip internal reference gene
            if row["name"] != "Internal Reference Gene":
                pathways.append(Pathway(name=row["name"]))
    return pathways


def generate_combinations(agents: list[Agent], pathways: list[Pathway]) -> list[tuple[Agent, Pathway]]:
    """Generate all agent-pathway combinations for validation."""
    return [(agent, pathway) for agent in agents for pathway in pathways]
