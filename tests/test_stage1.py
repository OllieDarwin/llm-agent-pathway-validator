"""Test Stage 1 generateInteraction against known controls."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MEDIPHI_MODEL, PARSER_MODEL, MODEL_CACHE_DIR
from src.data.loader import Agent, Pathway
from src.models.mediphi import MediPhiModel
from src.models.parser import ResponseParser
from src.stages.generate import generate_interaction_with_reasoning


# Test cases - using pathways from all-tsnc-pathways.csv
POSITIVE_CONTROLS = [
    # Should return non-empty list (direct target in pathway)
    {
        "agent": Agent(name="Trastuzumab", category="immunotherapy"),
        "pathway": Pathway(name="ERBB2 Signaling"),
        "description": "Trastuzumab + ERBB2 Signaling (HER2 is pathway component)",
        "expected_cancers": ["Breast cancer"],
    },
    {
        "agent": Agent(name="Vemurafenib", category="immunotherapy"),
        "pathway": Pathway(name="MAPK Signaling"),
        "description": "Vemurafenib + MAPK Signaling (BRAF is pathway component)",
        "expected_cancers": ["Melanoma"],
    },
    {
        "agent": Agent(name="Erlotinib", category="immunotherapy"),
        "pathway": Pathway(name="EGFR Signaling"),
        "description": "Erlotinib + EGFR Signaling (EGFR is pathway component)",
        "expected_cancers": ["NSCLC"],
    },
]

NEGATIVE_CONTROLS = [
    # Should return empty list
    {
        "agent": Agent(name="Curcumin", category="natural_agent"),
        "pathway": Pathway(name="NF-kB Signaling"),
        "description": "Curcumin + NF-kB (no Phase III data)",
    },
    {
        "agent": Agent(name="Pembrolizumab", category="immunotherapy"),
        "pathway": Pathway(name="Tumor Antigen"),
        "description": "Pembrolizumab + Tumor Antigen (PD-1 not in pathway)",
    },
    {
        "agent": Agent(name="Gemcitabine", category="chemotherapy"),
        "pathway": Pathway(name="Androgen Signaling"),
        "description": "Gemcitabine + Androgen Signaling (no connection)",
    },
    {
        "agent": Agent(name="Imatinib-mesylate", category="immunotherapy"),
        "pathway": Pathway(name="Cell Cycle"),
        "description": "Imatinib + Cell Cycle (BCR-ABL not a Cell Cycle component)",
    },
]


def run_tests(model: MediPhiModel, parser: ResponseParser, verbose: bool = True) -> dict:
    """Run all test cases and return results."""
    results = {
        "positive_passed": 0,
        "positive_failed": 0,
        "negative_passed": 0,
        "negative_failed": 0,
        "details": [],
    }

    print("\n=== POSITIVE CONTROLS (should return interactions) ===\n")

    for test in POSITIVE_CONTROLS:
        print(f"\n--- Testing: {test['agent'].name} + {test['pathway'].name} ---")
        print("Generating analysis...")

        interactions, reasoning = generate_interaction_with_reasoning(
            test["agent"], test["pathway"], model, parser
        )

        print("Analysis complete.")

        print(f"DEBUG: MediPhi reasoning length: {len(reasoning)} chars")
        print(f"DEBUG: Number of interactions returned: {len(interactions)}")

        passed = len(interactions) > 0

        if passed:
            results["positive_passed"] += 1
            status = "PASS"
        else:
            results["positive_failed"] += 1
            status = "FAIL"

        print(f"\n[{status}] {test['description']}")
        if interactions:
            for i in interactions:
                print(f"       -> {i.cancer_type} ({i.agent_effect.value} {i.primary_target})")
        else:
            print("       -> No interactions returned")
            print(f"       DEBUG: Full reasoning:\n{reasoning}")

        print()

        results["details"].append({
            "type": "positive",
            "description": test["description"],
            "passed": passed,
            "reasoning": reasoning,
            "interactions": [
                {
                    "cancer_type": i.cancer_type,
                    "primary_target": i.primary_target,
                    "agent_effect": i.agent_effect.value,
                }
                for i in interactions
            ],
        })

    print("\n=== NEGATIVE CONTROLS (should return empty) ===\n")
    for test in NEGATIVE_CONTROLS:
        interactions, reasoning = generate_interaction_with_reasoning(
            test["agent"], test["pathway"], model, parser
        )
        passed = len(interactions) == 0

        if passed:
            results["negative_passed"] += 1
            status = "PASS"
        else:
            results["negative_failed"] += 1
            status = "FAIL"

        print(f"[{status}] {test['description']}")
        if interactions:
            for i in interactions:
                print(f"       -> UNEXPECTED: {i.cancer_type} ({i.agent_effect} {i.primary_target})")
        else:
            print("       -> Correctly returned empty")

        if verbose and not passed:
            print(f"       REASONING:\n{reasoning[:500]}...")
        print()

        results["details"].append({
            "type": "negative",
            "description": test["description"],
            "passed": passed,
            "reasoning": reasoning,
            "interactions": [
                {
                    "cancer_type": i.cancer_type,
                    "primary_target": i.primary_target,
                    "agent_effect": i.agent_effect.value,
                }
                for i in interactions
            ],
        })

    # Summary
    total_positive = results["positive_passed"] + results["positive_failed"]
    total_negative = results["negative_passed"] + results["negative_failed"]
    total_passed = results["positive_passed"] + results["negative_passed"]
    total_tests = total_positive + total_negative

    print("\n=== SUMMARY ===")
    print(f"Positive controls: {results['positive_passed']}/{total_positive}")
    print(f"Negative controls: {results['negative_passed']}/{total_negative}")
    print(f"Overall: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")

    return results


def main():
    print("=== TWO-STEP ARCHITECTURE ===")
    print("Step 1: MediPhi-PubMed (biomedical reasoning)")
    print("Step 2: Mistral-7B-Instruct (JSON extraction)\n")

    print("Loading MediPhi-PubMed model...")
    model = MediPhiModel(
        model_name=MEDIPHI_MODEL,
        cache_dir=MODEL_CACHE_DIR,
    )
    model.load()
    print("MediPhi loaded.\n")

    print("Loading Mistral parser model...")
    parser = ResponseParser(
        model_name=PARSER_MODEL,
        cache_dir=MODEL_CACHE_DIR,
    )
    parser.load()
    print("Parser loaded.\n")

    results = run_tests(model, parser)

    # Success criteria: >=90% correct classification
    total_passed = results["positive_passed"] + results["negative_passed"]
    total_tests = len(POSITIVE_CONTROLS) + len(NEGATIVE_CONTROLS)
    accuracy = total_passed / total_tests

    if accuracy >= 0.9:
        print(f"\nSUCCESS: {accuracy*100:.1f}% accuracy meets >=90% threshold")
        return 0
    else:
        print(f"\nNEEDS IMPROVEMENT: {accuracy*100:.1f}% accuracy below 90% threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())
