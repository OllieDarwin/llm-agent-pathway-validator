#!/usr/bin/env python
"""Run the full 5-stage pipeline for agent-pathway validation."""

import sys
import json
import signal
from pathlib import Path
from datetime import datetime

from config import (
    REASONING_MODEL,
    PARSER_MODEL,
    MODEL_CACHE_DIR,
    AGENTS_CSV,
    PATHWAYS_CSV,
)
from data.loader import load_agents, load_pathways, generate_combinations
from models.reasoning import ReasoningModel
from models.parser import Parser
from utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Global variable to store partial results
_partial_results = None
_results_saved = False


def run_stage1(agents, pathways, reasoning_model, parser):
    """Stage 1: Generate initial agent-pathway-cancer interactions."""
    global _partial_results

    logger.info("\n" + "="*60)
    logger.info("STAGE 1: Generate Interactions")
    logger.info("="*60 + "\n")

    combinations = generate_combinations(agents, pathways)
    logger.info(f"Processing {len(combinations)} agent-pathway combinations")

    # Process one combination at a time so we can save partial results
    from stages.generate import generate_interaction

    results = {}

    for i, (agent, pathway) in enumerate(combinations):
        # Update partial results after each combination
        key = (agent.name, pathway.name)
        interactions = generate_interaction(agent, pathway, reasoning_model, parser)
        results[key] = interactions

        # Update global partial results (for Ctrl+C handler)
        _partial_results = results

        # Progress logging
        if (i + 1) % 100 == 0 or (i + 1) == len(combinations):
            pct = ((i + 1) / len(combinations)) * 100
            logger.info(f"  [{i + 1}/{len(combinations)} - {pct:.1f}%] Progress checkpoint")

    # Summary
    total_interactions = sum(len(interactions) for interactions in results.values())
    combinations_with_interactions = sum(1 for interactions in results.values() if interactions)

    logger.info(f"\nStage 1 Complete:")
    logger.info(f"  - Combinations processed: {len(results)}")
    logger.info(f"  - Combinations with interactions: {combinations_with_interactions}")
    logger.info(f"  - Total interactions found: {total_interactions}")

    return results


def run_stage2(stage1_results):
    """Stage 2: Get publications via Exa.ai (TODO)."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 2: Get Publications")
    logger.info("="*60 + "\n")

    logger.info("TODO: Implement Stage 2 - Literature search via Exa.ai")
    logger.info("  - Input: Agent-pathway-cancer interactions from Stage 1")
    logger.info("  - Output: Relevant publications (max 10 per interaction)")

    # Placeholder - return stage1 results unchanged for now
    return stage1_results


def run_stage3(stage2_results):
    """Stage 3: Verify publications relevance (TODO)."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 3: Verify Publications")
    logger.info("="*60 + "\n")

    logger.info("TODO: Implement Stage 3 - Relevance scoring")
    logger.info("  - Input: Publications from Stage 2")
    logger.info("  - Output: Filtered publications with confidence scores")

    # Placeholder
    return stage2_results


def run_stage4(stage3_results):
    """Stage 4: Distill publications for mechanistic data (TODO)."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 4: Distill Publications")
    logger.info("="*60 + "\n")

    logger.info("TODO: Implement Stage 4 - Extract mechanistic data")
    logger.info("  - Input: Verified publications from Stage 3")
    logger.info("  - Output: Key findings, mechanisms, evidence quality")

    # Placeholder
    return stage3_results


def run_stage5(stage4_results):
    """Stage 5: Synthesize final confidence assessment (TODO)."""
    logger.info("\n" + "="*60)
    logger.info("STAGE 5: Synthesize Interaction")
    logger.info("="*60 + "\n")

    logger.info("TODO: Implement Stage 5 - Final confidence assessment")
    logger.info("  - Input: Distilled mechanistic data from Stage 4")
    logger.info("  - Output: Overall confidence, recommendation tier, caveats")

    # Placeholder
    return stage4_results


def save_results(results, stage_num, partial=False):
    """Save stage results to JSON."""
    global _results_saved

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "partial_" if partial else ""
    output_file = Path(f"{prefix}stage{stage_num}_results_{timestamp}.json")

    status = "partial" if partial else "complete"
    logger.info(f"\nSaving {status} Stage {stage_num} results to {output_file}")

    # Convert to serializable format
    results_serializable = {
        f"{agent}_{pathway}": [
            {
                "agent_name": i.agent_name,
                "pathway_name": i.pathway_name,
                "cancer_type": i.cancer_type,
                "agent_effect": i.agent_effect.value,
                "primary_target": i.primary_target,
                "target_status": i.target_status.value,
                "mechanism_type": i.mechanism_type.value,
            }
            for i in interactions
        ]
        for (agent, pathway), interactions in results.items()
    }

    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"✓ Results saved to {output_file}")
    logger.info(f"  - Combinations processed: {len(results_serializable)}")

    _results_saved = True
    return output_file


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully - save partial results."""
    global _partial_results, _results_saved

    logger.info("\n\n" + "="*60)
    logger.info("⚠️  INTERRUPTED BY USER (Ctrl+C)")
    logger.info("="*60)

    if _partial_results is not None and not _results_saved:
        logger.info("\nSaving partial results before exit...")
        try:
            output_file = save_results(_partial_results, stage_num=1, partial=True)
            logger.info(f"\n✓ Partial results saved successfully!")
            logger.info(f"  Resume later by excluding already-processed combinations")
        except Exception as e:
            logger.error(f"\n✗ Failed to save partial results: {e}")
    else:
        logger.info("\nNo partial results to save.")

    logger.info("\nExiting...")
    sys.exit(0)


def main():
    """Run the full 5-stage pipeline."""
    # Register signal handler for graceful Ctrl+C handling
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("="*60)
    logger.info("TUMOUR SIGNALLING PANEL PIPELINE")
    logger.info("="*60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("(Press Ctrl+C to save partial results and exit)")

    # Load data
    logger.info("\nLoading agents and pathways...")
    agents = load_agents(AGENTS_CSV)
    pathways = load_pathways(PATHWAYS_CSV)
    logger.info(f"  - {len(agents)} agents")
    logger.info(f"  - {len(pathways)} pathways")
    logger.info(f"  - {len(agents) * len(pathways)} total combinations")

    # Load models for Stage 1
    logger.info("\nLoading models...")
    logger.info(f"  - Reasoning: {REASONING_MODEL}")
    reasoning_model = ReasoningModel(model_name=REASONING_MODEL, cache_dir=MODEL_CACHE_DIR)
    reasoning_model.load()

    logger.info(f"  - Parser: {PARSER_MODEL}")
    parser = Parser(model_name=PARSER_MODEL, cache_dir=MODEL_CACHE_DIR)
    parser.load()

    logger.info("✓ Models loaded")

    # Run pipeline stages
    stage1_results = run_stage1(agents, pathways, reasoning_model, parser)
    save_results(stage1_results, stage_num=1)

    stage2_results = run_stage2(stage1_results)
    # save_results(stage2_results, stage_num=2)  # Uncomment when implemented

    stage3_results = run_stage3(stage2_results)
    # save_results(stage3_results, stage_num=3)  # Uncomment when implemented

    stage4_results = run_stage4(stage3_results)
    # save_results(stage4_results, stage_num=4)  # Uncomment when implemented

    final_results = run_stage5(stage4_results)
    # save_results(final_results, stage_num=5)  # Uncomment when implemented

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nNext steps:")
    logger.info("  1. Review Stage 1 results")
    logger.info("  2. Implement Stage 2 (getPublications)")
    logger.info("  3. Implement Stage 3 (verifyPublications)")
    logger.info("  4. Implement Stage 4 (distillPublications)")
    logger.info("  5. Implement Stage 5 (synthesizeInteraction)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
