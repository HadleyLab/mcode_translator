"""
Patient-Trial Matching CLI Commands
"""

import typer
from rich.console import Console

from src.matching import (
    RegexRulesEngine,
    LLMMatchingEngine,
    CoreMemoryGraphEngine,
    EnsembleDecisionEngine,
)
from src.matching.ensemble_decision_engine import ConsensusMethod
from src.services.heysol_client import OncoCoreClient
from src.config.patterns_config import load_regex_rules

app = typer.Typer(
    name="matching",
    help="Patient-trial matching operations",
    add_completion=True,
)
console = Console()

@app.command("run")
def run_matching(
    patient_file: str = typer.Option(..., help="Path to patient data file (JSON)."),
    trial_file: str = typer.Option(..., help="Path to trial data file (JSON)."),
    engine: str = typer.Option("llm", help="Matching engine to use: 'regex', 'llm', 'memory', or 'ensemble'."),
    cache_enabled: bool = typer.Option(True, help="Enable caching for API calls and results."),
    max_retries: int = typer.Option(3, help="Maximum number of retries on failure."),
    continue_on_error: bool = typer.Option(True, help="Continue processing even if individual matches fail."),
    # Ensemble-specific parameters
    consensus_method: str = typer.Option("dynamic_weighting", help="Consensus method for ensemble (weighted_majority_vote, confidence_weighted, bayesian_ensemble, dynamic_weighting)."),
    min_experts: int = typer.Option(2, help="Minimum number of experts for ensemble (2-3)."),
    max_experts: int = typer.Option(3, help="Maximum number of experts for ensemble (2-3)."),
    enable_rule_integration: bool = typer.Option(True, help="Enable rule-based integration in ensemble."),
    enable_dynamic_weighting: bool = typer.Option(True, help="Enable dynamic weighting in ensemble."),
):
    """
    Run patient-trial matching using a specified engine.
    """
    console.print(f"🚀 Running patient-trial matching with '{engine}' engine...")
    
    # Load patient and trial data
    # (In a real implementation, this would be more robust)
    import json
    with open(patient_file, "r") as f:
        patient_data = json.load(f)
    with open(trial_file, "r") as f:
        trial_data = json.load(f)

    # Initialize engines with enhanced options
    try:
        heysol_client = OncoCoreClient.from_env()
        regex_rules = load_regex_rules()

        engines = {
            "regex": RegexRulesEngine(rules=regex_rules, cache_enabled=cache_enabled, max_retries=max_retries),
            "llm": LLMMatchingEngine(model_name="deepseek-coder", prompt_name="patient_matcher", cache_enabled=cache_enabled, max_retries=max_retries),
            "memory": CoreMemoryGraphEngine(heysol_client=heysol_client, cache_enabled=cache_enabled, max_retries=max_retries),
            "ensemble": EnsembleDecisionEngine(
                model_name="deepseek-coder",
                consensus_method=ConsensusMethod(consensus_method),
                min_experts=min_experts,
                max_experts=max_experts,
                enable_rule_based_integration=enable_rule_integration,
                enable_dynamic_weighting=enable_dynamic_weighting,
                cache_enabled=cache_enabled,
                max_retries=max_retries
            ),
        }

        matching_engine = engines.get(engine)
        if not matching_engine:
            console.print(f"❌ Error: Invalid engine '{engine}'.")
            raise typer.Exit(1)

        # Run matching with recovery
        import asyncio
        if continue_on_error:
            result = asyncio.run(matching_engine.match_with_recovery(patient_data, trial_data))
            if result.error:
                console.print(f"⚠️  Warning: Matching completed with errors: {result.error}")
                console.print(f"   Metadata: {result.metadata}")
            results = result.elements
        else:
            results = asyncio.run(matching_engine.match(patient_data, trial_data))

    except Exception as e:
        console.print(f"❌ Error initializing or running {engine} engine: {e}")
        if continue_on_error:
            console.print("💡 Continuing with empty results due to continue_on_error=True")
            results = []
        else:
            raise typer.Exit(1)

    # Print results
    console.print(f"\n📊 Matching Results:")
    if results:
        for result in results:
            console.print(f"  - {result.element_type}: {result.display} (Score: {result.confidence_score:.2f})")
    else:
        console.print("  No matches found.")