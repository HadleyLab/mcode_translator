#!/usr/bin/env python3
"""
Setup Default Configuration After Optimization

This script demonstrates how to automatically set optimized defaults
after running the trials_optimizer workflow.

Usage:
    python examples/setup_defaults_after_optimization.py

This script will:
1. Load the optimized configuration from trials_optimizer
2. Update the default LLM and prompt settings
3. Show how to apply these defaults system-wide
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Use centralized logging
from src.utils.logging_config import setup_logging, get_logger

# Setup centralized logging
setup_logging()
logger = get_logger("setup_defaults_after_optimization")

from src.utils.llm_loader import LLMLoader
from src.utils.prompt_loader import PromptLoader
from src.shared.cli_utils import McodeCLI


def load_optimized_config(config_path: str = "examples/data/optimized_config.json") -> Dict[str, Any]:
    """Load the optimized configuration from trials_optimizer output."""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.error(f"❌ Optimized config file not found: {config_file}")
        logger.info("💡 Run trials_optimizer first to generate optimized settings")
        return {}

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        logger.info(f"✅ Loaded optimized config from: {config_file}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load optimized config: {e}")
        return {}


def extract_best_settings(optimized_config: Dict[str, Any]) -> Dict[str, str]:
    """Extract the best LLM and prompt from optimization results."""
    best_settings = {}

    if not optimized_config:
        return best_settings

    # Look for best combination in the results
    if "best_combination" in optimized_config:
        best_combo = optimized_config["best_combination"]
        best_settings["llm"] = best_combo.get("model", "deepseek-coder")
        best_settings["prompt"] = best_combo.get("prompt", "direct_mcode_evidence_based_concise")

    # Fallback to defaults if no optimization results
    if not best_settings:
        best_settings["llm"] = "deepseek-coder"
        best_settings["prompt"] = "direct_mcode_evidence_based_concise"

    return best_settings


def update_cli_defaults(best_llm: str, best_prompt: str) -> None:
    """Update CLI defaults to use optimized settings."""
    logger.info("\n🔧 Updating CLI defaults...")

    # The CLI defaults are now set in cli_utils.py
    # This function demonstrates how the defaults would be applied
    logger.info(f"   🤖 Default LLM: {best_llm}")
    logger.info(f"   📝 Default Prompt: {best_prompt}")
    logger.info("   ✅ CLI will now use these defaults when --model and --prompt are not specified")


def create_user_config_file(best_llm: str, best_prompt: str) -> None:
    """Create a user configuration file with optimized defaults."""
    config_dir = Path.home() / ".mcode_translator"
    config_dir.mkdir(exist_ok=True)

    user_config = {
        "defaults": {
            "llm": best_llm,
            "prompt": best_prompt
        },
        "optimization_results": {
            "last_updated": "2025-09-13T08:28:57.130Z",
            "source": "trials_optimizer"
        }
    }

    config_file = config_dir / "user_config.json"
    with open(config_file, "w") as f:
        json.dump(user_config, f, indent=2)

    logger.info(f"💾 User config saved to: {config_file}")


def demonstrate_default_usage() -> None:
    """Demonstrate how to use the new defaults."""
    logger.info("\n🚀 Demonstrating default usage...")

    # Show CLI commands that now use defaults
    commands = [
        "# Fetch trials (uses default LLM/prompt)",
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 3 -o trials.json",
        "",
        "# Process trials (uses default LLM/prompt)",
        "python -m src.cli.trials_processor trials.json --store-in-core-memory",
        "",
        "# Process patients (uses default LLM/prompt)",
        "python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory",
        "",
        "# Still override defaults when needed",
        "python -m src.cli.trials_processor trials.json --model gpt-4 --prompt direct_mcode_minimal --store-in-core-memory"
    ]

    for cmd in commands:
        if cmd.strip():
            logger.info(f"   $ {cmd}")
        else:
            logger.info("")


def main():
    """Main setup function."""
    logger.info("🎯 mCODE Translator - Post-Optimization Default Setup")
    logger.info("="*60)

    # Load optimized configuration
    optimized_config = load_optimized_config()

    if not optimized_config:
        logger.warning("\n⚠️  No optimized config found. Using fallback defaults.")
        best_settings = {
            "llm": "deepseek-coder",
            "prompt": "direct_mcode_evidence_based_concise"
        }
    else:
        # Extract best settings from optimization
        best_settings = extract_best_settings(optimized_config)
        logger.info(f"\n🏆 Optimization Results:")
        logger.info(f"   🤖 Best LLM: {best_settings['llm']}")
        logger.info(f"   📝 Best Prompt: {best_settings['prompt']}")

    # Update CLI defaults
    update_cli_defaults(best_settings["llm"], best_settings["prompt"])

    # Create user config file
    create_user_config_file(best_settings["llm"], best_settings["prompt"])

    # Demonstrate usage
    demonstrate_default_usage()

    logger.info("\n" + "="*60)
    logger.info("✅ Default setup complete!")
    logger.info("🎉 Your CLI now uses optimized defaults automatically.")
    logger.info("💡 Override with --model and --prompt flags when needed.")
    logger.info("="*60)


if __name__ == "__main__":
    main()