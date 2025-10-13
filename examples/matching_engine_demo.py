#!/usr/bin/env python3
"""
Triple-Engine Patient-Trial Matching Demo

This script demonstrates the three patient-trial matching engines:
1. RegexRulesEngine - Fast, deterministic pattern-based matching
2. LLMMatchingEngine - Intelligent LLM-powered matching
3. CoreMemoryGraphEngine - Knowledge graph-based relationship matching
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matching import (
    RegexRulesEngine,
    LLMMatchingEngine,
    CoreMemoryGraphEngine,
    tools,
)
from src.services.heysol_client import OncoCoreClient
from src.config.patterns_config import load_regex_rules

async def demo_matching_engines():
    """Demonstrate basic matching capabilities of each engine."""
    print("🚀 Triple-Engine Patient-Trial Matching Demo")
    print("=" * 50)

    # Initialize services
    heysol_client = OncoCoreClient.from_env()

    # Initialize engines
    regex_rules = load_regex_rules()
    regex_engine = RegexRulesEngine(rules=regex_rules)
    llm_engine = LLMMatchingEngine(model_name="deepseek-coder", prompt_name="direct_mcode_evidence_based_concise")
    memory_engine = CoreMemoryGraphEngine(heysol_client=heysol_client)

    # Create sample patient and trial data (simplified for demo)
    patient_data = {"patient_id": "P001", "conditions": ["breast cancer", "hypertension"]}
    trial_criteria = {
        "eligibilityCriteria": """
        Inclusion Criteria:
        - Age >= 18 years
        - Histologically confirmed breast cancer
        - ECOG performance status 0-1

        Exclusion Criteria:
        - Pregnancy or lactation
        - Active infection
        """
    }

    # Test each engine
    engines = [
        ("RegexRulesEngine", regex_engine),
        ("LLMMatchingEngine", llm_engine),
        ("CoreMemoryGraphEngine", memory_engine),
    ]

    print("\n📊 Matching Results (Streaming):")
    print("-" * 30)

    for engine_name, engine in engines:
        try:
            print(f"\n🔍 {engine_name}:")
            async for update in engine.match_streaming(patient_data, trial_criteria):
                if update["status"] == "completed":
                    results = update["results"]
                    print(f"   Found {len(results)} matches.")
                    for result in results:
                        print(f"     - {result['element_type']}: {result['display']} (Score: {result['confidence_score']:.2f})")
                elif update["status"] == "error":
                    print(f"   ❌ Error: {update['error']}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

async def demo_tools():
    """Demonstrate the use of matching tools."""
    print("\n\n🛠️ Matching Tools Demo")
    print("=" * 50)

    client = OncoCoreClient.from_env()

    # Ingest a new memory
    print("Ingesting a new memory...")
    ingest_result = await tools.ingest_memory(
        client, "Patient P001 was successfully matched to trial NCT04555577."
    )
    print(f"Ingest result: {ingest_result}")

    # Search for the memory
    print("\nSearching for the new memory...")
    search_results = await tools.search_memory(client, query="Patient P001")
    print(f"Search results: {search_results}")

async def main():
    """Run all demos."""
    try:
        await demo_matching_engines()
        await demo_tools()
        print("\n\n✅ Demo completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())