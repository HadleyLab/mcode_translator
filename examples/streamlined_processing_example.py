#!/usr/bin/env python3
"""
Example: Streamlined Clinical Trial Processing with New Architecture

This example demonstrates the complete streamlined data flow:
1. Fetch clinical trial data
2. Validate data using Pydantic models
3. Process with mCODE mapping
4. Store results in Core Memory

Usage:
    python examples/streamlined_processing_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.data_flow_coordinator import process_clinical_trials_flow
from src.workflows.streamlined_workflow import create_trial_processor
from src.shared.models import WorkflowResult


async def main():
    """Demonstrate streamlined clinical trial processing."""

    print("🧪 mCODE Translator - Streamlined Processing Example")
    print("=" * 60)

    # Example trial IDs (replace with real NCT IDs)
    trial_ids = [
        "NCT03170960",  # Example breast cancer trial
        "NCT03805399",  # Example TNBC trial
    ]

    print(f"📋 Processing {len(trial_ids)} clinical trials...")
    print(f"   Trial IDs: {', '.join(trial_ids)}")
    print()

    try:
        # Method 1: Complete streamlined data flow
        print("🚀 Method 1: Complete Data Flow Coordinator")
        print("-" * 40)

        result = process_clinical_trials_flow(
            trial_ids=trial_ids,
            config={
                "validate_data": True,
                "store_results": True,
                "batch_size": 2,
                "processor": {
                    "model_name": "deepseek-coder",
                    "prompt_name": "direct_mcode_evidence_based_concise"
                }
            }
        )

        if result.success:
            print("✅ Complete flow successful!")
            metadata = result.metadata
            print(f"   📊 Trials requested: {metadata['total_trials_requested']}")
            print(f"   📥 Trials fetched: {metadata['trials_fetched']}")
            print(f"   🔬 Trials processed: {metadata['trials_processed']}")
            print(".1%")
        else:
            print("❌ Complete flow failed!")
            print(f"   Error: {result.error_message}")

        print()

        # Method 2: Individual component usage
        print("🔧 Method 2: Individual Component Usage")
        print("-" * 40)

        processor = create_trial_processor(config={
            "processor": {
                "model_name": "deepseek-coder",
                "prompt_name": "direct_mcode_evidence_based_concise"
            }
        })

        print("📊 Processing statistics:")
        stats = processor.get_processing_stats()
        print(f"   Pipeline type: {stats['pipeline_type']}")
        print(f"   Has validator: {stats['has_validator']}")
        print(f"   Has processor: {stats['has_processor']}")
        print(f"   Has storage: {stats['has_storage']}")

        print()

        # Method 3: Type-safe data handling
        print("🛡️  Method 3: Type-Safe Data Models")
        print("-" * 40)

        # Example of creating validated data models
        from src.shared.models import ClinicalTrialData, McodeElement

        # This would normally come from the API
        sample_trial_data = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT123456",
                    "briefTitle": "Sample Breast Cancer Trial",
                    "officialTitle": "A Study of Treatment for Breast Cancer"
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Patients with breast cancer",
                    "healthyVolunteers": False,
                    "sex": "female",
                    "minimumAge": "18 years",
                    "maximumAge": "75 years"
                }
            },
            "hasResults": False,
            "studyType": "Interventional",
            "overallStatus": "Recruiting",
            "phase": "Phase 2"
        }

        # Automatic validation and type safety
        try:
            trial = ClinicalTrialData(**sample_trial_data)
            print("✅ Clinical trial data validated successfully!")
            print(f"   NCT ID: {trial.nct_id}")
            print(f"   Title: {trial.brief_title}")
            print(f"   Study type: {trial.studyType}")
        except Exception as e:
            print(f"❌ Validation failed: {e}")

        print()

        print("🎉 Example completed successfully!")
        print("\nKey Benefits Demonstrated:")
        print("• Type-safe data validation with Pydantic")
        print("• Streamlined fetch → validate → process → store flow")
        print("• Dependency injection for clean architecture")
        print("• Comprehensive error handling and logging")
        print("• Easy-to-use high-level APIs")

    except Exception as e:
        print(f"💥 Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)