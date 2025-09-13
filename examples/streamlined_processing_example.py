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

# Use centralized logging
from src.utils.logging_config import setup_logging, get_logger

# Setup centralized logging
setup_logging()
logger = get_logger("streamlined_processing_example")

from src.core.data_flow_coordinator import process_clinical_trials_flow
from src.workflows.streamlined_workflow import create_trial_processor
from src.shared.models import WorkflowResult


async def main():
    """Demonstrate streamlined clinical trial processing."""

    logger.info("🧪 mCODE Translator - Streamlined Processing Example")
    logger.info("=" * 60)

    # Example trial IDs (replace with real NCT IDs)
    trial_ids = [
        "NCT03170960",  # Example breast cancer trial
        "NCT03805399",  # Example TNBC trial
    ]

    logger.info(f"📋 Processing {len(trial_ids)} clinical trials...")
    logger.info(f"   Trial IDs: {', '.join(trial_ids)}")
    logger.info("")

    try:
        # Method 1: Complete streamlined data flow
        logger.info("🚀 Method 1: Complete Data Flow Coordinator")
        logger.info("-" * 40)

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
            logger.info("✅ Complete flow successful!")
            metadata = result.metadata
            logger.info(f"   📊 Trials requested: {metadata['total_trials_requested']}")
            logger.info(f"   📥 Trials fetched: {metadata['trials_fetched']}")
            logger.info(f"   🔬 Trials processed: {metadata['trials_processed']}")
            logger.info(".1%")
        else:
            logger.error("❌ Complete flow failed!")
            logger.error(f"   Error: {result.error_message}")

        logger.info("")

        # Method 2: Individual component usage
        logger.info("🔧 Method 2: Individual Component Usage")
        logger.info("-" * 40)

        processor = create_trial_processor(config={
            "processor": {
                "model_name": "deepseek-coder",
                "prompt_name": "direct_mcode_evidence_based_concise"
            }
        })

        logger.info("📊 Processing statistics:")
        stats = processor.get_processing_stats()
        logger.info(f"   Pipeline type: {stats['pipeline_type']}")
        logger.info(f"   Has validator: {stats['has_validator']}")
        logger.info(f"   Has processor: {stats['has_processor']}")
        logger.info(f"   Has storage: {stats['has_storage']}")

        logger.info("")

        # Method 3: Type-safe data handling
        logger.info("🛡️  Method 3: Type-Safe Data Models")
        logger.info("-" * 40)

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
            logger.info("✅ Clinical trial data validated successfully!")
            logger.info(f"   NCT ID: {trial.nct_id}")
            logger.info(f"   Title: {trial.brief_title}")
            logger.info(f"   Study type: {trial.studyType}")
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")

        logger.info("")

        logger.info("🎉 Example completed successfully!")
        logger.info("\nKey Benefits Demonstrated:")
        logger.info("• Type-safe data validation with Pydantic")
        logger.info("• Streamlined fetch → validate → process → store flow")
        logger.info("• Dependency injection for clean architecture")
        logger.info("• Comprehensive error handling and logging")
        logger.info("• Easy-to-use high-level APIs")

    except Exception as e:
        logger.error(f"💥 Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)