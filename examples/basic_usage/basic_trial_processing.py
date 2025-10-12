#!/usr/bin/env python3
"""
🚀 mCODE Translator - Basic Trial Processing Example

This example demonstrates the fundamental usage of the mCODE Translator
for processing a single clinical trial and extracting mCODE elements.

Features demonstrated:
- Basic trial data fetching
- Simple mCODE extraction
- Result validation
- Output formatting
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_flow_coordinator import process_clinical_trials_flow


def basic_trial_processing() -> bool:
    """Demonstrate basic trial processing workflow."""
    print("🚀 mCODE Translator - Basic Trial Processing")
    print("=" * 60)

    # Configuration for basic processing
    config = {
        "validate_data": True,
        "store_results": False,  # Don't store for demo
        "enable_logging": True,
        "processing_engine": "regex"  # Use fast regex engine for demo
    }

    print("📋 Configuration:")
    print(f"   • Engine: {config['processing_engine']}")
    print(f"   • Validation: {config['validate_data']}")
    print(f"   • Storage: {config['store_results']}")
    print()

    # Example trial ID (well-known breast cancer trial)
    trial_id = "NCT02364999"  # PALOMA-2 trial

    print(f"🎯 Processing Trial: {trial_id}")
    print("-" * 40)

    try:
        # Process the trial
        result = process_clinical_trials_flow(
            trial_ids=[trial_id],
            config=config
        )

        print("✅ Processing completed successfully!")
        print()

        # Display results
        if result.data and len(result.data) > 0:
            trial_data = result.data[0]

            print("📊 Trial Information:")
            print(f"   • NCT ID: {trial_data.get('nct_id', 'N/A')}")
            print(f"   • Title: {trial_data.get('brief_title', 'N/A')[:80]}...")
            print(f"   • Phase: {trial_data.get('phase', 'N/A')}")
            print(f"   • Condition: {trial_data.get('condition', 'N/A')}")
            print()

            # Show mCODE mappings if available
            if 'mcode_mappings' in trial_data and trial_data['mcode_mappings']:
                print("🧬 mCODE Elements Extracted:")
                mappings = trial_data['mcode_mappings'][:5]  # Show first 5
                for i, mapping in enumerate(mappings, 1):
                    element_type = mapping.get('element_type', 'Unknown')
                    confidence = mapping.get('confidence_score', 0)
                    print(f"   {i}. {element_type} ({confidence:.1%} confidence)")
                if len(trial_data['mcode_mappings']) > 5:
                    print(f"   ... and {len(trial_data['mcode_mappings']) - 5} more")
                print()

            # Show processing metrics
            if 'metadata' in result and result['metadata']:
                print("⚡ Processing Metrics:")
                metadata = result['metadata']
                if 'processing_time_seconds' in metadata:
                    print(f"   • Processing Time: {metadata['processing_time_seconds']:.2f}s")
                if 'entities_count' in metadata:
                    print(f"   • Elements Extracted: {metadata['entities_count']}")
                print()

        # Show validation results
        if 'validation_results' in result:
            validation = result['validation_results']
            print("✅ Validation Results:")
            if 'compliance_score' in validation:
                print(f"   • Compliance Score: {validation['compliance_score']:.1%}")
            if validation.get('validation_errors'):
                print(f"   • Errors: {len(validation['validation_errors'])}")
            print()

        print("🎉 Basic trial processing example completed!")
        print()
        print("💡 Next Steps:")
        print("   • Try different trial IDs")
        print("   • Experiment with LLM engine (--engine llm)")
        print("   • Enable storage (--store-results)")
        print("   • Process multiple trials at once")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("   • Check your internet connection")
        print("   • Verify the trial ID exists")
        print("   • Ensure all dependencies are installed")
        return False

    return True


if __name__ == "__main__":
    success = basic_trial_processing()
    sys.exit(0 if success else 1)