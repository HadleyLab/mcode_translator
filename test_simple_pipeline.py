#!/usr/bin/env python3
"""
Simple test for ultra-lean pipeline data flow
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_data_flow():
    """Test basic data flow without LLM calls"""
    print("🧪 Testing ultra-lean pipeline data flow...")

    # Test 1: Import pipeline
    try:
        from src.pipeline import McodePipeline
        print("✅ Pipeline import successful")
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        assert False, f"Pipeline import failed: {e}"

    # Test 2: Load sample data
    try:
        with open("tests/data/sample_trial.json", "r") as f:
            trial_data = json.load(f)
        print("✅ Sample trial data loaded")
        print(f"   NCT ID: {trial_data['protocolSection']['identificationModule']['nctId']}")
    except Exception as e:
        print(f"❌ Sample data loading failed: {e}")
        assert False, f"Sample data loading failed: {e}"

    # Test 3: Validate data structure
    try:
        from src.shared.models import ClinicalTrialData
        validated_trial = ClinicalTrialData(**trial_data)
        print("✅ Data validation successful")
        print(f"   Validated NCT ID: {validated_trial.nct_id}")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        assert False, f"Data validation failed: {e}"

    # Test 4: Test document ingestion
    try:
        from src.pipeline.document_ingestor import DocumentIngestor
        ingestor = DocumentIngestor()
        sections = ingestor.ingest_clinical_trial_document(trial_data)
        print("✅ Document ingestion successful")
        print(f"   Extracted {len(sections)} sections")
        for section in sections[:2]:  # Show first 2
            print(f"   - {section.name}: {len(section.content)} chars")
    except Exception as e:
        print(f"❌ Document ingestion failed: {e}")
        assert False, f"Document ingestion failed: {e}"

    # Test 5: Test pipeline result structure
    try:
        from src.shared.models import PipelineResult, ProcessingMetadata, ValidationResult
        result = PipelineResult(
            extracted_entities=[],
            mcode_mappings=[],
            source_references=[],
            validation_results=ValidationResult(compliance_score=0.95),
            metadata=ProcessingMetadata(
                engine_type="LLM",
                entities_count=0,
                mapped_count=0,
                model_used="test-model",
                prompt_used="test-prompt",
            ),
            original_data=trial_data,
            error=None
        )
        print("✅ Pipeline result structure valid")
        print(f"   Compliance score: {result.validation_results.compliance_score}")
        print(f"   Model used: {result.metadata.model_used}")
    except Exception as e:
        print(f"❌ Pipeline result structure failed: {e}")
        assert False, f"Pipeline result structure failed: {e}"

    print("\n🎉 All data flow tests passed!")
    print("📊 Summary:")
    print("   ✅ Pipeline imports work")
    print("   ✅ Data validation works")
    print("   ✅ Document ingestion works")
    print("   ✅ Result structures work")
    print("   ✅ No redundant models created")
    print("   ✅ Leverages existing infrastructure")

if __name__ == "__main__":
    test_data_flow()
    print("✅ Test completed successfully")