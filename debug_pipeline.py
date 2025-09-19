#!/usr/bin/env python3
"""
Debug script to test the full pipeline and see where the issue occurs.
"""

import json
from src.pipeline.pipeline import McodePipeline
from src.pipeline.document_ingestor import DocumentIngestor

def test_full_pipeline():
    """Test the full pipeline with complete trial data."""

    # Load the complete trial data
    with open('examples/data/complete_trial.json', 'r') as f:
        trial_data = json.load(f)[0]  # Get first trial

    print("Testing full pipeline with complete trial data...")
    print(f"Trial NCT ID: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId')}")
    print()

    # First, test document ingestion
    print("=== DOCUMENT INGESTION ===")
    ingestor = DocumentIngestor()
    sections = ingestor.ingest_clinical_trial_document(trial_data)

    print(f"Number of sections extracted: {len(sections)}")
    for i, section in enumerate(sections):
        print(f"Section {i+1}: {section.name} ({section.source_type}) - {len(section.content)} chars")
        if len(section.content) < 500:  # Only show short sections
            print(f"  Content preview: {section.content[:200]}...")
        print()

    # Check which sections have content
    valid_sections = [s for s in sections if s.content and s.content.strip()]
    print(f"Sections with content: {len(valid_sections)}")
    print()

    # Now test the full pipeline
    print("=== FULL PIPELINE TEST ===")
    pipeline = McodePipeline(
        model_name="deepseek-coder",
        prompt_name="direct_mcode_evidence_based_concise"
    )

    result = pipeline.process(trial_data)

    print("Pipeline Result:")
    print(f"Success: {result.error is None}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Number of mCODE elements extracted: {len(result.mcode_mappings)}")
    print(f"Compliance score: {result.validation_results.compliance_score}")
    print()

    if result.mcode_mappings:
        print("Extracted mCODE elements:")
        for i, element in enumerate(result.mcode_mappings[:5]):  # Show first 5
            print(f"  {i+1}. {element.element_type}: {element.display}")
        if len(result.mcode_mappings) > 5:
            print(f"  ... and {len(result.mcode_mappings) - 5} more")
    else:
        print("❌ No mCODE elements extracted!")
        print("This indicates an issue in the pipeline processing.")
        print()

        # Debug: Test each section individually
        print("=== TESTING INDIVIDUAL SECTIONS ===")
        for section in valid_sections[:3]:  # Test first 3 sections
            print(f"Testing section: {section.name}")
            try:
                elements = pipeline.llm_service.map_to_mcode(section.content)
                print(f"  Elements extracted: {len(elements)}")
                if elements:
                    for elem in elements[:2]:  # Show first 2 elements
                        print(f"    - {elem.element_type}: {elem.display}")
            except Exception as e:
                print(f"  Error: {e}")
            print()

if __name__ == "__main__":
    test_full_pipeline()