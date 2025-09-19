#!/usr/bin/env python3
"""
Debug script to test LLM service directly and see what responses we're getting.
"""

import json
import os
from src.pipeline.llm_service import LLMService
from src.utils.config import Config

def test_llm_service():
    """Test the LLM service with sample clinical text."""

    # Initialize config
    config = Config()

    # Initialize LLM service
    llm_service = LLMService(
        config=config,
        model_name="deepseek-coder",
        prompt_name="direct_mcode_evidence_based_concise"
    )

    # Sample clinical text from the trial data
    clinical_text = """
    This phase II trial studies the side effects and how well palbociclib and letrozole or fulvestrant works in treating patients aged 70 years and older with estrogen receptor positive, HER2 negative breast cancer that has spread to other places in the body.

    PRIMARY OBJECTIVES:
    I. To estimate the safety and tolerability of the combination of palbociclib and letrozole or fulvestrant in adults age 70 or older with estrogen receptor-positive, HER2-negative metastatic breast cancer.

    Conditions: Estrogen Receptor-positive Breast Cancer, HER2/Neu Negative, Stage IV Breast Cancer AJCC v6 and v7

    Interventions: Palbociclib, Letrozole, Fulvestrant

    Eligibility Criteria:
    Inclusion Criteria:
    * Documentation of disease: estrogen receptor positive and/or progesterone receptor (PR) positive, HER2 negative metastatic breast cancer; histologic confirmation is required
    * Measurable disease or non-measurable disease
    * Planning to begin palbociclib for metastatic disease
    """

    print("Testing LLM service with sample clinical text...")
    print(f"Text length: {len(clinical_text)} characters")
    print(f"Model: deepseek-coder")
    print(f"Prompt: direct_mcode_evidence_based_concise")
    print()

    try:
        # Test the LLM service
        result = llm_service.map_to_mcode(clinical_text)

        print("LLM Service Result:")
        print(f"Number of mCODE elements extracted: {len(result)}")
        print()

        if result:
            for i, element in enumerate(result):
                print(f"Element {i+1}:")
                print(f"  Type: {element.element_type}")
                print(f"  Code: {element.code}")
                print(f"  Display: {element.display}")
                print(f"  System: {element.system}")
                print(f"  Confidence: {element.confidence_score}")
                print(f"  Evidence: {element.evidence_text}")
                print()
        else:
            print("No mCODE elements were extracted!")
            print("This indicates the LLM is not finding any mCODE-relevant information.")
            print()

        # Check API key
        try:
            api_key = config.get_api_key("deepseek-coder")
            if api_key:
                print(f"✅ API key found for deepseek-coder (length: {len(api_key)})")
            else:
                print("❌ No API key found for deepseek-coder")
        except Exception as e:
            print(f"❌ API key error: {e}")

    except Exception as e:
        print(f"❌ Error testing LLM service: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_service()