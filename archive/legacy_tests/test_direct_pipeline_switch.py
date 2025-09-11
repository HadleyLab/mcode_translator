#!/usr/bin/env python3
"""
Test script to demonstrate switching from the two-step NLP -> mCODE pipeline
to the direct-to-mCODE pipeline approach.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Define paths
EXAMPLES_DIR = Path("examples/breast_cancer_data")
TEST_CASES_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.trial.json"

def load_test_data() -> Dict[str, Any]:
    """Load test case data"""
    logger.info("Loading test data...")
    
    with open(TEST_CASES_FILE, "r") as f:
        test_cases_data = json.load(f)
    
    # Extract the breast cancer test case
    test_case = test_cases_data["test_cases"]["breast_cancer_her2_positive"]
    
    logger.info("Test data loaded successfully")
    return test_case

def test_nlp_mcode_pipeline(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test the traditional two-step NLP -> mCODE pipeline"""
    try:
        logger.info("=== Testing Traditional NLP -> mCODE Pipeline ===")
        
        # Initialize traditional pipeline
        pipeline = NlpMcodePipeline(
            nlp_prompt_name="generic_extraction",
            mcode_prompt_name="generic_mapping"
        )
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        logger.info("✅ Traditional pipeline completed successfully")
        logger.info(f"   📊 Extracted entities: {len(result.extracted_entities)}")
        logger.info(f"   🗺️  mCODE mappings: {len(result.mcode_mappings)}")
        logger.info(f"   🎯 Compliance score: {result.validation_results.get('compliance_score', 0):.2%}")
        
        return {
            'success': True,
            'pipeline': 'nlp_mcode',
            'entities_count': len(result.extracted_entities),
            'mappings_count': len(result.mcode_mappings),
            'compliance_score': result.validation_results.get('compliance_score', 0),
            'result': result,
            'error': None
        }
    except Exception as e:
        logger.error(f"❌ Traditional pipeline failed: {e}")
        return {
            'success': False,
            'pipeline': 'nlp_mcode',
            'entities_count': 0,
            'mappings_count': 0,
            'compliance_score': 0,
            'result': None,
            'error': str(e)
        }

def test_direct_mcode_pipeline(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test the direct-to-mCODE pipeline"""
    try:
        logger.info("=== Testing Direct-to-mCODE Pipeline ===")
        
        # Initialize direct pipeline with direct mCODE prompt
        pipeline = McodePipeline(prompt_name="direct_mcode")
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        logger.info("✅ Direct pipeline completed successfully")
        logger.info(f"   🗺️  mCODE mappings: {len(result.mcode_mappings)}")
        logger.info(f"   🎯 Compliance score: {result.validation_results.get('compliance_score', 0):.2%}")
        
        return {
            'success': True,
            'pipeline': 'direct_mcode',
            'entities_count': 0,  # Direct pipeline doesn't extract entities separately
            'mappings_count': len(result.mcode_mappings),
            'compliance_score': result.validation_results.get('compliance_score', 0),
            'result': result,
            'error': None
        }
    except Exception as e:
        logger.error(f"❌ Direct pipeline failed: {e}")
        return {
            'success': False,
            'pipeline': 'direct_mcode',
            'entities_count': 0,
            'mappings_count': 0,
            'compliance_score': 0,
            'result': None,
            'error': str(e)
        }

def compare_pipelines(traditional_result: Dict[str, Any], direct_result: Dict[str, Any]) -> None:
    """Compare the results from both pipelines"""
    logger.info("=== Pipeline Comparison ===")
    
    # Success comparison
    logger.info(f"Traditional pipeline success: {traditional_result['success']}")
    logger.info(f"Direct pipeline success: {direct_result['success']}")
    
    if traditional_result['success'] and direct_result['success']:
        # Detailed comparison
        logger.info(f"📊 Traditional: {traditional_result['entities_count']} entities → {traditional_result['mappings_count']} mappings")
        logger.info(f"📊 Direct: Skip entities → {direct_result['mappings_count']} mappings")
        logger.info(f"🎯 Traditional compliance: {traditional_result['compliance_score']:.2%}")
        logger.info(f"🎯 Direct compliance: {direct_result['compliance_score']:.2%}")
        
        # Mapping count comparison
        mapping_diff = direct_result['mappings_count'] - traditional_result['mappings_count']
        if mapping_diff > 0:
            logger.info(f"✨ Direct pipeline generated {mapping_diff} more mappings (+{mapping_diff/traditional_result['mappings_count']*100:.1f}%)")
        elif mapping_diff < 0:
            logger.info(f"📉 Direct pipeline generated {abs(mapping_diff)} fewer mappings ({mapping_diff/traditional_result['mappings_count']*100:.1f}%)")
        else:
            logger.info("📊 Both pipelines generated the same number of mappings")
    
    elif direct_result['success'] and not traditional_result['success']:
        logger.info("✅ Direct pipeline succeeded where traditional pipeline failed!")
        logger.info(f"   🗺️  Generated {direct_result['mappings_count']} mappings")
        logger.info(f"   ❌ Traditional error: {traditional_result['error']}")
    
    elif traditional_result['success'] and not direct_result['success']:
        logger.info("⚠️  Traditional pipeline succeeded but direct pipeline failed")
        logger.info(f"   🗺️  Traditional generated {traditional_result['mappings_count']} mappings")
        logger.info(f"   ❌ Direct error: {direct_result['error']}")
    
    else:
        logger.info("❌ Both pipelines failed")
        logger.info(f"   Traditional error: {traditional_result['error']}")
        logger.info(f"   Direct error: {direct_result['error']}")

def save_results(traditional_result: Dict[str, Any], direct_result: Dict[str, Any]) -> None:
    """Save comparison results to file"""
    output_data = {
        'comparison_timestamp': '2025-09-07',
        'test_case': 'breast_cancer_her2_positive',
        'traditional_pipeline': {
            'success': traditional_result['success'],
            'entities_count': traditional_result['entities_count'],
            'mappings_count': traditional_result['mappings_count'],
            'compliance_score': traditional_result['compliance_score'],
            'error': traditional_result['error']
        },
        'direct_pipeline': {
            'success': direct_result['success'],
            'mappings_count': direct_result['mappings_count'],
            'compliance_score': direct_result['compliance_score'],
            'error': direct_result['error']
        }
    }
    
    # Add detailed mappings if successful
    if traditional_result['success'] and traditional_result['result']:
        output_data['traditional_pipeline']['sample_mappings'] = [
            {
                'mcode_element': mapping.get('mcode_element', 'Unknown'),
                'value': mapping.get('value', 'No value'),
                'confidence': mapping.get('mapping_confidence', 0)
            }
            for mapping in traditional_result['result'].mcode_mappings[:5]  # First 5 mappings
        ]
    
    if direct_result['success'] and direct_result['result']:
        output_data['direct_pipeline']['sample_mappings'] = [
            {
                'mcode_element': mapping.get('mcode_element', 'Unknown'),
                'value': mapping.get('value', 'No value'),
                'confidence': mapping.get('mapping_confidence', 0)
            }
            for mapping in direct_result['result'].mcode_mappings[:5]  # First 5 mappings
        ]
    
    output_file = Path("pipeline_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"📁 Comparison results saved to {output_file}")

def main():
    """Main function to test both pipelines and compare"""
    logger.info("🚀 Starting Pipeline Comparison Test")
    logger.info("=" * 60)
    
    try:
        # Load test data
        test_case = load_test_data()
        
        # Test traditional pipeline
        traditional_result = test_nlp_mcode_pipeline(test_case)
        
        # Test direct pipeline
        direct_result = test_direct_mcode_pipeline(test_case)
        
        # Compare results
        compare_pipelines(traditional_result, direct_result)
        
        # Save results
        save_results(traditional_result, direct_result)
        
        # Final recommendation
        logger.info("=" * 60)
        logger.info("🎯 RECOMMENDATION:")
        
        if direct_result['success'] and not traditional_result['success']:
            logger.info("✅ SWITCH TO DIRECT PIPELINE - Traditional pipeline is failing")
            logger.info("   The direct pipeline bypasses the entity extraction issues")
            return True
        elif direct_result['success'] and traditional_result['success']:
            if direct_result['mappings_count'] > traditional_result['mappings_count']:
                logger.info("✅ CONSIDER DIRECT PIPELINE - Generates more comprehensive mappings")
                logger.info(f"   Direct: {direct_result['mappings_count']} vs Traditional: {traditional_result['mappings_count']}")
            else:
                logger.info("🔍 EVALUATE BASED ON QUALITY - Both pipelines work, check mapping quality")
            return True
        elif traditional_result['success'] and not direct_result['success']:
            logger.info("⚠️  STICK WITH TRADITIONAL - Direct pipeline has issues")
            return False
        else:
            logger.info("❌ BOTH PIPELINES FAILING - Need to debug core issues")
            return False
        
    except Exception as e:
        logger.error(f"Error running comparison: {e}")
        logger.exception(e)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)