#!/usr/bin/env python3
"""
Evidence-Based mCODE Prompt Validation Test
Compares the new evidence-based prompt against previous prompts to validate improved quality.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the testing environment"""
    import sys
    import os
    
    # Add src to path for imports
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    
    # Set environment for API usage
    if not os.getenv('DEEPSEEK_API_KEY'):
        logger.warning("DEEPSEEK_API_KEY not set - may cause API failures")

def load_test_trials():
    """Load the selected breast cancer trials for testing"""
    try:
        with open('selected_breast_cancer_trials.json', 'r') as f:
            data = json.load(f)
        
        trials = data.get('successful_trials', [])
        logger.info(f"Loaded {len(trials)} test trials")
        return trials  # Use all 5 trials for comprehensive validation
    except Exception as e:
        logger.error(f"Failed to load test trials: {e}")
        return []

def run_single_extraction(trial_data: Dict[str, Any], model: str, prompt: str) -> Dict[str, Any]:
    """Run mCODE extraction on a single trial with specified model and prompt"""
    try:
        from src.pipeline.mcode_pipeline import McodePipeline
        
        # Initialize pipeline
        pipeline = McodePipeline(
            model_name=model,
            prompt_name=prompt
        )
        
        # Process trial
        start_time = time.time()
        result = pipeline.process_clinical_trial(trial_data)
        duration = time.time() - start_time
        
        # Convert PipelineResult to dict format
        result_dict = {
            'trial_id': trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown'),
            'processing_time': duration,
            'mcode_mappings': result.mcode_mappings,
            'source_references': result.source_references,
            'validation_results': result.validation_results,
            'metadata': result.metadata
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Extraction failed for {model}/{prompt}: {e}")
        return {
            'error': str(e),
            'trial_id': trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown'),
            'processing_time': 0,
            'mcode_mappings': []
        }

def analyze_mapping_quality(mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the quality of mCODE mappings"""
    if not mappings:
        return {
            'total_mappings': 0,
            'avg_confidence': 0.0,
            'high_confidence_count': 0,
            'with_source_text': 0,
            'source_text_ratio': 0.0,
            'confidence_distribution': {},
            'quality_score': 0.0
        }
    
    # Calculate metrics
    total = len(mappings)
    confidences = [m.get('mapping_confidence', 0.0) for m in mappings]
    avg_confidence = sum(confidences) / total if total > 0 else 0.0
    high_confidence = sum(1 for c in confidences if c >= 0.8)
    
    # Check source text quality
    with_actual_source = sum(
        1 for m in mappings 
        if m.get('source_text_fragment') and 
        m.get('source_text_fragment') != 'Entity index unknown' and
        not m.get('source_text_fragment').startswith('Entity index') and
        len(m.get('source_text_fragment', '').strip()) > 10
    )
    source_text_ratio = with_actual_source / total if total > 0 else 0.0
    
    # Confidence distribution
    conf_dist = {
        'high (0.8-1.0)': sum(1 for c in confidences if c >= 0.8),
        'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c < 0.8),
        'low (0.0-0.5)': sum(1 for c in confidences if c < 0.5)
    }
    
    # Calculate composite quality score
    quality_score = (
        (avg_confidence * 0.4) +  # 40% weight on confidence
        (source_text_ratio * 0.4) +  # 40% weight on source text quality
        (high_confidence / total * 0.2) if total > 0 else 0.0  # 20% weight on high confidence ratio
    )
    
    return {
        'total_mappings': total,
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_confidence,
        'with_source_text': with_actual_source,
        'source_text_ratio': source_text_ratio,
        'confidence_distribution': conf_dist,
        'quality_score': quality_score
    }

def validate_evidence_based_improvements(results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the evidence-based prompt shows improvements"""
    validation = {
        'improvements_detected': [],
        'concerns_identified': [],
        'overall_assessment': 'unknown'
    }
    
    # Get results for comparison
    evidence_results = results.get('direct_mcode_evidence_based_concise', {})
    simple_results = results.get('direct_mcode_simple', {})
    
    if not evidence_results or not simple_results:
        validation['concerns_identified'].append("Missing comparison data")
        return validation
    
    evidence_quality = evidence_results.get('quality_analysis', {})
    simple_quality = simple_results.get('quality_analysis', {})
    
    # Check source text improvements
    if evidence_quality.get('source_text_ratio', 0) > simple_quality.get('source_text_ratio', 0):
        validation['improvements_detected'].append(
            f"Source text quality improved: {evidence_quality.get('source_text_ratio', 0):.2f} vs {simple_quality.get('source_text_ratio', 0):.2f}"
        )
    
    # Check confidence improvements
    if evidence_quality.get('avg_confidence', 0) > simple_quality.get('avg_confidence', 0):
        validation['improvements_detected'].append(
            f"Average confidence improved: {evidence_quality.get('avg_confidence', 0):.2f} vs {simple_quality.get('avg_confidence', 0):.2f}"
        )
    
    # Check mapping count (should be lower but higher quality)
    evidence_count = evidence_quality.get('total_mappings', 0)
    simple_count = simple_quality.get('total_mappings', 0)
    if evidence_count < simple_count:
        validation['improvements_detected'].append(
            f"Reduced over-mapping: {evidence_count} vs {simple_count} mappings (quality over quantity)"
        )
    elif evidence_count > simple_count * 1.5:
        validation['concerns_identified'].append(
            f"Potential over-mapping: {evidence_count} vs {simple_count} mappings"
        )
    
    # Overall quality score
    evidence_score = evidence_quality.get('quality_score', 0)
    simple_score = simple_quality.get('quality_score', 0)
    if evidence_score > simple_score:
        validation['improvements_detected'].append(
            f"Overall quality score improved: {evidence_score:.3f} vs {simple_score:.3f}"
        )
    
    # Determine overall assessment
    if len(validation['improvements_detected']) >= 2 and len(validation['concerns_identified']) == 0:
        validation['overall_assessment'] = 'significant_improvement'
    elif len(validation['improvements_detected']) > len(validation['concerns_identified']):
        validation['overall_assessment'] = 'moderate_improvement'
    elif len(validation['concerns_identified']) > len(validation['improvements_detected']):
        validation['overall_assessment'] = 'needs_work'
    else:
        validation['overall_assessment'] = 'mixed_results'
    
    return validation

def main():
    """Main validation function"""
    setup_environment()
    
    logger.info("🧪 Starting Evidence-Based mCODE Prompt Validation")
    logger.info("=" * 60)
    
    # Load test data
    trials = load_test_trials()
    if not trials:
        logger.error("No test trials available")
        return
    
    # Test configuration
    test_model = "deepseek-coder"
    prompts_to_test = [
        "direct_mcode_simple",  # Current best performer
        "direct_mcode_evidence_based_concise"  # New concise evidence-based prompt
    ]
    
    # Results storage
    results = {}
    
    # Run tests
    for prompt in prompts_to_test:
        logger.info(f"🔬 Testing prompt: {prompt}")
        prompt_results = {
            'prompt_name': prompt,
            'model_name': test_model,
            'trials_processed': 0,
            'total_processing_time': 0,
            'trial_results': [],
            'quality_analysis': {}
        }
        
        all_mappings = []
        
        for trial in trials:
            trial_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')
            logger.info(f"  Processing trial: {trial_id}")
            
            # Run extraction
            result = run_single_extraction(trial, test_model, prompt)
            prompt_results['trial_results'].append(result)
            prompt_results['trials_processed'] += 1
            prompt_results['total_processing_time'] += result.get('processing_time', 0)
            
            # Collect mappings for analysis
            mappings = result.get('mcode_mappings', [])
            all_mappings.extend(mappings)
            
            logger.info(f"    Generated {len(mappings)} mappings in {result.get('processing_time', 0):.2f}s")
        
        # Analyze quality
        prompt_results['quality_analysis'] = analyze_mapping_quality(all_mappings)
        results[prompt] = prompt_results
        
        logger.info(f"  Quality Score: {prompt_results['quality_analysis']['quality_score']:.3f}")
        logger.info(f"  Source Text Ratio: {prompt_results['quality_analysis']['source_text_ratio']:.3f}")
        logger.info(f"  Avg Confidence: {prompt_results['quality_analysis']['avg_confidence']:.3f}")
    
    # Validate improvements
    logger.info("📊 Analyzing Evidence-Based Improvements")
    validation = validate_evidence_based_improvements(results)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"evidence_based_validation_report_{timestamp}.json"
    
    report = {
        'validation_metadata': {
            'timestamp': timestamp,
            'test_model': test_model,
            'prompts_tested': prompts_to_test,
            'trials_count': len(trials),
            'validation_version': '1.0'
        },
        'results': results,
        'validation_analysis': validation,
        'recommendations': generate_recommendations(validation, results)
    }
    
    # Save report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("🎯 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for improvement in validation['improvements_detected']:
        logger.info(f"✅ {improvement}")
    
    for concern in validation['concerns_identified']:
        logger.info(f"⚠️  {concern}")
    
    logger.info(f"📋 Overall Assessment: {validation['overall_assessment']}")
    logger.info(f"📄 Detailed report saved: {report_file}")
    
    return validation['overall_assessment'] in ['significant_improvement', 'moderate_improvement']

def generate_recommendations(validation: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on validation results"""
    recommendations = []
    
    assessment = validation.get('overall_assessment', 'unknown')
    
    if assessment == 'significant_improvement':
        recommendations.append("✅ Deploy evidence-based prompt for production use")
        recommendations.append("✅ Update default prompt configuration")
        recommendations.append("✅ Consider this as new quality benchmark")
    
    elif assessment == 'moderate_improvement':
        recommendations.append("✅ Evidence-based prompt shows promise")
        recommendations.append("🔧 Consider additional refinements based on specific improvements")
        recommendations.append("🧪 Run larger-scale validation before production deployment")
    
    elif assessment == 'needs_work':
        recommendations.append("🔧 Evidence-based prompt needs significant improvements")
        recommendations.append("📋 Review specific concerns and iterate on prompt design")
        recommendations.append("🚫 Do not deploy in current form")
    
    else:
        recommendations.append("🧪 Results are mixed - requires manual analysis")
        recommendations.append("📊 Consider additional test cases and metrics")
    
    # Add specific technical recommendations
    evidence_results = results.get('direct_mcode_evidence_based_concise', {})
    if evidence_results:
        quality = evidence_results.get('quality_analysis', {})
        if quality.get('source_text_ratio', 0) < 0.5:
            recommendations.append("🔧 Improve source text fragment capture mechanism")
        if quality.get('avg_confidence', 0) < 0.7:
            recommendations.append("🔧 Refine confidence scoring calibration")
    
    return recommendations

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)