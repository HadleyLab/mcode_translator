#!/usr/bin/env python3
"""
Main test runner for the mCODE Translator project
Runs all unit tests and provides a comprehensive test report
"""

import sys
import os
import unittest
import subprocess

def run_tests():
    """Run all tests using unittest discovery"""
    print("Running Comprehensive Test Suite for mCODE Translator")
    print("=" * 60)
    
    # Add src directory to PYTHONPATH
    sys.path.insert(0, os.path.abspath('src'))
    
    # Use unittest discovery to find all tests in tests/ directory and subdirectories
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Create test runner with detailed output
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Collect engine performance metrics
    import time
    from src.llm_nlp_engine import LLMNLPEngine
    from src.regex_nlp_engine import RegexNLPEngine
    from src.spacy_nlp_engine import SpacyNLPEngine
    
    # Test text sample
    test_text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer, stage IIA"
    num_runs = 5  # Number of runs for averaging
    
    def benchmark(engine, method_name):
        method = getattr(engine, method_name)
        start = time.time()
        for _ in range(num_runs):
            method(test_text)
        return (time.time() - start) / num_runs
    
    # Benchmark engines
    print("\nBenchmarking NLP Engines...")
    
    # Test Regex Engine
    regex_engine = RegexNLPEngine()
    regex_result = regex_engine.process_criteria(test_text)
    regex_time = benchmark(regex_engine, 'process_criteria')
    print("\nRegex NLP Engine Performance:")
    print(f"- Processing time: {regex_time:.4f} seconds")
    print(f"- Text length: {len(test_text)} characters")
    print(f"- Conditions found: {1 if regex_result.features.get('cancer_characteristics', {}).get('cancer_type') else 0}")
    print(f"- Procedures found: {len(regex_result.features.get('treatment_history', {}).get('procedures', []))}")
    
    # Test SpaCy Engine
    spacy_engine = SpacyNLPEngine()
    spacy_result = spacy_engine.process_criteria(test_text)
    spacy_time = benchmark(spacy_engine, 'process_criteria')
    print("\nSpaCy NLP Engine Performance:")
    print(f"- Processing time: {spacy_time:.4f} seconds")
    print(f"- Text length: {len(test_text)} characters")
    print(f"- Entities found: {len(spacy_result.entities)}")
    print(f"- Average confidence: {sum(e.get('confidence', 0) for e in spacy_result.entities)/len(spacy_result.entities) if spacy_result.entities else 0:.2f}")
    
    # Test LLM Engine
    llm_engine = LLMNLPEngine()
    llm_result = llm_engine.extract_mcode_features(test_text)
    llm_time = benchmark(llm_engine, 'extract_mcode_features')
    print("\nLLM NLP Engine Performance:")
    print(f"- Processing time: {llm_time:.4f} seconds")
    print(f"- Text length: {len(test_text)} characters")
    print(f"- Genomic variants found: {len(llm_result.features.get('genomic_variants', []))}")
    print(f"- Biomarkers found: {len(llm_result.features.get('biomarkers', []))}")
    
    # Print comparative metrics
    print("\nComparative Performance Metrics (avg over 5 runs):")
    print("-" * 60)
    print(f"Regex NLP Engine: {regex_time*1000:.2f}ms")
    print(f"SpaCy NLP Engine: {spacy_time*1000:.2f}ms")
    print(f"LLM NLP Engine: {llm_time*1000:.2f}ms")
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
        return True
    else:
        print("\nTest failures/errors detected")
        return False

def main():
    """Main function"""
    success = run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()