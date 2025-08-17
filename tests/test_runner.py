#!/usr/bin/env python3
"""
Main test runner for the mCODE Translator project
Runs all tests and provides comprehensive reporting
"""

import sys
import os
import pytest
import time
import json
from typing import Dict, Any


class TestRunner:
    """Main test runner for mCODE Translator"""
    
    def __init__(self):
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'component_tests': {},
            'performance_metrics': {},
            'coverage_report': {}
        }
    
    def run_all_tests(self) -> bool:
        """Run all tests and generate comprehensive report"""
        print("Running Comprehensive Test Suite for mCODE Translator")
        print("=" * 60)
        
        # Run unit tests
        unit_success = self.run_unit_tests()
        
        # Run integration tests
        integration_success = self.run_integration_tests()
        
        # Run component tests
        component_success = self.run_component_tests()
        
        # Collect performance metrics
        self.collect_performance_metrics()
        
        # Generate summary report
        self.generate_summary_report(unit_success, integration_success, component_success)
        
        return unit_success and integration_success and component_success
    
    def run_unit_tests(self) -> bool:
        """Run all unit tests"""
        print("\nRunning Unit Tests...")
        print("-" * 30)
        
        # Use pytest to run unit tests
        try:
            result = pytest.main([
                '-v',
                '--tb=short',
                'tests/unit/',
                '-m', 'not slow'
            ])
            success = result == 0
            self.results['unit_tests']['success'] = success
            return success
        except Exception as e:
            print(f"Error running unit tests: {e}")
            self.results['unit_tests']['success'] = False
            return False
    
    def run_integration_tests(self) -> bool:
        """Run all integration tests"""
        print("\nRunning Integration Tests...")
        print("-" * 30)
        
        # Use pytest to run integration tests
        try:
            result = pytest.main([
                '-v',
                '--tb=short',
                'tests/integration/',
                '-m', 'integration'
            ])
            success = result == 0
            self.results['integration_tests']['success'] = success
            return success
        except Exception as e:
            print(f"Error running integration tests: {e}")
            self.results['integration_tests']['success'] = False
            return False
    
    def run_component_tests(self) -> bool:
        """Run all component tests"""
        print("\nRunning Component Tests...")
        print("-" * 30)
        
        # Use pytest to run component tests
        try:
            result = pytest.main([
                '-v',
                '--tb=short',
                'tests/component/',
                '-m', 'component'
            ])
            success = result == 0
            self.results['component_tests']['success'] = success
            return success
        except Exception as e:
            print(f"Error running component tests: {e}")
            self.results['component_tests']['success'] = False
            return False
    
    def collect_performance_metrics(self):
        """Collect performance metrics from test execution"""
        print("\nCollecting Performance Metrics...")
        print("-" * 30)
        
        try:
            # Import engines for benchmarking
            from src.nlp_engine.regex_nlp_engine import RegexNLPEngine
            from src.nlp_engine.spacy_nlp_engine import SpacyNLPEngine
            
            # Test text sample
            test_text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer, stage IIA"
            num_runs = 5  # Number of runs for averaging
            
            def benchmark_engine(engine, method_name, text, runs):
                method = getattr(engine, method_name)
                start = time.time()
                for _ in range(runs):
                    method(text)
                return (time.time() - start) / runs
            
            # Benchmark engines
            print("Benchmarking NLP Engines...")
            
            # Test Regex Engine
            regex_engine = RegexNLPEngine()
            regex_time = benchmark_engine(regex_engine, 'process_criteria', test_text, num_runs)
            print(f"- Regex NLP Engine: {regex_time*1000:.2f}ms avg")
            
            # Test SpaCy Engine
            spacy_engine = SpacyNLPEngine()
            spacy_time = benchmark_engine(spacy_engine, 'process_criteria', test_text, num_runs)
            print(f"- SpaCy NLP Engine: {spacy_time*1000:.2f}ms avg")
            
            # Store performance metrics
            self.results['performance_metrics'] = {
                'regex_engine_avg_time': regex_time,
                'spacy_engine_avg_time': spacy_time,
                'test_text_length': len(test_text),
                'benchmark_runs': num_runs
            }
        except Exception as e:
            print(f"Error collecting performance metrics: {e}")
    
    def generate_summary_report(self, unit_success: bool, integration_success: bool, component_success: bool):
        """Generate summary report of test execution"""
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)
        
        # Test execution summary
        print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
        print(f"Integration Tests: {'PASSED' if integration_success else 'FAILED'}")
        print(f"Component Tests: {'PASSED' if component_success else 'FAILED'}")
        
        # Performance metrics
        if self.results['performance_metrics']:
            perf = self.results['performance_metrics']
            print(f"\nPerformance Metrics:")
            print(f"- Regex Engine: {perf['regex_engine_avg_time']*1000:.2f}ms")
            print(f"- SpaCy Engine: {perf['spacy_engine_avg_time']*1000:.2f}ms")
        
        # Overall status
        all_passed = unit_success and integration_success and component_success
        print(f"\nOverall Status: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        
        # Save results to file
        with open('test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("\nDetailed results saved to test_results.json")


def main():
    """Main function"""
    runner = TestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()