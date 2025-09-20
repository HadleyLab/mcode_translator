#!/usr/bin/env python3
"""
Comprehensive Test Runner for HeySol API Client

Runs all test suites and provides detailed reporting on test results,
performance metrics, and recommendations.
"""

import os
import sys
import time
import json
import pytest
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import test suites
from test_comprehensive_authentication import TestAuthenticationMechanisms, TestOAuth2AdvancedFeatures
from test_comprehensive_endpoints import TestEndpointValidation, TestErrorHandling, TestDataSerialization
from test_load_performance import TestLoadPerformance, TestRateLimitHandling
from test_security_vulnerabilities import TestSecurityVulnerabilities, TestEdgeCases


class TestResult:
    """Class to store test results."""

    def __init__(self, test_name: str, status: str, duration: float, details: str = ""):
        self.test_name = test_name
        self.status = status  # "PASS", "FAIL", "ERROR", "SKIP"
        self.duration = duration
        self.details = details
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration": self.duration,
            "details": self.details,
            "timestamp": self.timestamp
        }


class ComprehensiveTestRunner:
    """Comprehensive test runner for HeySol API client."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None
        self.test_suites = {
            "authentication": TestAuthenticationMechanisms,
            "oauth2_advanced": TestOAuth2AdvancedFeatures,
            "endpoints": TestEndpointValidation,
            "error_handling": TestErrorHandling,
            "data_serialization": TestDataSerialization,
            "load_performance": TestLoadPerformance,
            "rate_limiting": TestRateLimitHandling,
            "security": TestSecurityVulnerabilities,
            "edge_cases": TestEdgeCases
        }

    def run_test(self, test_class, test_method_name: str) -> TestResult:
        """Run a single test method."""
        start_time = time.time()

        try:
            # Create test instance
            test_instance = test_class()
            test_instance.setup_method()

            # Get test method
            test_method = getattr(test_instance, test_method_name)

            # Run test
            test_method()

            duration = time.time() - start_time
            return TestResult(test_method_name, "PASS", duration)

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(test_method_name, "FAIL", duration, str(e))

    def run_test_suite(self, suite_name: str, test_class) -> List[TestResult]:
        """Run all tests in a test suite."""
        print(f"\n🧪 Running {suite_name} tests...")

        suite_results = []
        test_instance = test_class()

        # Get all test methods
        test_methods = [method for method in dir(test_instance)
                       if method.startswith('test_') and callable(getattr(test_instance, method))]

        for method_name in test_methods:
            result = self.run_test(test_class, method_name)
            suite_results.append(result)

            if result.status == "PASS":
                print(f"  ✅ {method_name}")
            else:
                print(f"  ❌ {method_name}: {result.details}")

        return suite_results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        self.start_time = time.time()
        print("🚀 Starting Comprehensive HeySol API Client Testing Suite")
        print("=" * 60)

        all_results = {}

        for suite_name, test_class in self.test_suites.items():
            suite_results = self.run_test_suite(suite_name, test_class)
            all_results[suite_name] = [result.to_dict() for result in suite_results]
            self.results.extend(suite_results)

        self.end_time = time.time()
        return all_results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        # Calculate statistics
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        avg_test_time = sum(r.duration for r in self.results) / total_tests if total_tests > 0 else 0

        # Group results by suite
        suite_stats = {}
        for suite_name, suite_results in results.items():
            suite_passed = len([r for r in suite_results if r["status"] == "PASS"])
            suite_total = len(suite_results)
            suite_stats[suite_name] = {
                "passed": suite_passed,
                "total": suite_total,
                "pass_rate": (suite_passed / suite_total * 100) if suite_total > 0 else 0
            }

        # Generate detailed report
        report = f"""
{'='*80}
COMPREHENSIVE HEYSOL API CLIENT TESTING REPORT
{'='*80}

EXECUTION SUMMARY
=================
Total Tests Run: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Pass Rate: {pass_rate:.1f}%
Total Execution Time: {total_time:.2f} seconds
Average Test Time: {avg_test_time:.3f} seconds

SUITE BREAKDOWN
===============
"""

        for suite_name, stats in suite_stats.items():
            suite_display = suite_name.replace('_', ' ').title()
            report += f"{suite_display:<15}: {stats['passed']:2d"2d"stats['total']:2d"2d"stats['pass_rate']:.1f"1f"\n"

        # Add detailed results
        report += f"""

DETAILED RESULTS
================
"""

        for suite_name, suite_results in results.items():
            report += f"\n{suite_name.upper()} TESTS\n{'-' * (len(suite_name) + 5)}\n"

            for result in suite_results:
                status_icon = "✅" if result["status"] == "PASS" else "❌"
                report += f"{status_icon} {result['test_name']:<40} ({result['duration']:.3f}s)\n"
                if result["status"] != "PASS" and result["details"]:
                    report += f"    Error: {result['details'][:100]}{'...' if len(result['details']) > 100 else ''}\n"

        # Add recommendations
        report += f"""

RECOMMENDATIONS
===============
"""

        if pass_rate >= 95:
            report += "🎉 EXCELLENT: All tests are passing! The API client is in excellent condition.\n"
        elif pass_rate >= 80:
            report += "✅ GOOD: Most tests are passing. Minor issues need attention.\n"
        elif pass_rate >= 60:
            report += "⚠️  MODERATE: Several tests are failing. Investigation and fixes needed.\n"
        else:
            report += "❌ CRITICAL: Many tests are failing. Major issues need immediate attention.\n"

        if failed_tests > 0:
            report += f"\n🔧 ACTION ITEMS:\n"
            report += f"   - Review {failed_tests} failing tests\n"
            report += f"   - Check error details in the detailed results above\n"
            report += f"   - Prioritize security and authentication fixes\n"
            report += f"   - Validate error handling mechanisms\n"

        # Add performance insights
        if avg_test_time > 1.0:
            report += f"\n⚡ PERFORMANCE NOTES:\n"
            report += f"   - Average test time is {avg_test_time:.3f} seconds\n"
            report += f"   - Consider optimizing slow-running tests\n"
            report += f"   - Check for performance bottlenecks in test setup\n"

        report += f"""
{'='*80}
Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        return report

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heysol_test_report_{timestamp}.json"

        # Create results directory if it doesn't exist
        results_dir = Path("heysol_api_client/tests")
        results_dir.mkdir(exist_ok=True)

        filepath = results_dir / filename

        # Prepare data for saving
        save_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "pass_rate": len([r for r in self.results if r.status == "PASS"]) / len(self.results) * 100 if self.results else 0,
                "execution_time": self.end_time - self.start_time if self.start_time and self.end_time else 0
            },
            "results": results
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"📊 Test results saved to: {filepath}")
        return filepath


def main():
    """Main function to run comprehensive tests."""
    print("🤖 HeySol API Client Comprehensive Testing Suite")
    print("This will run all test suites and provide detailed reporting.")

    # Check if user wants to continue
    try:
        input("\nPress Enter to start testing (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n❌ Testing cancelled by user.")
        return

    # Create and run test runner
    runner = ComprehensiveTestRunner()

    try:
        # Run all tests
        results = runner.run_all_tests()

        # Generate and display report
        report = runner.generate_report(results)
        print(report)

        # Save results
        runner.save_results(results)

        # Exit with appropriate code
        failed_tests = len([r for r in runner.results if r.status != "PASS"])
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} tests failed. Check the report for details.")
            sys.exit(1)
        else:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()