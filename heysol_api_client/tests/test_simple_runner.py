#!/usr/bin/env python3
"""
Simple Test Runner for HeySol API Client

A simplified test runner that focuses on core functionality testing.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import test suites
from test_comprehensive_authentication import TestAuthenticationMechanisms
from test_comprehensive_endpoints import TestEndpointValidation, TestErrorHandling
from test_security_vulnerabilities import TestSecurityVulnerabilities


class SimpleTestRunner:
    """Simple test runner for HeySol API client."""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None

    def run_test(self, test_class, test_method_name):
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
            result = {
                "test_name": test_method_name,
                "status": "PASS",
                "duration": duration,
                "details": ""
            }
            print(f"✅ {test_method_name}")
            return result

        except Exception as e:
            duration = time.time() - start_time
            result = {
                "test_name": test_method_name,
                "status": "FAIL",
                "duration": duration,
                "details": str(e)
            }
            print(f"❌ {test_method_name}: {str(e)}")
            return result

    def run_test_suite(self, suite_name, test_class):
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
            self.results.append(result)

        return suite_results

    def run_all_tests(self):
        """Run all test suites."""
        self.start_time = time.time()
        print("🚀 Starting HeySol API Client Testing")
        print("=" * 50)

        test_suites = [
            ("Authentication", TestAuthenticationMechanisms),
            ("Endpoints", TestEndpointValidation),
            ("Error Handling", TestErrorHandling),
            ("Security", TestSecurityVulnerabilities)
        ]

        all_results = {}

        for suite_name, test_class in test_suites:
            suite_results = self.run_test_suite(suite_name, test_class)
            all_results[suite_name] = suite_results

        self.end_time = time.time()
        return all_results

    def generate_report(self, results):
        """Generate test report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.results if r["status"] == "FAIL"])
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0

        # Calculate pass rate
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Generate report
        report = "=" * 60 + "\n"
        report += "HEYSOL API CLIENT TESTING REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "EXECUTION SUMMARY\n"
        report += "-" * 20 + "\n"
        report += f"Total Tests Run: {total_tests}\n"
        report += f"Passed: {passed_tests}\n"
        report += f"Failed: {failed_tests}\n"
        report += f"Pass Rate: {pass_rate".1f"}%\n"
        report += f"Total Execution Time: {total_time".2f"} seconds\n\n"

        # Suite breakdown
        report += "SUITE BREAKDOWN\n"
        report += "-" * 15 + "\n"

        for suite_name, suite_results in results.items():
            suite_passed = len([r for r in suite_results if r["status"] == "PASS"])
            suite_total = len(suite_results)
            suite_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0
            report += f"{suite_name"<15"}: {suite_passed"2d"}/{suite_total"2d"} ({suite_rate:".1f")\n"

        # Detailed results
        report += "\nDETAILED RESULTS\n"
        report += "-" * 16 + "\n"

        for suite_name, suite_results in results.items():
            report += f"\n{suite_name.upper()} TESTS\n"
            report += "-" * (len(suite_name) + 5) + "\n"

            for result in suite_results:
                status_icon = "✅" if result["status"] == "PASS" else "❌"
                report += f"{status_icon} {result['test_name']"<40"} ({result['duration']".3f"}s)\n"
                if result["status"] != "PASS" and result["details"]:
                    report += f"    Error: {result['details'][:100]}\n"

        # Recommendations
        report += "\nRECOMMENDATIONS\n"
        report += "-" * 13 + "\n"

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
            report += f"   - Check error details above\n"
            report += f"   - Prioritize security and authentication fixes\n"

        report += f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 60 + "\n"

        return report


def main():
    """Main function to run tests."""
    print("🤖 HeySol API Client Testing Suite")
    print("This will run comprehensive tests on the API client.")

    try:
        input("\nPress Enter to start testing (or Ctrl+C to cancel)...")
    except KeyboardInterrupt:
        print("\n❌ Testing cancelled by user.")
        return

    # Create and run test runner
    runner = SimpleTestRunner()

    try:
        # Run all tests
        results = runner.run_all_tests()

        # Generate and display report
        report = runner.generate_report(results)
        print(report)

        # Exit with appropriate code
        failed_tests = len([r for r in runner.results if r["status"] != "PASS"])
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} tests failed.")
            sys.exit(1)
        else:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)

    except Exception as e:
        print(f"\n💥 Critical error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()