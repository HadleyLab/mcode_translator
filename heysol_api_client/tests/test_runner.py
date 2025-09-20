"""
Comprehensive test runner for HeySol API client testing suite.
"""

import pytest
import sys
import os
import time
from datetime import datetime
import json
from typing import Dict, List, Any
import subprocess


class TestRunner:
    """Comprehensive test runner for HeySol API client."""

    def __init__(self):
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "test_suite": "HeySol API Client Testing Suite",
            "categories": {},
            "summary": {},
            "findings": [],
            "recommendations": []
        }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories and collect results."""
        print("🚀 Starting HeySol API Client Testing Suite")
        print("=" * 60)

        test_categories = [
            ("Authentication Tests", "test_authentication.py"),
            ("Valid Endpoint Tests", "test_endpoints_valid.py"),
            ("Invalid Endpoint Tests", "test_endpoints_invalid.py"),
            ("Error Handling Tests", "test_error_handling.py"),
            ("Load & Performance Tests", "test_load_performance.py"),
            ("Serialization Tests", "test_serialization.py"),
            ("Documentation Compliance Tests", "test_documentation_compliance.py"),
            ("Existing Tests", "test_config.py test_exceptions.py")
        ]

        total_passed = 0
        total_failed = 0
        total_errors = 0

        for category_name, test_files in test_categories:
            print(f"\n📋 Running {category_name}")
            print("-" * 40)

            category_results = self._run_test_category(test_files)

            self.test_results["categories"][category_name] = category_results

            passed = category_results.get("passed", 0)
            failed = category_results.get("failed", 0)
            errors = category_results.get("errors", 0)

            total_passed += passed
            total_failed += failed
            total_errors += errors

            print(f"✅ Passed: {passed}")
            print(f"❌ Failed: {failed}")
            print(f"⚠️  Errors: {errors}")

        # Generate summary
        self.test_results["summary"] = {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "total_tests": total_passed + total_failed + total_errors,
            "success_rate": (total_passed / (total_passed + total_failed + total_errors)) * 100 if (total_passed + total_failed + total_errors) > 0 else 0
        }

        # Analyze findings and generate recommendations
        self._analyze_findings()
        self._generate_recommendations()

        return self.test_results

    def _run_test_category(self, test_files: str) -> Dict[str, Any]:
        """Run a specific test category."""
        try:
            # Run pytest programmatically
            result = subprocess.run(
                ["python", "-m", "pytest", test_files, "-v", "--tb=short", "--json-report"],
                cwd="heysol_api_client/tests",
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse results (simplified - in practice you'd parse JSON output)
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.stdout.count("PASSED") if "PASSED" in result.stdout else 0,
                "failed": result.stdout.count("FAILED") if "FAILED" in result.stdout else 0,
                "errors": result.stdout.count("ERROR") if "ERROR" in result.stdout else 0
            }

        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "error": "Test timeout",
                "passed": 0,
                "failed": 0,
                "errors": 1
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "error": str(e),
                "passed": 0,
                "failed": 0,
                "errors": 1
            }

    def _analyze_findings(self):
        """Analyze test results and identify key findings."""
        findings = []

        # Check authentication test results
        auth_results = self.test_results["categories"].get("Authentication Tests", {})
        if auth_results.get("failed", 0) > 0:
            findings.append({
                "category": "Security",
                "severity": "High",
                "issue": "Authentication mechanism failures detected",
                "description": f"{auth_results.get('failed', 0)} authentication tests failed",
                "impact": "Potential security vulnerabilities in API access control"
            })

        # Check error handling results
        error_results = self.test_results["categories"].get("Error Handling Tests", {})
        if error_results.get("failed", 0) > 0:
            findings.append({
                "category": "Reliability",
                "severity": "Medium",
                "issue": "Error handling issues detected",
                "description": f"{error_results.get('failed', 0)} error handling tests failed",
                "impact": "Application may not handle API errors gracefully"
            })

        # Check performance results
        perf_results = self.test_results["categories"].get("Load & Performance Tests", {})
        if perf_results.get("failed", 0) > 0:
            findings.append({
                "category": "Performance",
                "severity": "Medium",
                "issue": "Performance issues detected",
                "description": f"{perf_results.get('failed', 0)} performance tests failed",
                "impact": "Application may have performance bottlenecks under load"
            })

        # Check serialization results
        serial_results = self.test_results["categories"].get("Serialization Tests", {})
        if serial_results.get("failed", 0) > 0:
            findings.append({
                "category": "Data Integrity",
                "severity": "High",
                "issue": "Data serialization/deserialization issues",
                "description": f"{serial_results.get('failed', 0)} serialization tests failed",
                "impact": "Data may be corrupted or improperly formatted"
            })

        # Check documentation compliance
        doc_results = self.test_results["categories"].get("Documentation Compliance Tests", {})
        if doc_results.get("failed", 0) > 0:
            findings.append({
                "category": "Documentation",
                "severity": "Low",
                "issue": "Documentation compliance issues",
                "description": f"{doc_results.get('failed', 0)} documentation tests failed",
                "impact": "API documentation may be outdated or inaccurate"
            })

        self.test_results["findings"] = findings

    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []

        summary = self.test_results.get("summary", {})

        if summary.get("success_rate", 0) < 80:
            recommendations.append({
                "priority": "High",
                "category": "Testing",
                "recommendation": "Improve test coverage and fix failing tests",
                "rationale": f"Current success rate is {summary.get('success_rate', 0):.1f}%, which is below acceptable threshold",
                "estimated_effort": "High"
            })

        # Authentication recommendations
        if self.test_results["categories"].get("Authentication Tests", {}).get("failed", 0) > 0:
            recommendations.append({
                "priority": "Critical",
                "category": "Security",
                "recommendation": "Review and fix authentication mechanisms",
                "rationale": "Authentication failures can lead to unauthorized access",
                "estimated_effort": "High"
            })

        # Error handling recommendations
        if self.test_results["categories"].get("Error Handling Tests", {}).get("failed", 0) > 0:
            recommendations.append({
                "priority": "High",
                "category": "Reliability",
                "recommendation": "Improve error handling and recovery mechanisms",
                "rationale": "Better error handling improves application robustness",
                "estimated_effort": "Medium"
            })

        # Performance recommendations
        if self.test_results["categories"].get("Load & Performance Tests", {}).get("failed", 0) > 0:
            recommendations.append({
                "priority": "Medium",
                "category": "Performance",
                "recommendation": "Optimize performance bottlenecks",
                "rationale": "Performance issues can affect user experience",
                "estimated_effort": "Medium"
            })

        # Serialization recommendations
        if self.test_results["categories"].get("Serialization Tests", {}).get("failed", 0) > 0:
            recommendations.append({
                "priority": "High",
                "category": "Data Integrity",
                "recommendation": "Fix data serialization and validation issues",
                "rationale": "Data integrity is critical for API reliability",
                "estimated_effort": "Medium"
            })

        # Documentation recommendations
        if self.test_results["categories"].get("Documentation Compliance Tests", {}).get("failed", 0) > 0:
            recommendations.append({
                "priority": "Low",
                "category": "Documentation",
                "recommendation": "Update and synchronize documentation",
                "rationale": "Accurate documentation improves developer experience",
                "estimated_effort": "Low"
            })

        # General recommendations
        recommendations.extend([
            {
                "priority": "Medium",
                "category": "Monitoring",
                "recommendation": "Implement comprehensive logging and monitoring",
                "rationale": "Better observability aids in debugging and maintenance",
                "estimated_effort": "Medium"
            },
            {
                "priority": "Medium",
                "category": "Testing",
                "recommendation": "Add integration tests with real API endpoints",
                "rationale": "Integration tests validate end-to-end functionality",
                "estimated_effort": "High"
            },
            {
                "priority": "Low",
                "category": "Code Quality",
                "recommendation": "Add type hints and improve code documentation",
                "rationale": "Better code quality reduces bugs and improves maintainability",
                "estimated_effort": "Medium"
            }
        ])

        self.test_results["recommendations"] = recommendations

    def save_report(self, filename: str = None):
        """Save test results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heysol_test_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n📄 Test report saved to: {filename}")

    def print_summary(self):
        """Print a summary of test results."""
        summary = self.test_results.get("summary", {})

        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"✅ Passed: {summary.get('total_passed', 0)}")
        print(f"❌ Failed: {summary.get('total_failed', 0)}")
        print(f"⚠️  Errors: {summary.get('total_errors', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print("\n🔍 KEY FINDINGS:")
        for finding in self.test_results.get("findings", []):
            print(f"• {finding['severity']}: {finding['issue']}")

        print("\n💡 RECOMMENDATIONS:")
        for rec in self.test_results.get("recommendations", []):
            print(f"• {rec['priority']}: {rec['recommendation']}")


def main():
    """Main entry point for the test runner."""
    runner = TestRunner()

    try:
        results = runner.run_all_tests()
        runner.print_summary()
        runner.save_report()

        # Exit with appropriate code
        success_rate = results["summary"].get("success_rate", 0)
        if success_rate >= 90:
            print("\n🎉 All tests passed with excellent results!")
            sys.exit(0)
        elif success_rate >= 75:
            print("\n👍 Tests completed with good results.")
            sys.exit(0)
        else:
            print("\n⚠️  Tests completed but with concerning results.")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()