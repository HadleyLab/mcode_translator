#!/usr/bin/env python3
"""
Test API functionality: Working vs Not Working

This test demonstrates the difference between:
✅ Working API endpoints: /spaces, /webhooks
❌ Not Working API endpoints: /user/profile, /memory/*, /oauth2/*

Tests both direct API calls and client integration.
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol.client import HeySolClient
from heysol.exceptions import HeySolError


class APIWorkingVsNotTest:
    """Test class to demonstrate API working vs not working."""

    def __init__(self):
        """Initialize test class."""
        self.api_key = os.getenv('HEYSOL_API_KEY')
        if not self.api_key:
            raise ValueError("HEYSOL_API_KEY environment variable not set")

        self.base_url = "https://core.heysol.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def test_working_api_endpoints(self) -> Dict[str, Any]:
        """Test working API endpoints."""
        print("✅ Testing WORKING API Endpoints")
        print("=" * 50)

        working_endpoints = [
            {"path": "/spaces", "method": "GET", "description": "List spaces"},
            {"path": "/webhooks", "method": "GET", "description": "List webhooks"}
        ]

        results = {}

        for endpoint in working_endpoints:
            print(f"Testing {endpoint['method']} {endpoint['path']}")

            try:
                response = requests.request(
                    method=endpoint["method"],
                    url=f"{self.base_url}{endpoint['path']}",
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 200:
                    results[endpoint["path"]] = {
                        "status": "working",
                        "status_code": response.status_code,
                        "content_type": response.headers.get("Content-Type", ""),
                        "description": endpoint["description"]
                    }
                    print(f"  ✅ {endpoint['path']} - Working")
                else:
                    results[endpoint["path"]] = {
                        "status": "not_working",
                        "status_code": response.status_code,
                        "error": response.text[:200],
                        "description": endpoint["description"]
                    }
                    print(f"  ❌ {endpoint['path']} - HTTP {response.status_code}")

            except Exception as e:
                results[endpoint["path"]] = {
                    "status": "error",
                    "error": str(e),
                    "description": endpoint["description"]
                }
                print(f"  ❌ {endpoint['path']} - Error: {e}")

        return results

    def test_not_working_api_endpoints(self) -> Dict[str, Any]:
        """Test not working API endpoints."""
        print("\n❌ Testing NOT WORKING API Endpoints")
        print("=" * 50)

        not_working_endpoints = [
            {"path": "/user/profile", "method": "GET", "description": "User profile"},
            {"path": "/memory/search", "method": "GET", "description": "Memory search"},
            {"path": "/memory/ingest", "method": "POST", "description": "Memory ingest"},
            {"path": "/memory/logs", "method": "GET", "description": "Memory logs"},
            {"path": "/oauth2/authorize", "method": "GET", "description": "OAuth2 authorize"},
            {"path": "/oauth2/token", "method": "POST", "description": "OAuth2 token"}
        ]

        results = {}

        for endpoint in not_working_endpoints:
            print(f"Testing {endpoint['method']} {endpoint['path']}")

            try:
                response = requests.request(
                    method=endpoint["method"],
                    url=f"{self.base_url}{endpoint['path']}",
                    headers=self.headers,
                    timeout=10
                )

                if response.status_code == 404:
                    results[endpoint["path"]] = {
                        "status": "not_found",
                        "status_code": 404,
                        "description": endpoint["description"],
                        "note": "Endpoint does not exist"
                    }
                    print(f"  ❌ {endpoint['path']} - 404 Not Found")
                elif response.status_code == 200:
                    results[endpoint["path"]] = {
                        "status": "working",
                        "status_code": 200,
                        "description": endpoint["description"],
                        "note": "Unexpectedly working"
                    }
                    print(f"  ✅ {endpoint['path']} - Working")
                else:
                    results[endpoint["path"]] = {
                        "status": "other_error",
                        "status_code": response.status_code,
                        "error": response.text[:200],
                        "description": endpoint["description"]
                    }
                    print(f"  ⚠️ {endpoint['path']} - HTTP {response.status_code}")

            except Exception as e:
                results[endpoint["path"]] = {
                    "status": "error",
                    "error": str(e),
                    "description": endpoint["description"]
                }
                print(f"  ❌ {endpoint['path']} - Error: {e}")

        return results

    def test_client_with_working_api(self) -> Dict[str, Any]:
        """Test HeySolClient with working API endpoints."""
        print("\n🔧 Testing HeySolClient with API endpoints")
        print("=" * 50)

        try:
            client = HeySolClient(api_key=self.api_key)

            # Test working endpoints
            working_tests = {}

            # Test spaces
            try:
                spaces = client.get_spaces()
                working_tests["spaces"] = {
                    "status": "working",
                    "result": f"Found {len(spaces) if isinstance(spaces, list) else 'N/A'} spaces"
                }
                print("  ✅ Spaces API - Working")
            except Exception as e:
                working_tests["spaces"] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"  ❌ Spaces API - Failed: {e}")

            # Test memory operations
            try:
                result = client.search("test", limit=1)
                working_tests["memory_search"] = {
                    "status": "working",
                    "result": f"Search returned {type(result).__name__}"
                }
                print("  ✅ Memory Search - Working")
            except Exception as e:
                working_tests["memory_search"] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"  ❌ Memory Search - Failed: {e}")

            return {
                "status": "completed",
                "working_tests": working_tests
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API working vs not working tests."""
        print("🚀 API Working vs Not Working Test Suite")
        print("=" * 60)

        results = {
            "working_endpoints": self.test_working_api_endpoints(),
            "not_working_endpoints": self.test_not_working_api_endpoints(),
            "client_tests": self.test_client_with_working_api()
        }

        # Summary
        print("\n📊 SUMMARY")
        print("=" * 60)

        working = results["working_endpoints"]
        not_working = results["not_working_endpoints"]

        working_count = sum(1 for r in working.values() if r.get("status") == "working")
        not_working_count = sum(1 for r in not_working.values() if r.get("status") == "not_found")

        print(f"✅ Working API Endpoints: {working_count}")
        for path, result in working.items():
            if result.get("status") == "working":
                print(f"   {path} - {result.get('description', '')}")

        print(f"\n❌ Not Working API Endpoints: {not_working_count}")
        for path, result in not_working.items():
            if result.get("status") == "not_found":
                print(f"   {path} - {result.get('description', '')}")

        client_test = results["client_tests"]
        if client_test.get("status") == "completed":
            print("\n✅ Client Integration: Working")
            working_tests = client_test.get("working_tests", {})
            for test_name, test_result in working_tests.items():
                if test_result.get("status") == "working":
                    print(f"   {test_name} - {test_result.get('result', '')}")
        else:
            print(f"\n❌ Client Integration: Failed - {client_test.get('error', 'Unknown error')}")

        return results


def main():
    """Main function."""
    try:
        test = APIWorkingVsNotTest()
        results = test.run_all_tests()

        # Save results
        with open("api_working_vs_not_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to: api_working_vs_not_results.json")
        print("🎉 API Working vs Not Working test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()