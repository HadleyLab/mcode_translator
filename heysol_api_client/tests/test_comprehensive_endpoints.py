#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing for HeySol API Client.

This script tests all documented endpoints systematically:
- USER endpoints
- MEMORY endpoints
- SPACES endpoints

Usage:
    python test_comprehensive_endpoints.py

Requirements:
    - Set HEYSOL_API_KEY in environment
    - Install required dependencies: pip install requests python-dotenv
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import HeySol client
from heysol.client import HeySolClient


class ComprehensiveEndpointTester:
    """Test all HeySol API endpoints systematically."""

    def __init__(self):
        """Initialize the endpoint tester."""
        self.api_key = os.getenv("HEYSOL_API_KEY")
        self.client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        self.client_secret = os.getenv("HEYSOL_OAUTH2_CLIENT_SECRET")
        self.base_url = "https://core.heysol.ai/api/v1"
        self.protocol = "https"
        self.domain = "core.heysol.ai"

        # Test data
        self.test_space_id = "test-space-123"
        self.test_episode_id = "test-episode-123"
        self.test_log_id = "test-log-123"
        self.test_statement_id = "test-statement-123"

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "api_key_available": bool(self.api_key),
            "base_url": self.base_url,
            "endpoints": {}
        }

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None,
                    headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with proper error handling."""
        url = endpoint.replace("{protocol}", self.protocol).replace("{domain}", self.domain)

        # Default headers
        request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        # Merge with custom headers
        if headers:
            request_headers.update(headers)

        try:
            start_time = time.time()

            if method.upper() == "GET":
                response = requests.get(url, headers=request_headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=request_headers, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=request_headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=request_headers, json=data, timeout=30)
            else:
                return {
                    "error": f"Unsupported method: {method}",
                    "status_code": None,
                    "response_time": 0
                }

            response_time = time.time() - start_time

            return {
                "status_code": response.status_code,
                "response_time": round(response_time * 1000, 2),  # Convert to milliseconds
                "response_text": response.text[:1000],  # Limit response text
                "headers": dict(response.headers),
                "error": None
            }

        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "status_code": None,
                "response_time": 0
            }

    def test_user_endpoints(self) -> Dict[str, Any]:
        """Test USER endpoints."""
        print("\n👤 Testing USER Endpoints")
        print("=" * 50)

        endpoints = {
            "GET_profile": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/profile",
                "data": None
            }
        }

        results = {}
        for name, config in endpoints.items():
            print(f"Testing {config['method']} {config['endpoint']}...")
            result = self.make_request(config['method'], config['endpoint'], config['data'])
            results[name] = result

            if result.get('status_code'):
                print(f"  ✅ {result['status_code']} ({result['response_time']}ms)")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")

        return results

    def test_oauth2_endpoints(self) -> Dict[str, Any]:
        """Test OAuth2 endpoints."""
        print("\n🔐 Testing OAuth2 Endpoints")
        print("=" * 50)

        endpoints = {
            "GET_oauth_authorize": {
                "method": "GET",
                "endpoint": "https://core.heysol.ai/oauth/authorize",
                "data": None
            },
            "POST_oauth_authorize": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/oauth/authorize",
                "data": {
                    "action": "allow",
                    "client_id": self.client_id,
                    "redirect_uri": "http://localhost:8080/callback",
                    "scope": "openid profile email",
                    "state": "test-state-123",
                    "code_challenge": "test-challenge-123",
                    "code_challenge_method": "S256"
                },
                "headers": {"Content-Type": "application/x-www-form-urlencoded"}
            },
            "POST_oauth_token": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/oauth/token",
                "data": {
                    "grant_type": "authorization_code",
                    "code": "test-auth-code-123",
                    "redirect_uri": "http://localhost:8080/callback",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code_verifier": "test-verifier-123",
                    "refresh_token": "test-refresh-token-123"
                }
            },
            "GET_oauth_userinfo": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/oauth/userinfo",
                "data": None
            },
            "GET_oauth_tokeninfo": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/oauth/tokeninfo",
                "data": None
            }
        }

        results = {}
        for name, config in endpoints.items():
            print(f"Testing {config['method']} {config['endpoint']}...")

            # Use custom headers if specified
            headers = config.get('headers')
            result = self.make_request(config['method'], config['endpoint'], config['data'], headers)
            results[name] = result

            if result.get('status_code'):
                print(f"  ✅ {result['status_code']} ({result['response_time']}ms)")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")

        return results

    def test_memory_endpoints(self) -> Dict[str, Any]:
        """Test MEMORY endpoints."""
        print("\n🧠 Testing MEMORY Endpoints")
        print("=" * 50)

        endpoints = {
            "POST_search": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/api/v1/search",
                "data": {
                    "spaceIds": [],
                    "includeInvalidated": False
                }
            },
            "POST_add": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/api/v1/add",
                "data": {
                    "episodeBody": "Test episode body for comprehensive testing",
                    "referenceTime": "2023-11-07T05:31:56Z",
                    "metadata": {},
                    "source": "comprehensive-test",
                    "spaceId": self.test_space_id,
                    "sessionId": "test-session-123"
                }
            },
            "GET_episodes_facts": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/episodes/{self.test_episode_id}/facts",
                "data": None
            },
            "GET_logs": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/logs",
                "data": None
            },
            "GET_logs_by_id": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/logs/{self.test_log_id}",
                "data": None
            },
            "DELETE_logs": {
                "method": "DELETE",
                "endpoint": "https://core.heysol.ai/api/v1/logs/{logId}".replace("{logId}", self.test_log_id),
                "data": {
                    "id": self.test_log_id
                }
            }
        }

        results = {}
        for name, config in endpoints.items():
            print(f"Testing {config['method']} {config['endpoint']}...")
            result = self.make_request(config['method'], config['endpoint'], config['data'])
            results[name] = result

            if result.get('status_code'):
                print(f"  ✅ {result['status_code']} ({result['response_time']}ms)")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")

        return results

    def test_spaces_endpoints(self) -> Dict[str, Any]:
        """Test SPACES endpoints."""
        print("\n🏢 Testing SPACES Endpoints")
        print("=" * 50)

        endpoints = {
            "GET_spaces": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/spaces",
                "data": None
            },
            "POST_spaces": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/api/v1/spaces",
                "data": {
                    "name": "Test Space",
                    "description": "Space created for comprehensive endpoint testing"
                }
            },
            "PUT_spaces_bulk": {
                "method": "PUT",
                "endpoint": "https://core.heysol.ai/api/v1/spaces",
                "data": {
                    "intent": "assign_statements",
                    "spaceId": self.test_space_id,
                    "statementIds": [self.test_statement_id],
                    "spaceIds": [self.test_space_id]
                }
            },
            "GET_spaces_by_id": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/spaces/{self.test_space_id}",
                "data": None
            },
            "PUT_spaces_by_id": {
                "method": "PUT",
                "endpoint": f"https://core.heysol.ai/api/v1/spaces/{self.test_space_id}",
                "data": {
                    "name": "Updated Test Space",
                    "description": "Updated space for comprehensive testing"
                }
            },
            "DELETE_spaces": {
                "method": "DELETE",
                "endpoint": f"https://core.heysol.ai/api/v1/spaces/{self.test_space_id}",
                "data": None
            }
        }

        results = {}
        for name, config in endpoints.items():
            print(f"Testing {config['method']} {config['endpoint']}...")
            result = self.make_request(config['method'], config['endpoint'], config['data'])
            results[name] = result

            if result.get('status_code'):
                print(f"  ✅ {result['status_code']} ({result['response_time']}ms)")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")

        return results

    def test_webhook_endpoints(self) -> Dict[str, Any]:
        """Test WEBHOOK endpoints."""
        print("\n🪝 Testing WEBHOOK Endpoints")
        print("=" * 50)

        endpoints = {
            "POST_webhooks": {
                "method": "POST",
                "endpoint": "https://core.heysol.ai/api/v1/webhooks",
                "data": {
                    "url": "https://example.com/webhook",
                    "secret": "test-secret-123"
                },
                "headers": {"Content-Type": "application/x-www-form-urlencoded"}
            },
            "GET_webhooks_by_id": {
                "method": "GET",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/webhooks/test-webhook-123",
                "data": None
            },
            "PUT_webhooks_by_id": {
                "method": "PUT",
                "endpoint": f"{self.protocol}://{self.domain}/api/v1/webhooks/test-webhook-123",
                "data": {
                    "url": "https://example.com/updated-webhook",
                    "secret": "updated-secret-123"
                },
                "headers": {"Content-Type": "application/x-www-form-urlencoded"}
            }
        }

        results = {}
        for name, config in endpoints.items():
            print(f"Testing {config['method']} {config['endpoint']}...")

            # Use custom headers if specified
            headers = config.get('headers')
            result = self.make_request(config['method'], config['endpoint'], config['data'], headers)
            results[name] = result

            if result.get('status_code'):
                print(f"  ✅ {result['status_code']} ({result['response_time']}ms)")
            else:
                print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")

        return results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all endpoint tests."""
        print("🚀 Comprehensive HeySol API Endpoint Testing")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"API Key Available: {'✅ YES' if self.api_key else '❌ NO'}")
        print("=" * 60)

        if not self.api_key:
            print("❌ ERROR: HEYSOL_API_KEY not found in environment")
            return self.results

        # Run all endpoint tests
        self.results["endpoints"]["USER"] = self.test_user_endpoints()
        self.results["endpoints"]["MEMORY"] = self.test_memory_endpoints()
        self.results["endpoints"]["SPACES"] = self.test_spaces_endpoints()
        self.results["endpoints"]["OAUTH2"] = self.test_oauth2_endpoints()
        self.results["endpoints"]["WEBHOOK"] = self.test_webhook_endpoints()

        # Calculate summary
        self.calculate_summary()

        return self.results

    def calculate_summary(self):
        """Calculate test summary statistics."""
        total_endpoints = 0
        working_endpoints = 0
        failed_endpoints = 0

        for category, endpoints in self.results["endpoints"].items():
            for endpoint_name, result in endpoints.items():
                total_endpoints += 1
                if result.get('status_code') and result['status_code'] < 400:
                    working_endpoints += 1
                else:
                    failed_endpoints += 1

        self.results["summary"] = {
            "total_endpoints": total_endpoints,
            "working_endpoints": working_endpoints,
            "failed_endpoints": failed_endpoints,
            "success_rate": round((working_endpoints / total_endpoints) * 100, 2) if total_endpoints > 0 else 0
        }

    def save_results(self, filename: str = None):
        """Save test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_endpoint_test_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filename}")
        return filename

    def print_summary(self):
        """Print test summary."""
        print("\n📊 COMPREHENSIVE ENDPOINT TEST SUMMARY")
        print("=" * 60)

        summary = self.results.get("summary", {})
        print(f"Total Endpoints Tested: {summary.get('total_endpoints', 0)}")
        print(f"Working Endpoints: {summary.get('working_endpoints', 0)}")
        print(f"Failed Endpoints: {summary.get('failed_endpoints', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0)}%")

        print("\n📋 Detailed Results by Category:")
        print("-" * 40)

        for category, endpoints in self.results["endpoints"].items():
            print(f"\n{category}:")
            for endpoint_name, result in endpoints.items():
                status_code = result.get('status_code')
                if status_code:
                    if status_code < 400:
                        print(f"  ✅ {endpoint_name}: {status_code}")
                    else:
                        print(f"  ❌ {endpoint_name}: {status_code}")
                else:
                    print(f"  ❌ {endpoint_name}: ERROR - {result.get('error', 'Unknown')}")


def main():
    """Main function to run comprehensive endpoint testing."""
    tester = ComprehensiveEndpointTester()
    results = tester.run_all_tests()
    tester.print_summary()
    filename = tester.save_results()

    print(f"\n🎉 Comprehensive endpoint testing completed!")
    print(f"📄 Detailed results saved to: {filename}")

    # Exit with appropriate code
    summary = results.get("summary", {})
    if summary.get("success_rate", 0) >= 80:
        print("✅ Overall test result: SUCCESS")
        return 0
    else:
        print("❌ Overall test result: FAILURE")
        return 1


if __name__ == "__main__":
    sys.exit(main())