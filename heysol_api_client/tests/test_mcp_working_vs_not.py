#!/usr/bin/env python3
"""
Test MCP functionality: Working vs Not Working

This test demonstrates the difference between:
✅ Working MCP URL: https://core.heysol.ai/api/v1/mcp?source=Kilo-Code
❌ Not Working MCP URLs: Various incorrect URLs

Tests both direct MCP calls and client integration.
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


class MCPWorkingVsNotTest:
    """Test class to demonstrate MCP working vs not working."""

    def __init__(self):
        """Initialize test class."""
        self.api_key = os.getenv('HEYSOL_API_KEY')
        if not self.api_key:
            raise ValueError("HEYSOL_API_KEY environment variable not set")

    def test_working_mcp_url(self) -> Dict[str, Any]:
        """Test the working MCP URL."""
        print("✅ Testing WORKING MCP URL: https://core.heysol.ai/api/v1/mcp?source=Kilo-Code")
        print("=" * 70)

        mcp_url = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
        init_payload = {
            "jsonrpc": "2.0",
            "id": "test-working",
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0.0",
                "capabilities": {"tools": True},
                "clientInfo": {"name": "heysol-python-client", "version": "1.0.0"},
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
        }

        try:
            response = requests.post(
                mcp_url,
                json=init_payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                # Parse SSE response
                lines = response.text.strip().split('\n')
                data_lines = [line for line in lines if line.startswith('data: ')]

                if data_lines:
                    last_data = data_lines[-1][6:]  # Remove 'data: ' prefix
                    result = json.loads(last_data)

                    # Check for tools
                    tools_response = requests.post(
                        mcp_url,
                        json={"jsonrpc": "2.0", "id": "test-tools", "method": "tools/list", "params": {}},
                        headers=headers,
                        timeout=30
                    )

                    if tools_response.status_code == 200:
                        tools_lines = tools_response.text.strip().split('\n')
                        tools_data_lines = [line for line in tools_lines if line.startswith('data: ')]

                        if tools_data_lines:
                            tools_data = tools_data_lines[-1][6:]
                            tools_result = json.loads(tools_data)
                            tools = tools_result.get("result", {}).get("tools", [])
                            tool_names = [tool.get("name", "") for tool in tools]

                            return {
                                "status": "working",
                                "url": mcp_url,
                                "server_info": result.get("result", {}).get("serverInfo", {}),
                                "tools_available": tool_names,
                                "session_id": response.headers.get("mcp-session-id"),
                                "response_status": response.status_code
                            }

            return {
                "status": "failed",
                "url": mcp_url,
                "error": f"HTTP {response.status_code}",
                "response_text": response.text[:500]
            }

        except Exception as e:
            return {
                "status": "error",
                "url": mcp_url,
                "error": str(e)
            }

    def test_not_working_mcp_urls(self) -> List[Dict[str, Any]]:
        """Test various non-working MCP URLs."""
        print("\n❌ Testing NOT WORKING MCP URLs")
        print("=" * 70)

        not_working_urls = [
            "https://core.heysol.ai/api/v1",
            "https://core.heysol.ai/api/v1/mcp",
            "https://core.heysol.ai/mcp",
            "https://core.heysol.ai/api/mcp",
            "https://core.heysol.ai/v1/mcp",
            "https://core.heysol.ai/mcp/v1"
        ]

        results = []

        for url in not_working_urls:
            print(f"Testing: {url}")

            try:
                response = requests.post(
                    url,
                    json={"jsonrpc": "2.0", "id": "test", "method": "initialize", "params": {}},
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=10
                )

                results.append({
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                    "error": "HTTP Error" if response.status_code != 200 else "OK"
                })

            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e)
                })

        return results

    def test_client_with_working_mcp(self) -> Dict[str, Any]:
        """Test HeySolClient with working MCP URL."""
        print("\n🔧 Testing HeySolClient with WORKING MCP URL")
        print("=" * 50)

        try:
            client = HeySolClient(api_key=self.api_key)

            # Check if client has correct MCP URL
            expected_url = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
            if client.mcp_url == expected_url:
                print(f"✅ Client MCP URL is correct: {client.mcp_url}")
            else:
                return {
                    "status": "failed",
                    "error": f"Client MCP URL mismatch. Expected: {expected_url}, Got: {client.mcp_url}"
                }

            # Test MCP functionality
            try:
                result = client.search("test query", limit=1)
                return {
                    "status": "working",
                    "client_url": client.mcp_url,
                    "search_result_type": type(result).__name__
                }
            except HeySolError as e:
                return {
                    "status": "mcp_failed_but_client_working",
                    "client_url": client.mcp_url,
                    "error": str(e),
                    "note": "MCP may not be available but client fallback works"
                }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all MCP working vs not working tests."""
        print("🚀 MCP Working vs Not Working Test Suite")
        print("=" * 60)

        results = {
            "working_mcp": self.test_working_mcp_url(),
            "not_working_mcps": self.test_not_working_mcp_urls(),
            "client_with_working_mcp": self.test_client_with_working_mcp()
        }

        # Summary
        print("\n📊 SUMMARY")
        print("=" * 60)

        working_mcp = results["working_mcp"]
        if working_mcp.get("status") == "working":
            print("✅ WORKING MCP URL:")
            print(f"   URL: {working_mcp['url']}")
            print(f"   Server: {working_mcp['server_info'].get('name', 'Unknown')} v{working_mcp['server_info'].get('version', 'Unknown')}")
            print(f"   Tools: {', '.join(working_mcp['tools_available'])}")
            print(f"   Session ID: {working_mcp.get('session_id', 'N/A')}")
        else:
            print("❌ Working MCP URL failed")

        print(f"\n❌ NOT WORKING MCP URLs: {len(results['not_working_mcps'])} tested")

        client_test = results["client_with_working_mcp"]
        if client_test.get("status") == "working":
            print("✅ Client with working MCP: SUCCESS")
        else:
            print(f"⚠️ Client with working MCP: {client_test.get('status', 'FAILED')}")

        return results


def main():
    """Main function."""
    try:
        test = MCPWorkingVsNotTest()
        results = test.run_all_tests()

        # Save results
        with open("mcp_working_vs_not_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print("\n💾 Results saved to: mcp_working_vs_not_results.json")
        print("🎉 MCP Working vs Not Working test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()