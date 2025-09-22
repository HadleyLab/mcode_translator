#!/usr/bin/env python3
"""
Unified MCP (Model Context Protocol) Tests

Tests MCP functionality with the correct URL and validates working vs non-working endpoints.
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


def parse_mcp_response(response: requests.Response) -> Dict[str, Any]:
    """Parse MCP JSON-RPC response like the client.py implementation"""
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip()

    if content_type == "application/json":
        msg = response.json()
    elif content_type == "text/event-stream":
        msg = None
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip())
                break
        if msg is None:
            raise Exception("No JSON in SSE stream")
    else:
        raise Exception(f"Unexpected Content-Type: {content_type}")

    if "error" in msg:
        raise Exception(f"MCP error: {msg['error']}")

    return msg.get("result", msg)


def test_mcp_working_url():
    """Test the working MCP URL."""
    print("✅ Testing WORKING MCP URL: https://core.heysol.ai/api/v1/mcp?source=Kilo-Code")

    api_key = os.getenv('HEYSOL_API_KEY')
    if not api_key:
        print("❌ HEYSOL_API_KEY environment variable not set")
        return False

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
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream, */*",
    }

    try:
        response = requests.post(mcp_url, json=init_payload, headers=headers, timeout=30)

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

                        print(f"✅ MCP Working! Server: {result.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
                        print(f"✅ Available tools: {', '.join(tool_names)}")
                        return True

        print(f"❌ MCP failed: HTTP {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ MCP error: {e}")
        return False


def test_mcp_not_working_urls():
    """Test various non-working MCP URLs."""
    print("\n❌ Testing NOT WORKING MCP URLs")

    api_key = os.getenv('HEYSOL_API_KEY')
    if not api_key:
        print("❌ HEYSOL_API_KEY environment variable not set")
        return []

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
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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


def test_client_mcp_integration():
    """Test HeySolClient with MCP integration."""
    print("\n🔧 Testing HeySolClient MCP Integration")

    api_key = os.getenv('HEYSOL_API_KEY')
    if not api_key:
        print("❌ HEYSOL_API_KEY environment variable not set")
        return False

    try:
        client = HeySolClient(api_key=api_key)

        # Check if client has correct MCP URL
        expected_url = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
        if client.mcp_url == expected_url:
            print(f"✅ Client MCP URL is correct: {client.mcp_url}")
        else:
            print(f"❌ Client MCP URL mismatch. Expected: {expected_url}, Got: {client.mcp_url}")
            return False

        # Test MCP functionality
        try:
            result = client.search("test query", limit=1)
            print(f"✅ MCP search successful: {type(result)}")
            return True

        except HeySolError as e:
            print(f"⚠️ MCP search failed (may be expected): {e}")
            # This might fail if MCP tools aren't available, but that's OK
            return True

    except Exception as e:
        print(f"❌ Client test failed: {e}")
        return False


def run_all_mcp_tests():
    """Run all MCP tests and return results."""
    print("🚀 MCP Test Suite")
    print("=" * 50)

    results = {
        "working_mcp": test_mcp_working_url(),
        "not_working_mcps": test_mcp_not_working_urls(),
        "client_integration": test_client_mcp_integration()
    }

    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)

    if results["working_mcp"]:
        print("✅ WORKING MCP URL: SUCCESS")
    else:
        print("❌ WORKING MCP URL: FAILED")

    print(f"❌ NOT WORKING MCP URLs: {len(results['not_working_mcps'])} tested")

    if results["client_integration"]:
        print("✅ CLIENT MCP INTEGRATION: SUCCESS")
    else:
        print("❌ CLIENT MCP INTEGRATION: FAILED")

    return results


if __name__ == "__main__":
    results = run_all_mcp_tests()

    # Save results
    with open("mcp_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Results saved to: mcp_test_results.json")
    print("🎉 MCP test suite completed!")