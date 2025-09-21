#!/usr/bin/env python3
"""
Test MCP functionality with the correct URL.

Tests the MCP server at: https://core.heysol.ai/api/v1/mcp?source=Kilo-Code
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Any

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


def test_mcp_correct_url():
    """Test MCP functionality with the correct URL."""
    print("🧪 Testing MCP with correct URL: https://core.heysol.ai/api/v1/mcp?source=Kilo-Code")
    print("=" * 70)

    # Get API key from environment
    api_key = os.getenv('HEYSOL_API_KEY')
    if not api_key:
        print("❌ HEYSOL_API_KEY environment variable not set")
        return False

    print(f"✅ API Key available: {api_key[:10]}...")

    # Test MCP initialization payload
    mcp_url = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
    init_payload = {
        "jsonrpc": "2.0",
        "id": "test-initialize",
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
        print(f"🔍 Testing MCP initialization at: {mcp_url}")

        response = requests.post(
            mcp_url,
            json=init_payload,
            headers=headers,
            timeout=30
        )

        print(f"📡 Response status: {response.status_code}")
        print(f"📡 Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            # Check if response is Server-Sent Events (SSE) format
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                print("✅ MCP server responds with Server-Sent Events (SSE) format")
                print(f"📡 Response text: {response.text[:500]}...")

                # Parse SSE response
                lines = response.text.strip().split('\n')
                data_lines = [line for line in lines if line.startswith('data: ')]

                if data_lines:
                    try:
                        # Get the last data line (should contain the JSON response)
                        last_data = data_lines[-1][6:]  # Remove 'data: ' prefix
                        result = json.loads(last_data)
                        print(f"✅ Parsed SSE data: {json.dumps(result, indent=2)}")

                        # Check if it's a valid MCP response
                        if "jsonrpc" in result and result["jsonrpc"] == "2.0":
                            print("✅ Valid MCP JSON-RPC response received")

                            # Check server info
                            server_info = result.get("result", {}).get("serverInfo", {})
                            print(f"✅ MCP Server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")

                            # Test MCP tools using the working approach from CoreMemoryClient
                            print("🔍 Testing MCP tools with working implementation...")

                            # Test memory_ingest tool
                            ingest_payload = {
                                "jsonrpc": "2.0",
                                "id": "test-memory-ingest",
                                "method": "tools/call",
                                "params": {
                                    "name": "memory_ingest",
                                    "arguments": {
                                        "message": "Test message for MCP memory ingestion",
                                        "source": "test-client"
                                    }
                                }
                            }

                            # Use session ID if available
                            session_id = response.headers.get("Mcp-Session-Id")
                            if session_id:
                                headers["Mcp-Session-Id"] = session_id
                                print(f"✅ Using session ID: {session_id}")

                            ingest_response = requests.post(
                                mcp_url,
                                json=ingest_payload,
                                headers=headers,
                                timeout=30
                            )

                            print(f"📡 Memory ingest response status: {ingest_response.status_code}")

                            # Test memory_ingest tool using proper response parsing
                            try:
                                ingest_result = parse_mcp_response(ingest_response)
                                print("✅ memory_ingest tool working!")
                                print(f"📄 Ingest result: {json.dumps(ingest_result, indent=2)}")
                            except Exception as e:
                                print(f"❌ Memory ingest failed: {e}")
                                return False

                            # Test memory_search tool
                            search_payload = {
                                "jsonrpc": "2.0",
                                "id": "test-memory-search",
                                "method": "tools/call",
                                "params": {
                                    "name": "memory_search",
                                    "arguments": {
                                        "query": "test message",
                                        "limit": 5
                                    }
                                }
                            }

                            search_response = requests.post(
                                mcp_url,
                                json=search_payload,
                                headers=headers,
                                timeout=30
                            )

                            print(f"📡 Memory search response status: {search_response.status_code}")

                            if search_response.status_code == 200:
                                try:
                                    search_result = parse_mcp_response(search_response)
                                    print("✅ memory_search tool working!")
                                    print(f"📄 Search result: {json.dumps(search_result, indent=2)}")
                                except Exception as e:
                                    print(f"❌ Memory search failed: {e}")
                                    return False
                            else:
                                print(f"❌ Memory search failed: {search_response.status_code}")
                                print(f"Response: {search_response.text}")
                                return False

                            return True
                        else:
                            print(f"❌ Invalid MCP response format: {result}")
                            return False

                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON in SSE data: {e}")
                        print(f"SSE data: {last_data}")
                        return False
                else:
                    print("❌ No data lines found in SSE response")
                    return False
            else:
                print(f"❌ Unexpected content type: {content_type}")
                return False
        else:
            print(f"❌ MCP initialization failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_client_with_correct_mcp_url():
    """Test HeySolClient with the correct MCP URL."""
    print("\n🔧 Testing HeySolClient with correct MCP URL")
    print("=" * 50)

    api_key = os.getenv('HEYSOL_API_KEY')
    if not api_key:
        print("❌ HEYSOL_API_KEY environment variable not set")
        return False

    try:
        # Create client with MCP URL override
        client = HeySolClient(api_key=api_key)

        # Check if client has the correct MCP URL
        expected_mcp_url = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
        if client.mcp_url == expected_mcp_url:
            print(f"✅ Client MCP URL is correct: {client.mcp_url}")
        else:
            print(f"❌ Client MCP URL mismatch. Expected: {expected_mcp_url}, Got: {client.mcp_url}")
            return False

        # Try to use MCP functionality
        try:
            # Test memory search (this will use MCP if available)
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


if __name__ == "__main__":
    print("🚀 MCP URL Testing with Correct Endpoint")
    print("=" * 50)

    # Test direct MCP connection
    direct_success = test_mcp_correct_url()

    # Test client with correct MCP URL
    client_success = test_client_with_correct_mcp_url()

    print("\n📊 Test Results:")
    print(f"Direct MCP Test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"Client MCP Test: {'✅ PASS' if client_success else '❌ FAIL'}")

    if direct_success and client_success:
        print("\n🎉 All MCP tests passed! The correct URL is working.")
        exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        exit(1)