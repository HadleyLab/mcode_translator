"""
Debug script to test authentication with live API
"""

import os
import sys
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
API_KEY = os.getenv('COREAI_API_KEY')
if not API_KEY:
    print("❌ COREAI_API_KEY not found in environment")
    sys.exit(1)

print(f"🔑 Using API Key: {API_KEY[:20]}...")
print(f"🌐 Base URL: https://core.heysol.ai/api/v1/mcp")

# Test basic authentication
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream, */*",
    "User-Agent": "heysol-python-client/debug"
}

print("\n🔍 Testing basic authentication...")

# Test 1: Direct API call to profile endpoint
try:
    response = requests.get(
        "https://core.heysol.ai/api/profile",
        headers=headers,
        timeout=10
    )
    print(f"📡 Profile endpoint response: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ Profile data: {response.json()}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")

# Test 2: MCP initialize endpoint
print("\n🔍 Testing MCP initialization...")
try:
    init_payload = {
        "jsonrpc": "2.0",
        "id": "debug-test-123",
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0.0",
            "capabilities": {"tools": True},
            "clientInfo": {
                "name": "debug-client",
                "version": "1.0.0"
            }
        }
    }

    response = requests.post(
        "https://core.heysol.ai/api/v1/mcp",
        headers=headers,
        json=init_payload,
        timeout=10
    )
    print(f"📡 MCP init response: {response.status_code}")
    if response.status_code == 200:
        print(f"✅ MCP init successful: {response.json()}")
    else:
        print(f"❌ MCP init error: {response.text}")
except Exception as e:
    print(f"❌ MCP init connection error: {e}")

# Test 3: Check if API key format is correct
print("\n🔍 API Key format analysis:")
print(f"Length: {len(API_KEY)}")
print(f"Starts with: {API_KEY[:10]}")
print(f"Format: {'Valid GitLab PAT format' if API_KEY.startswith('rc_pat_') else 'Unknown format'}")

# Test 4: Try different base URLs
print("\n🔍 Testing different base URLs...")
base_urls = [
    "https://core.heysol.ai/api/v1",
    "https://core.heysol.ai/api",
    "https://api.heysol.ai/v1",
    "https://gitlab.com/api/v4"
]

for base_url in base_urls:
    try:
        test_url = f"{base_url}/profile" if "profile" not in base_url else base_url
        response = requests.get(test_url, headers=headers, timeout=5)
        print(f"📡 {base_url}: {response.status_code}")
        if response.status_code == 200:
            print(f"✅ Found working endpoint: {base_url}")
            break
    except:
        print(f"📡 {base_url}: Connection failed")

print("\n🔚 Debug complete")