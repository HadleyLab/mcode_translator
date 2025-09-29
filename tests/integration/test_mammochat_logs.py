#!/usr/bin/env python3
"""
Test script to list logs from MammoChat HeySol instance.
"""

import sys
sys.path.insert(0, '../heysol_api_client')

from heysol import HeySolClient

def main():
    # Use the MammoChat API key from .env
    api_key = "rc_pat_e6qbnv7mq4a3j538ghq3gidc6r5m268i1iz413hu"

    try:
        client = HeySolClient(api_key=api_key)
        print("Connected to HeySol API")

        # List logs
        logs = client.get_ingestion_logs(limit=10)
        print(f"Found {len(logs)} logs:")
        for log in logs:
            print(f"- ID: {log.get('id', 'N/A')}, Source: {log.get('source', 'N/A')}, Status: {log.get('status', 'N/A')}")

        client.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()