#!/usr/bin/env python3
"""
Test script to verify test cases can be loaded properly
"""

import json
from pathlib import Path

def test_load_cases():
    """Test loading both test case files"""
    
    # Test various cancers file
    various_cancers_path = Path("examples/test_cases/various_cancers_test_cases.json")
    if various_cancers_path.exists():
        try:
            with open(various_cancers_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Various cancers test cases loaded successfully: {len(data)} cases")
            for case_id in list(data.keys())[:3]:  # Show first 3 cases
                title = data[case_id]['protocolSection']['identificationModule']['briefTitle']
                print(f"   - {case_id}: {title}")
        except Exception as e:
            print(f"❌ Failed to load various cancers: {e}")
    else:
        print("❌ Various cancers file not found")
    
    # Test clinical cases file
    clinical_path = Path("examples/test_cases/clinical_test_cases.json")
    if clinical_path.exists():
        try:
            with open(clinical_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Clinical test cases loaded successfully: {len(data)} cases")
            for case_id in list(data.keys())[:3]:  # Show first 3 cases
                title = data[case_id]['protocolSection']['identificationModule']['briefTitle']
                print(f"   - {case_id}: {title}")
        except Exception as e:
            print(f"❌ Failed to load clinical cases: {e}")
    else:
        print("❌ Clinical cases file not found")

if __name__ == "__main__":
    test_load_cases()