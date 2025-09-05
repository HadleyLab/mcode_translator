#!/usr/bin/env python3
"""
Test script to demonstrate the --process-trial flag functionality
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.fetcher import main
from src.pipeline.prompt_model_interface import set_extraction_prompt, set_mapping_prompt, set_model

def test_process_trial_flag():
    """Test the --process-trial flag functionality"""
    print("🧪 Testing --process-trial flag functionality")
    
    try:
        # Configure the pipeline
        print("🔧 Configuring pipeline...")
        set_extraction_prompt("generic_extraction")
        set_mapping_prompt("generic_mapping")
        set_model("deepseek-coder")
        print("✅ Pipeline configured successfully")
        
        # Test with a simple NCT ID and the process-trial flag
        print("🚀 Testing with --process-trial flag...")
        # We'll simulate the command line call
        import subprocess
        import json
        
        # Run the fetcher with process-trial flag
        cmd = [
            sys.executable, 
            "-m", 
            "src.pipeline.fetcher", 
            "--nct-id", 
            "NCT06965361", 
            "--process-trial"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Process trial flag test completed successfully")
            print(f"Output length: {len(result.stdout)} characters")
            # Try to parse the output as JSON to verify it's valid
            try:
                data = json.loads(result.stdout)
                print(f"✅ Output is valid JSON with {len(str(data))} characters")
                if 'McodeResults' in data:
                    print("✅ Mcode results found in output")
                    Mcode_results = data['McodeResults']
                    if 'extracted_entities' in Mcode_results:
                        print(f"📊 Extracted {len(Mcode_results['extracted_entities'])} entities")
                    if 'mcode_mappings' in Mcode_results:
                        print(f"🗺️  Mapped {len(Mcode_results['mcode_mappings'])} Mcode elements")
                else:
                    print("⚠️  No Mcode results found in output")
            except json.JSONDecodeError:
                print("⚠️  Output is not valid JSON")
                print("First 500 characters of output:")
                print(result.stdout[:500])
        else:
            print("❌ Process trial flag test failed")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_process_trial_flag()
    if success:
        print("\n✅ All process-trial flag tests completed successfully!")
    else:
        print("\n❌ Some process-trial flag tests failed!")
        sys.exit(1)