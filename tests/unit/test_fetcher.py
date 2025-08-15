#!/usr/bin/env python3
"""
Test script for the Clinical Trial Data Fetcher
"""

import sys
import os
import json

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.clinical_trials_api import ClinicalTrialsAPI
from src.config import Config


def test_search_trials():
    """Test searching for clinical trials"""
    print("Testing search for clinical trials...")
    
    # Initialize the API client
    config = Config()
    api = ClinicalTrialsAPI(config)
    
    try:
        # Search for a simple term that should return results
        print("Searching for 'cancer'...")
        fields = [
            "NCTId", "BriefTitle", "EligibilityCriteria", "Condition",
            "Gender", "MinimumAge", "MaximumAge"
        ]
        result = api.search_trials("cancer", fields, 5)
        
        # Try to parse the response correctly
        if 'studies' in result:
            studies = result['studies']
            print(f"Found {len(studies)} trials")
            for i, study in enumerate(studies[:5]):  # Show up to 5 results
                # Try different possible structures for NCT ID
                nct_id = 'Unknown'
                title = 'No title'
                
                # Check if study has protocolSection
                if 'protocolSection' in study:
                    protocol_section = study['protocolSection']
                    # Check for identificationModule
                    if 'identificationModule' in protocol_section:
                        identification_module = protocol_section['identificationModule']
                        if 'nctId' in identification_module:
                            nct_id = identification_module['nctId']
                        if 'briefTitle' in identification_module:
                            title = identification_module['briefTitle']
                
                print(f"  {i+1}. {nct_id}: {title}")
        else:
            print("No 'studies' key found in response")
            print("Available keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")
        
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def test_get_full_study():
    """Test fetching a full study record"""
    print("\nTesting fetch of full study record...")
    
    # Initialize the API client
    config = Config()
    api = ClinicalTrialsAPI(config)
    
    try:
        # Try to fetch a real trial ID if we have any from the search
        # For now, we'll use a placeholder
        nct_id = "NCT00000000"  # This is a placeholder, you would use a real NCT ID
        print(f"Fetching study {nct_id}...")
        result = api.get_full_study(nct_id)
        
        print("Study fetched successfully")
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
        # This is expected to fail with the placeholder ID
        return None


def main():
    """Main test function"""
    print("Running Clinical Trial Data Fetcher Tests")
    print("=" * 50)
    
    # Test search functionality
    search_result = test_search_trials()
    
    # Test full study fetch (this will likely fail with the placeholder)
    study_result = test_get_full_study()
    
    print("\nTests completed.")


if __name__ == '__main__':
    main()