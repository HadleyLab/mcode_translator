"""
Integration tests to verify data structures from live ClinicalTrials.gov API.
These tests help ensure that our mocks match the actual API responses.
"""

import json
from pytrials.client import ClinicalTrials


def test_search_trials_structure():
    """Test the structure of search trials response from live API"""
    print("Testing search trials structure...")
    
    ct = ClinicalTrials()
    
    # Test search trials with study fields
    result = ct.get_study_fields(
        search_expr="breast cancer",
        fields=["NCTId", "BriefTitle", "Condition", "OverallStatus"],
        max_studies=2,
        fmt="json"
    )
    
    print("Search trials result structure:")
    print(f"  Type: {type(result)}")
    print(f"  Keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
    
    if 'studies' in result:
        print(f"  Number of studies: {len(result['studies'])}")
        if result['studies']:
            first_study = result['studies'][0]
            print(f"  First study keys: {list(first_study.keys()) if hasattr(first_study, 'keys') else 'No keys'}")
            if 'protocolSection' in first_study:
                protocol_section = first_study['protocolSection']
                print(f"  Protocol section keys: {list(protocol_section.keys()) if hasattr(protocol_section, 'keys') else 'No keys'}")
                if 'identificationModule' in protocol_section:
                    id_module = protocol_section['identificationModule']
                    print(f"  Identification module keys: {list(id_module.keys()) if hasattr(id_module, 'keys') else 'No keys'}")
    
    # Save result for reference
    with open('search_trials_sample.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def test_full_studies_structure():
    """Test the structure of full studies response from live API"""
    print("\nTesting full studies structure...")
    
    ct = ClinicalTrials()
    
    # Try a broader search to get full studies
    result = ct.get_full_studies(
        search_expr="breast cancer",
        max_studies=1,
        fmt="json"
    )
    
    print("Full studies result structure:")
    print(f"  Type: {type(result)}")
    print(f"  Keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
    
    if 'studies' in result:
        print(f"  Number of studies: {len(result['studies'])}")
        if result['studies']:
            first_study = result['studies'][0]
            print(f"  First study keys: {list(first_study.keys()) if hasattr(first_study, 'keys') else 'No keys'}")
            
            # Check key sections
            for section in ['protocolSection', 'derivedSection', 'hasResults']:
                if section in first_study:
                    print(f"  {section}: Present")
                    if section == 'protocolSection':
                        protocol_section = first_study[section]
                        print(f"    Protocol section keys: {list(protocol_section.keys()) if hasattr(protocol_section, 'keys') else 'No keys'}")
                        if 'identificationModule' in protocol_section:
                            id_module = protocol_section['identificationModule']
                            print(f"    Identification module keys: {list(id_module.keys()) if hasattr(id_module, 'keys') else 'No keys'}")
                            if 'nctId' in id_module:
                                print(f"    Sample NCT ID: {id_module['nctId']}")
        else:
            print("  No studies found")
    else:
        print("  No 'studies' key in result")
    
    # Save result for reference
    with open('full_study_sample.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result


def test_single_full_study():
    """Test retrieving a single full study by NCT ID"""
    print("\nTesting single full study retrieval...")
    
    ct = ClinicalTrials()
    
    # First, get a valid NCT ID from search results
    search_result = ct.get_study_fields(
        search_expr="cancer",
        fields=["NCTId"],
        max_studies=1,
        fmt="json"
    )
    
    if search_result.get('studies') and len(search_result['studies']) > 0:
        nct_id = search_result['studies'][0]['protocolSection']['identificationModule']['nctId']
        print(f"Using NCT ID: {nct_id}")
        
        # Try to get the full study using different search expressions
        # Method 1: Direct search by NCT ID
        result1 = ct.get_full_studies(
            search_expr=f'NCTId = "{nct_id}"',
            max_studies=1,
            fmt="json"
        )
        
        print(f"  Method 1 (NCTId = \"{nct_id}\"): Found {len(result1.get('studies', []))} studies")
        
        # Method 2: Search by NCT ID without quotes
        result2 = ct.get_full_studies(
            search_expr=f'NCTId = {nct_id}',
            max_studies=1,
            fmt="json"
        )
        
        print(f"  Method 2 (NCTId = {nct_id}): Found {len(result2.get('studies', []))} studies")
        
        # Return the one that worked (if any)
        result = result1 if len(result1.get('studies', [])) > 0 else result2
        return result
    else:
        print("No studies found in search results")
        return None


def compare_structures():
    """Compare the structures and output findings"""
    print("\n" + "="*50)
    print("STRUCTURE COMPARISON REPORT")
    print("="*50)
    
    print("\n1. Search Trials Structure:")
    print("   - Root keys: ['studies', 'nextPageToken']")
    print("   - Each study has: 'protocolSection'")
    print("   - Protocol section has: 'identificationModule', 'statusModule', 'conditionsModule', etc.")
    print("   - Identification module has: 'nctId', 'briefTitle'")
    
    print("\n2. Full Studies Structure:")
    print("   - Root keys: ['studies', 'nextPageToken'] (when studies found)")
    print("   - Each study has: 'protocolSection', 'derivedSection', 'hasResults'")
    print("   - Protocol section has: 'identificationModule', etc.")
    print("   - Identification module has: 'nctId', 'briefTitle'")
    
    print("\n3. Mock Structure Recommendations:")
    print("   - Update mocks to use 'studies' instead of 'studies' or 'FullStudies'")
    print("   - Use lowercase key names (e.g., 'nctId' instead of 'NCTId')")
    print("   - Match the nested structure: studies -> protocolSection -> identificationModule -> nctId")
    print("   - For full studies, expect 'protocolSection', 'derivedSection', 'hasResults'")


def main():
    """Main test function"""
    print("ClinicalTrials.gov API Structure Verification")
    print("="*50)
    
    try:
        search_result = test_search_trials_structure()
        full_result = test_full_studies_structure()
        single_result = test_single_full_study()
        compare_structures()
        
        print("\n" + "="*50)
        print("Test completed successfully!")
        print("Sample data saved to search_trials_sample.json and full_study_sample.json")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()