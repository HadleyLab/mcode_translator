import click
import json
import sys
import time
import hashlib
from pytrials.client import ClinicalTrials
from src.utils.config import Config
from src.utils.cache import CacheManager
from src.nlp_engine.nlp_engine import NLPEngine
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.structured_data_generator.structured_data_generator import StructuredDataGenerator

# Map our field names to valid ClinicalTrials.gov API field names
FIELD_MAPPING = {
    "NCTId": "NCTId",
    "BriefTitle": "BriefTitle",
    "Condition": "Condition",
    "OverallStatus": "OverallStatus"
}

DEFAULT_SEARCH_FIELDS = ["NCTId", "BriefTitle", "Condition", "OverallStatus"]


class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass


def search_trials(search_expr: str, fields=None, max_results: int = 100):
    """
    Search for clinical trials matching the expression
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        max_results: Maximum number of results to return (default: 100)
        
    Returns:
        Dictionary containing search results
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Create cache key
    cache_key_data = f"search:{search_expr}:{','.join(fields) if fields else 'all'}:{max_results}"
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    
    # Try to get from cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Map field names to valid API names
        api_fields = [FIELD_MAPPING.get(field, field) for field in fields] if fields else DEFAULT_SEARCH_FIELDS
        
        # Use correct API method for searching
        result = ct.get_study_fields(
            search_expr=search_expr,
            fields=api_fields,
            max_studies=max_results,
            fmt="json"
        )
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


def get_full_study(nct_id: str):
    """
    Get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT00000000")
        
    Returns:
        Dictionary containing the full study record
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Create cache key
    cache_key = f"full_study:{nct_id}"
    
    # Try to get from cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Use search to get study by NCT ID
        search_result = ct.get_study_fields(
            search_expr=f'NCTId = "{nct_id}"',
            fields=["NCTId", "BriefTitle"],
            max_studies=1,
            fmt="json"
        )
        if not search_result.get("StudyFields") or len(search_result["StudyFields"]) == 0:
            raise ValueError(f"No study found for NCT ID {nct_id}")
        
        # Get study details using search result
        study_fields = search_result["StudyFields"][0]
        result = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": study_fields.get("BriefTitle", [""])[0]
                }
            }
        }
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


@click.command()
@click.option('--condition', '-c', help='Condition to search for (e.g., "breast cancer")')
@click.option('--nct-id', '-n', help='Specific NCT ID to fetch (e.g., "NCT00000000")')
@click.option('--limit', '-l', default=10, help='Maximum number of results to return')
@click.option('--export', '-e', type=click.Path(), help='Export results to JSON file')
@click.option('--process-criteria', '-p', is_flag=True, help='Process eligibility criteria with NLP engine')
def main(condition, nct_id, limit, export, process_criteria):
    """
    Clinical Trial Data Fetcher for mCODE Translator
    
    Examples:
      python fetcher.py --condition "breast cancer" --limit 10
      python fetcher.py --nct-id NCT00000000
      python fetcher.py --condition "lung cancer" --export results.json
      python fetcher.py --nct-id NCT00000000 --process-criteria
    """
    
    try:
        if nct_id:
            # Fetch a specific trial by NCT ID
            click.echo(f"Fetching trial {nct_id}...")
            result = get_full_study(nct_id)
            display_single_study(result, export, process_criteria)
        elif condition:
            # Search for trials by condition
            click.echo(f"Searching for trials matching '{condition}'...")
            result = search_trials(condition, None, limit)
            display_search_results(result, export)
        else:
            # Show help if no arguments provided
            click.echo("Please specify either a condition to search for or an NCT ID to fetch.")
            click.echo("Use --help for more information.")
            sys.exit(1)
            
    except ClinicalTrialsAPIError as e:
        click.echo(f"API Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected Error: {str(e)}", err=True)
        sys.exit(1)


def display_single_study(result, export_path=None, process_criteria=False):
    """
    Display a single study result or export to file
    
    Args:
        result: Study result to display
        export_path: Optional path to export JSON file
        process_criteria: Whether to process eligibility criteria with NLP
    """
    # Process eligibility criteria with NLP if requested
    if process_criteria and 'protocolSection' in result:
        protocol_section = result['protocolSection']
        if 'eligibilityModule' in protocol_section:
            eligibility_module = protocol_section['eligibilityModule']
            if 'eligibilityCriteria' in eligibility_module:
                criteria_text = eligibility_module['eligibilityCriteria']
                if criteria_text:
                    try:
                        nlp_engine = NLPEngine()
                        # Ensure criteria_text is a string
                        if isinstance(criteria_text, list):
                            criteria_text = ' '.join(str(item) for item in criteria_text)
                        elif not isinstance(criteria_text, str):
                            criteria_text = str(criteria_text)
                        nlp_result = nlp_engine.process_criteria(criteria_text)
                        
                        # Process through the full pipeline
                        # Step 1: Extract codes
                        code_extractor = CodeExtractionModule()
                        code_result = code_extractor.process_criteria_for_codes(criteria_text, nlp_result['entities'])
                        
                        # Step 2: Map to mCODE
                        mapper = MCODEMappingEngine()
                        mapping_result = mapper.process_nlp_output(nlp_result)
                        
                        # Step 3: Generate structured data
                        generator = StructuredDataGenerator()
                        structured_result = generator.generate_mcode_resources(
                            mapping_result['mapped_elements'],
                            nlp_result.get('demographics', {})
                        )
                        
                        # Add all results to the study data
                        if 'mcodeResults' not in result:
                            result['mcodeResults'] = {}
                        result['mcodeResults']['nlp'] = nlp_result
                        result['mcodeResults']['codes'] = code_result
                        result['mcodeResults']['mappings'] = mapping_result
                        result['mcodeResults']['structured_data'] = structured_result
                    except Exception as e:
                        click.echo(f"Warning: Error processing criteria with NLP: {str(e)}", err=True)
    
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Study exported to {export_path}")
    else:
        # Display study to console
        click.echo(json.dumps(result, indent=2))


def display_search_results(result, export_path=None):
    """
    Display search results or export to file
    
    Args:
        result: Search results to display
        export_path: Optional path to export JSON file
    """
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Results exported to {export_path}")
    else:
        # Display results to console
        if 'studies' in result:
            studies = result['studies']
            click.echo(f"Found {len(studies)} trials:")
            for i, study in enumerate(studies[:10]):  # Show up to 10 results
                # Extract NCT ID and title
                nct_id = 'Unknown'
                title = 'No title'
                
                if 'protocolSection' in study:
                    protocol_section = study['protocolSection']
                    if 'identificationModule' in protocol_section:
                        identification_module = protocol_section['identificationModule']
                        if 'nctId' in identification_module:
                            nct_id = identification_module['nctId']
                        if 'briefTitle' in identification_module:
                            title = identification_module['briefTitle']
                
                click.echo(f"  {i+1}. {nct_id}: {title}")
        else:
            click.echo("No studies found in results")


def display_results(results, export_path=None):
    """
    Display results or export to file
    
    Args:
        results: Results to display
        export_path: Optional path to export JSON file
    """
    if export_path:
        # Export to JSON file
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results exported to {export_path}")
    else:
        # Display results to console
        click.echo(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()