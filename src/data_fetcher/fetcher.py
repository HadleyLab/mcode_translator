import click
import json
import sys
import time
import hashlib
import os
import re
import requests
# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pytrials.client import ClinicalTrials
from src.utils.config import Config
from src.utils.cache import CacheManager
from src.pipeline.dynamic_extraction_pipeline import DynamicExtractionPipeline
from src.utils.logging_config import get_logger

# Get logger instance
logger = get_logger(__name__)

# Map our field names to valid ClinicalTrials.gov API field names
FIELD_MAPPING = {
    "NCTId": "NCTId",
    "BriefTitle": "BriefTitle",
    "Condition": "Condition",
    "OverallStatus": "OverallStatus",
    "BriefSummary": "BriefSummary",
    "StartDate": "StartDate",
    "CompletionDate": "CompletionDate"
}

# Fields that are valid for JSON format in ClinicalTrials API
VALID_JSON_FIELDS = [
    "NCTId",
    "BriefTitle",
    "Condition",
    "OverallStatus",
    "BriefSummary",
    "StartDate",
    "CompletionDate"
]

DEFAULT_SEARCH_FIELDS = [
    "NCTId",
    "BriefTitle",
    "Condition",
    "OverallStatus",
    "BriefSummary",
    "StartDate",
    "CompletionDate"
]


class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass


def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None, use_cache: bool = True):
    """
    Search for clinical trials matching the expression with pagination support
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        max_results: Maximum number of results to return (default: 100)
        page_token: Page token for pagination (default: None)
        use_cache: Whether to use caching (default: True)
        
    Returns:
        Dictionary containing search results with pagination metadata
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Create cache key that includes pagination parameters
    if page_token:
        cache_key_data = f"search:{search_expr}:{','.join(fields) if fields else 'all'}:{max_results}:token{page_token}"
    else:
        cache_key_data = f"search:{search_expr}:{','.join(fields) if fields else 'all'}:{max_results}:tokenNone"
    logger.info(f"search_trials: Creating cache key for search_expr='{search_expr}', max_results={max_results}, page_token={page_token}")
    cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
    
    # Try to get from cache first (if caching is enabled)
    if use_cache:
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            studies_count = len(cached_result.get('studies', []))
            logger.info(f"search_trials: Returning cached result for cache_key={cache_key}, studies_count={studies_count}")
            return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Map field names to valid API names
        api_fields = [FIELD_MAPPING.get(field, field) for field in fields] if fields else DEFAULT_SEARCH_FIELDS
        
        # Use direct API call to ClinicalTrials.gov API v2 for proper token-based pagination
        import requests
        
        # Build the API URL
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "format": "json",
            "query.term": search_expr,
            "pageSize": max_results,
            "fields": ",".join(api_fields)
        }
        
        # Add page token if provided
        if page_token:
            params["pageToken"] = page_token
        
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        result = response.json()
        
        # Add pagination metadata to the result
        if isinstance(result, dict):
            studies_count = len(result.get('studies', []))
            logger.info(f"search_trials: API returned {studies_count} studies")
            result['pagination'] = {
                'max_results': max_results
            }
            
            # Add nextPageToken if it exists in the response
            if 'nextPageToken' in result:
                result['nextPageToken'] = result['nextPageToken']
                logger.debug(f"search_trials: nextPageToken found in response")
        
        # Cache the result
        logger.info(f"search_trials: Caching result for cache_key={cache_key}")
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
        logger.info(f"get_full_study: Returning cached result for NCT ID {nct_id}")
        return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Log the NCT ID we're trying to fetch
        logger.info(f"Attempting to fetch full study details for NCT ID: {nct_id}")
        
        # Use get_full_studies with the NCT ID as search expression
        # Try different search expression formats to handle API variations
        search_expr = nct_id
        logger.info(f"Trying search expression: {search_expr}")
        
        # Rate limiting
        logger.debug(f"Applying rate limit delay: {config.rate_limit_delay}s")
        time.sleep(config.rate_limit_delay)
        
        # First try with the NCT ID directly
        try:
            logger.info(f"Calling get_full_studies API for NCT ID: {nct_id}")
            result = ct.get_full_studies(
                search_expr=search_expr,
                max_studies=1,
                fmt="json"
            )
            logger.info(f"API call completed for NCT ID: {nct_id}")
        except Exception as e:
            logger.error(f"Exception in get_full_studies for NCT ID {nct_id} with search expression '{search_expr}': {str(e)}")
            result = None
        
        # Log the result for debugging
        logger.info(f"First attempt result type: {type(result)}, value: {result}")
        
        # If we get None or empty result, try with quotes around the NCT ID
        if result is None or (isinstance(result, dict) and 'studies' in result and len(result['studies']) == 0):
            search_expr = f'"{nct_id}"'
            logger.info(f"First attempt failed, trying search expression with quotes: {search_expr}")
            try:
                result = ct.get_full_studies(
                    search_expr=search_expr,
                    max_studies=1,
                    fmt="json"
                )
            except Exception as e:
                logger.error(f"Exception in get_full_studies for NCT ID {nct_id} with search expression '{search_expr}': {str(e)}")
                result = None
            logger.info(f"Second attempt result type: {type(result)}, value: {result}")
        
        # Log the result for debugging
        logger.info(f"Attempt result type: {type(result)}, value: {result}")
        
        # Check if we got a valid response
        if result is None:
            raise ValueError(f"No response received from API for NCT ID {nct_id} after multiple attempts")
        
        # Verify result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(f"Invalid response format from API for NCT ID {nct_id}: expected dict, got {type(result)}")
        
        # Check if studies key exists and has content
        if 'studies' not in result:
            raise ValueError(f"API response missing 'studies' key for NCT ID {nct_id}")
        
        studies = result['studies']
        if not isinstance(studies, list) or len(studies) == 0:
            raise ValueError(f"No study found for NCT ID {nct_id} after multiple attempts")
        
        # Extract the study from the response
        study = studies[0]
        
        # Cache the result
        cache_manager.set(cache_key, study)
        
        return study
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed for NCT ID {nct_id}: {str(e)}")


def get_study_fields():
    """
    Get the list of available study fields from the ClinicalTrials.gov API
    
    Returns:
        Dictionary containing available fields for different formats
    """
    try:
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Return the study fields attribute
        return ct.study_fields
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


@click.command()
@click.option('--condition', '-c', help='Condition to search for (e.g., "breast cancer")')
@click.option('--nct-id', '-n', help='Specific NCT ID to fetch (e.g., "NCT00000000")')
@click.option('--limit', '-l', default=10, help='Maximum number of results to return')
@click.option('--count', '--calculate-count', is_flag=True, help='Calculate total number of studies and pages')
@click.option('--export', '-e', type=click.Path(), help='Export results to JSON file')
@click.option('--process-criteria', '-p', is_flag=True, help='Process eligibility criteria with NLP engine')
def main(condition, nct_id, limit, count, export, process_criteria):
    """
    Clinical Trial Data Fetcher for mCODE Translator
    
    Examples:
      python fetcher.py --condition "breast cancer" --limit 10
      python fetcher.py --nct-id NCT00000000
      python fetcher.py --condition "lung cancer" --export results.json
      python fetcher.py --nct-id NCT00000000 --process-criteria
      python fetcher.py --condition "breast cancer" --count
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
            if count:
                # Calculate total studies and pages
                click.echo("Calculating total number of studies...")
                stats = calculate_total_studies(condition)
                click.echo(f"Total studies: {stats['total_studies']}")
                click.echo(f"Total pages (with {stats['page_size']} studies per page): {stats['total_pages']}")
                if export:
                    with open(export, 'w') as f:
                        json.dump(stats, f, indent=2)
                    click.echo(f"Results exported to {export}")
            else:
                result = search_trials(condition, None, limit, None, use_cache=True)
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
    if process_criteria:
        # Handle both cases: result could be a single study or a dict with 'studies' key
        if isinstance(result, dict):
            if 'studies' in result and len(result['studies']) > 0:
                study = result['studies'][0]
            else:
                # If it's already a study object
                study = result
        else:
            # If result is directly a study object
            study = result
            
        if 'protocolSection' in study:
            protocol_section = study['protocolSection']
            if 'eligibilityModule' in protocol_section:
                eligibility_module = protocol_section['eligibilityModule']
                if 'eligibilityCriteria' in eligibility_module:
                    criteria_text = eligibility_module['eligibilityCriteria']
                    if criteria_text:
                        try:
                            # Process through the new dynamic extraction pipeline
                            pipeline = DynamicExtractionPipeline()
                            
                            # Ensure criteria_text is a string
                            if isinstance(criteria_text, list):
                                criteria_text = ' '.join(str(item) for item in criteria_text)
                            elif not isinstance(criteria_text, str):
                                criteria_text = str(criteria_text)
                            
                            # Create section context for source tracking
                            section_context = {
                                'source_type': 'clinical_trial',
                                'source_id': study['protocolSection']['identificationModule'].get('nctId', ''),
                                'content_type': 'eligibility_criteria',
                                'criteria_section': 'inclusion' if 'inclusion' in criteria_text.lower() else 'exclusion' if 'exclusion' in criteria_text.lower() else 'unspecified',
                                'source_section': 'eligibilityModule',
                                'source_document': 'protocolSection'
                            }
                            
                            # Process criteria through the dynamic pipeline
                            pipeline_result = pipeline.process_eligibility_criteria(criteria_text, section_context)
                            
                            # Add results to the study data
                            if 'mcodeResults' not in result:
                                result['mcodeResults'] = {}
                            
                            result['mcodeResults'] = {
                                'extracted_entities': pipeline_result.extracted_entities,
                                'mcode_mappings': pipeline_result.mcode_mappings,
                                'source_references': pipeline_result.source_references,
                                'validation': pipeline_result.validation_results,
                                'metadata': pipeline_result.metadata,
                                'error': pipeline_result.error
                            }
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
            
            # Show pagination info if available
            if 'pagination' in result:
                pagination = result['pagination']
                max_results = pagination.get('max_results', 100)
                click.echo(f"Showing {len(studies)} results")
                click.echo(f"Results per page: {max_results}")
            
            for i, study in enumerate(studies):
                # Extract NCT ID and title from the new structure
                protocol_section = study.get('protocolSection', {})
                identification_module = protocol_section.get('identificationModule', {})
                nct_id = identification_module.get('nctId', 'Unknown')
                title = identification_module.get('briefTitle', 'No title')
                
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


def calculate_total_studies(search_expr: str, fields=None, page_size: int = 100):
    """
    Calculate the total number of studies matching the search expression
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        page_size: Number of studies per page (default: 100)
        
    Returns:
        Dictionary containing total studies count and number of pages
    """
    # Initialize config and cache manager
    config = Config()
    cache_manager = CacheManager(config)
    
    # Create cache key
    cache_key = f"total_studies:{search_expr}:{page_size}"
    
    # Try to get from cache first
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Rate limiting
        time.sleep(config.rate_limit_delay)
        
        # Initialize pytrials client
        ct = ClinicalTrials()
        
        # Add countTotal=true to the search expression
        # Use a small page size since we only need the totalCount field
        modified_search_expr = f"{search_expr}&countTotal=true"
        
        # Use pytrials client to get the study count
        result = ct.get_study_fields(
            search_expr=modified_search_expr,
            fields=["NCTId"],  # Just get one field to minimize data
            max_studies=1,
            fmt="json"
        )
        
        # Extract the total count from the response
        total_studies = result.get('totalCount', 0)
        
        # Calculate pages
        total_pages = (total_studies + page_size - 1) // page_size if total_studies > 0 else 0
        
        result = {
            'total_studies': total_studies,
            'total_pages': total_pages,
            'page_size': page_size
        }
        
        # Cache the result
        cache_manager.set(cache_key, result)
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


if __name__ == '__main__':
    main()