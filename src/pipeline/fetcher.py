import click
import json
import sys
import time
import hashlib
import os
import re
import requests
from functools import lru_cache
# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import Config
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.pipeline.prompt_model_interface import create_configured_pipeline
from src.utils.logging_config import get_logger
import diskcache

# Get logger instance
# Initialize disk cache
cache = diskcache.Cache('./cache/fetcher_cache')
logger = get_logger(__name__)

# ClinicalTrials.gov API field names
CLINICAL_TRIALS_FIELDS = [
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


@cache.memoize()
def _cached_search_trials(search_expr: str, fields_str: str, max_results: int, page_token: str):
    """
    Cached search for clinical trials matching the expression with pagination support
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields_str: Comma-separated list of fields to retrieve
        max_results: Maximum number of results to return
        page_token: Page token for pagination
        
    Returns:
        Dictionary containing search results with pagination metadata
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
        # Parse fields string back to list
        fields = fields_str.split(',') if fields_str else None
        
        # Use clinical trials fields
        api_fields = fields if fields else CLINICAL_TRIALS_FIELDS
        
        # Use direct API call to ClinicalTrials.gov API v2 for proper token-based pagination
        import requests
        
        # Build the API URL from config
        base_url = config.get_clinical_trials_base_url()
        params = {
            "format": "json",
            "query.term": search_expr,
            "pageSize": max_results,
            "fields": ",".join(api_fields)
        }
        
        # Add page token if provided
        if page_token and page_token != "None":
            params["pageToken"] = page_token
        
        # Make the API request
        response = requests.get(base_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        
        # Add pagination metadata to the result
        if isinstance(result, dict):
            studies_count = len(result.get('studies', []))
            logger.info(f"_cached_search_trials: API returned {studies_count} studies for search '{search_expr}'")
            logger.info(f"   📊 Request params - pageSize: {max_results}, fields: {len(api_fields)}")
            result['pagination'] = {
                'max_results': max_results
            }
            
            # Add nextPageToken if it exists in the response
            if 'nextPageToken' in result:
                result['nextPageToken'] = result['nextPageToken']
                logger.debug(f"_cached_search_trials: nextPageToken found in response")
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


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
    # Convert fields to string for caching
    fields_str = ','.join(fields) if fields else ""
    page_token_str = page_token if page_token else "None"
    
    # Call the cached function if caching is enabled
    if use_cache:
        logger.info(f"search_trials: Using cached search for search_expr='{search_expr}', max_results={max_results}, page_token={page_token}")
        return _cached_search_trials(search_expr, fields_str, max_results, page_token_str)
    else:
        # Create a temporary version without caching
        logger.info(f"search_trials: Performing uncached search for search_expr='{search_expr}', max_results={max_results}, page_token={page_token}")
        return _cached_search_trials.__wrapped__(search_expr, fields_str, max_results, page_token_str)


@cache.memoize()
def _cached_get_full_study(nct_id: str):
    """
    Cached get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT00000")
        
    Returns:
        Dictionary containing the full study record
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Initialize config
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
        # Log the NCT ID we're trying to fetch
        logger.info(f"_cached_get_full_study: Attempting to fetch full study details for NCT ID: {nct_id}")
        
        # Use direct API call to ClinicalTrials.gov API v2
        import requests
        
        # Build the API URL from config
        base_url = config.get_clinical_trials_base_url()
        # For fetching a specific study, we need to append the NCT ID to the base URL
        study_url = f"{base_url}/{nct_id}"
        
        params = {
            "format": "json"
        }
        
        # Make the API request
        logger.info(f"_cached_get_full_study: Calling ClinicalTrials.gov API v2 for NCT ID: {nct_id}")
        response = requests.get(study_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        logger.info(f"_cached_get_full_study: API call completed for NCT ID: {nct_id}")
        logger.info(f"   📊 Response size: {len(str(result))} characters")
        if isinstance(result, dict):
            protocol_section = result.get('protocolSection', {})
            if protocol_section:
                identification_module = protocol_section.get('identificationModule', {})
                if identification_module:
                    brief_title = identification_module.get('briefTitle', 'No title')
                    logger.info(f"   📋 Study title: {brief_title[:100]}{'...' if len(brief_title) > 100 else ''}")
        
        # Check if we got a valid response
        if result is None:
            raise ValueError(f"No response received from API for NCT ID {nct_id}")
        
        # Verify result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(f"Invalid response format from API for NCT ID {nct_id}: expected dict, got {type(result)}")
        
        # For v2 API, the study data is directly in the result, not in a 'studies' array
        study = result
        
        return study
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed for NCT ID {nct_id}: {str(e)}")


def get_full_study(nct_id: str, use_cache: bool = True):
    """
    Get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT00000")
        use_cache: Whether to use caching (default: True)
        
    Returns:
        Dictionary containing the full study record
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Call the cached function if caching is enabled
    if use_cache:
        logger.info(f"get_full_study: Using cached study for NCT ID {nct_id}")
        return _cached_get_full_study(nct_id)
    else:
        # Create a temporary version without caching
        logger.info(f"get_full_study: Performing uncached study fetch for NCT ID {nct_id}")
        return _cached_get_full_study.__wrapped__(nct_id)


def get_study_fields():
    """
    Get information about available study fields from the ClinicalTrials.gov API v2
    
    Returns:
        Dictionary containing available fields information
    """
    try:
        # Initialize config
        config = Config()
        
        # Use direct API call to ClinicalTrials.gov API v2
        import requests
        
        # For v2 API, we need to make a search request with a small result set to get field info
        base_url = config.get_clinical_trials_base_url()
        params = {
            "format": "json",
            "query.term": "cancer", # Generic search term
            "pageSize": 1,  # Just get one study
            "fields": "NCTId"  # Minimal fields
        }
        
        # Make the API request to get the structure
        response = requests.get(base_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        
        # Return the structure information
        return {
            "fields": CLINICAL_TRIALS_FIELDS
        }
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


@click.command()
@click.option('--condition', '-c', help='Condition to search for (e.g., "breast cancer")')
@click.option('--nct-id', '-n', help='Specific NCT ID to fetch (e.g., "NCT000000")')
@click.option('--limit', '-l', default=10, help='Maximum number of results to return')
@click.option('--count', '--calculate-count', is_flag=True, help='Calculate total number of studies and pages')
@click.option('--export', '-e', type=click.Path(), help='Export results to JSON file')
@click.option('--process-criteria', '-p', is_flag=True, help='Process eligibility criteria with NLP engine')
@click.option('--process-trial', '-t', is_flag=True, help='Process complete trial with NLP engine')
def main(condition, nct_id, limit, count, export, process_criteria, process_trial):
    """
    Clinical Trial Data Fetcher for mCODE Translator
    
    Examples:
      python fetcher.py --condition "breast cancer" --limit 10
      python fetcher.py --nct-id NCT00000000
      python fetcher.py --condition "lung cancer" --export results.json
      python fetcher.py --nct-id NCT000000 --process-criteria
      python fetcher.py --condition "breast cancer" --count
    """
    
    try:
        if nct_id:
            # Fetch a specific trial by NCT ID
            click.echo(f"Fetching trial {nct_id}...")
            logger.info(f"main: Fetching trial {nct_id}")
            result = get_full_study(nct_id)
            logger.info(f"main: Fetched trial {nct_id} successfully")
            display_single_study(result, export, process_criteria, process_trial)
        elif condition:
            # Search for trials by condition
            click.echo(f"Searching for trials matching '{condition}'...")
            if count:
                # Calculate total studies and pages
                click.echo("Calculating total number of studies...")
                logger.info(f"main: Calculating total studies for condition '{condition}'")
                stats = calculate_total_studies(condition)
                logger.info(f"main: Calculation completed - {stats['total_studies']:,} studies, {stats['total_pages']:,} pages")
                click.echo(f"Total studies: {stats['total_studies']:,}")
                click.echo(f"Total pages (with {stats['page_size']} studies per page): {stats['total_pages']:,}")
                if export:
                    with open(export, 'w') as f:
                        json.dump(stats, f, indent=2)
                    click.echo(f"Results exported to {export}")
            else:
                logger.info(f"main: Searching for trials matching '{condition}' with limit {limit}")
                result = search_trials(condition, None, limit, None, use_cache=True)
                logger.info(f"main: Search completed successfully")
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


def display_single_study(result, export_path=None, process_criteria=False, process_trial=False):
    """
    Display a single study result or export to file
    
    Args:
        result: Study result to display
        export_path: Optional path to export JSON file
        process_criteria: Whether to process eligibility criteria with NLP
    """
    # Process complete trial with NLP if requested
    if process_trial:
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
            
        try:
            # Import the strict dynamic extraction pipeline
            from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
            
            # Create a configured pipeline using the prompt/model interface
            pipeline = StrictDynamicExtractionPipeline()
            
            # Process the complete trial through the pipeline
            # This will raise exceptions for missing assets or configuration issues
            pipeline_result = pipeline.process_clinical_trial(result)
            
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
            
            # Log detailed processing information
            logger.info(f"✅ NLP processing completed for complete trial")
            logger.info(f"   📊 Extracted {len(pipeline_result.extracted_entities)} entities")
            logger.info(f"   🗺️  Mapped {len(pipeline_result.mcode_mappings)} mCODE elements")
            logger.info(f"   📈 Validation score: {pipeline_result.validation_results.get('compliance_score', 0):.2%}")
            
            # Log token usage if available
            if hasattr(pipeline_result, 'metadata') and pipeline_result.metadata:
                metadata = pipeline_result.metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    logger.info(f"   📊 Token usage - Prompt: {token_usage.get('prompt_tokens', 'N/A')}, "
                               f"Completion: {token_usage.get('completion_tokens', 'N/A')}, "
                               f"Total: {token_usage.get('total_tokens', 'N/A')}")
                
                # Log aggregate token usage if available
                if 'aggregate_token_usage' in metadata:
                    agg_usage = metadata['aggregate_token_usage']
                    if isinstance(agg_usage, dict):
                        logger.info(f"   📊 Aggregate usage - Prompt: {agg_usage.get('prompt_tokens', 0)}, "
                                   f"Completion: {agg_usage.get('completion_tokens', 0)}, "
                                   f"Total: {agg_usage.get('total_tokens', 0)}")
            
            # Show sample entities if available
            if pipeline_result.extracted_entities:
                logger.info("   🔍 Sample extracted entities:")
                for i, entity in enumerate(pipeline_result.extracted_entities[:3]):
                    logger.info(f"     {i+1}. {entity.get('text', 'No text')} "
                               f"({entity.get('type', 'Unknown type')}) - "
                               f"Confidence: {entity.get('confidence', 0.0):.2f}")
            
            # Show sample mappings if available
            if pipeline_result.mcode_mappings:
                logger.info("   🔍 Sample mCODE mappings:")
                for i, mapping in enumerate(pipeline_result.mcode_mappings[:3]):
                    logger.info(f"     {i+1}. {mapping.get('resourceType', 'Unknown')} - "
                               f"{mapping.get('element_name', 'No name')} - "
                               f"Confidence: {mapping.get('mapping_confidence', 0.0):.2f}")
        except Exception as e:
            # In strict mode, we re-raise the exception to fail fast
            # This ensures that any missing assets or configuration issues are immediately reported
            raise ClinicalTrialsAPIError(f"STRICT NLP processing failed: {str(e)}") from e
    
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
                            # Use strict implementation without fallbacks
                            # Create a configured pipeline using the prompt/model interface
                            pipeline = create_configured_pipeline()
                            
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
                            # This will raise exceptions for missing assets or configuration issues
                            pipeline_result = pipeline.process_eligibility_criteria(criteria_text, section_context)
                            
                            # Log detailed processing information
                            logger.info(f"✅ NLP processing completed for {section_context['source_id']}")
                            logger.info(f"   📊 Extracted {len(pipeline_result.extracted_entities)} entities")
                            logger.info(f"   🗺️  Mapped {len(pipeline_result.mcode_mappings)} mCODE elements")
                            logger.info(f"   📈 Validation score: {pipeline_result.validation_results.get('compliance_score', 0):.2%}")
                            
                            # Log token usage if available
                            if hasattr(pipeline_result, 'metadata') and pipeline_result.metadata:
                                metadata = pipeline_result.metadata
                                if 'token_usage' in metadata:
                                    token_usage = metadata['token_usage']
                                    logger.info(f"   📊 Token usage - Prompt: {token_usage.get('prompt_tokens', 'N/A')}, "
                                               f"Completion: {token_usage.get('completion_tokens', 'N/A')}, "
                                               f"Total: {token_usage.get('total_tokens', 'N/A')}")
                                
                                # Log aggregate token usage if available
                                if 'aggregate_token_usage' in metadata:
                                    agg_usage = metadata['aggregate_token_usage']
                                    if isinstance(agg_usage, dict):
                                        logger.info(f"   📊 Aggregate usage - Prompt: {agg_usage.get('prompt_tokens', 0)}, "
                                                   f"Completion: {agg_usage.get('completion_tokens', 0)}, "
                                                   f"Total: {agg_usage.get('total_tokens', 0)}")
                            
                            # Show sample entities if available
                            if pipeline_result.extracted_entities:
                                logger.info("   🔍 Sample extracted entities:")
                                for i, entity in enumerate(pipeline_result.extracted_entities[:3]):
                                    logger.info(f"     {i+1}. {entity.get('text', 'No text')} "
                                               f"({entity.get('type', 'Unknown type')}) - "
                                               f"Confidence: {entity.get('confidence', 0.0):.2f}")
                            
                            # Show sample mappings if available
                            if pipeline_result.mcode_mappings:
                                logger.info("   🔍 Sample mCODE mappings:")
                                for i, mapping in enumerate(pipeline_result.mcode_mappings[:3]):
                                    logger.info(f"     {i+1}. {mapping.get('resourceType', 'Unknown')} - "
                                               f"{mapping.get('element_name', 'No name')} - "
                                               f"Confidence: {mapping.get('mapping_confidence', 0.0):.2f}")
                            
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
                            # In strict mode, we re-raise the exception to fail fast
                            # This ensures that any missing assets or configuration issues are immediately reported
                            raise ClinicalTrialsAPIError(f"STRICT NLP processing failed: {str(e)}") from e
    
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


@cache.memoize()
def _cached_calculate_total_studies(search_expr: str, fields_str: str, page_size: int):
    """
    Cached calculate the total number of studies matching the search expression
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields_str: Comma-separated list of fields to retrieve
        page_size: Number of studies per page
        
    Returns:
        Dictionary containing total studies count and number of pages
    """
    # Initialize config
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
        # Parse fields string back to list
        fields = fields_str.split(',') if fields_str else None
        
        # Use direct API call to ClinicalTrials.gov API v2
        import requests
        
        # Build the API URL from config
        base_url = config.get_clinical_trials_base_url()
        params = {
            "format": "json",
            "query.term": search_expr,
            "pageSize": 1,  # Just get one study to minimize data
            "countTotal": "true"  # Request total count
        }
        
        # Make the API request
        response = requests.get(base_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        
        # Extract the total count from the response
        total_studies = result.get('totalCount', 0)
        
        # Calculate pages
        total_pages = (total_studies + page_size - 1) // page_size if total_studies > 0 else 0
        
        result = {
            'total_studies': total_studies,
            'total_pages': total_pages,
            'page_size': page_size
        }
        
        # Log detailed statistics
        logger.info(f"_cached_calculate_total_studies: Search '{search_expr}' statistics")
        logger.info(f"   📊 Total studies: {total_studies:,}")
        logger.info(f"   📄 Total pages: {total_pages:,} (at {page_size} studies/page)")
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


def calculate_total_studies(search_expr: str, fields=None, page_size: int = 100, use_cache: bool = True):
    """
    Calculate the total number of studies matching the search expression
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        page_size: Number of studies per page (default: 100)
        use_cache: Whether to use caching (default: True)
        
    Returns:
        Dictionary containing total studies count and number of pages
    """
    # Convert fields to string for caching
    fields_str = ','.join(fields) if fields else ""
    
    # Call the cached function if caching is enabled
    if use_cache:
        logger.info(f"calculate_total_studies: Using cached calculation for search_expr='{search_expr}', page_size={page_size}")
        return _cached_calculate_total_studies(search_expr, fields_str, page_size)
    else:
        # Create a temporary version without caching
        logger.info(f"calculate_total_studies: Performing uncached calculation for search_expr='{search_expr}', page_size={page_size}")
        return _cached_calculate_total_studies.__wrapped__(search_expr, fields_str, page_size)


if __name__ == '__main__':
    main()