import asyncio
import hashlib
import json
import os
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict

import click
import requests

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils import (Config, ModelLoader, PromptLoader, UnifiedAPIManager,
                       get_logger)

# Get logger instance
logger = get_logger(__name__)

# Initialize API manager and cache
api_manager = UnifiedAPIManager()
clinical_trials_cache = api_manager.get_cache("clinical_trials")

# Initialize prompt and model loaders
prompt_loader = PromptLoader()
model_loader = ModelLoader()

def process_eligibility_criteria_with_mcode(criteria_text, prompt_name="direct_mcode", model_name=None):
    """
    Process eligibility criteria text directly with mCODE pipeline
    
    Args:
        criteria_text: Eligibility criteria text to process
        prompt_name: Prompt name to use for direct mCODE mapping
        model_name: Model name to use for processing
        
    Returns:
        Dictionary containing processing results similar to PipelineResult
    """
    try:
        logger.info("📋 Processing eligibility criteria with mCODE pipeline")
        logger.info(f"   📝 Criteria length: {len(criteria_text)} characters")
        
        # Create pipeline
        pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)
        
        # Since McodePipeline expects trial data, we need to create a mock trial structure
        # with just the eligibility criteria
        mock_trial_data = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": 'eligibility_criteria',
                    "briefTitle": "Eligibility Criteria Processing"
                },
                "eligibilityModule": {
                    "eligibilityCriteria": criteria_text
                }
            }
        }
        
        # Process through pipeline
        result = pipeline.process_clinical_trial(mock_trial_data)
        
        logger.info(f"✅ mCODE processing completed")
        logger.info(f"   📊 Mapped {len(result.mcode_mappings)} mCODE elements")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ mCODE criteria processing failed: {str(e)}")
        raise


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


def _search_trials(search_expr: str, fields_str: str, max_results: int, page_token: str):
    """
    Search for clinical trials matching the expression with pagination support
    
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
            logger.info(f"_search_trials: API returned {studies_count} studies for search '{search_expr}'")
            logger.info(f"   📊 Request params - pageSize: {max_results}, fields: {len(api_fields)}")
            result['pagination'] = {
                'max_results': max_results
            }
            
            # Add nextPageToken if it exists in the response
            if 'nextPageToken' in result:
                result['nextPageToken'] = result['nextPageToken']
                logger.debug(f"_search_trials: nextPageToken found in response")
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


def search_trials(search_expr: str, fields=None, max_results: int = 100, page_token: str = None):
    """
    Search for clinical trials matching the expression with pagination support
    
    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        max_results: Maximum number of results to return (default: 100)
        page_token: Page token for pagination (default: None)
        
    Returns:
        Dictionary containing search results with pagination metadata
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Convert fields to string for caching
    fields_str = ','.join(fields) if fields else ""
    page_token_str = page_token if page_token else "None"
    
    # Generate cache key
    cache_key_data = {
        "function": "search_trials",
        "search_expr": search_expr,
        "fields_str": fields_str,
        "max_results": max_results,
        "page_token_str": page_token_str
    }
    
    # Try to get cached result
    cached_result = clinical_trials_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        logger.info("Cache HIT for search_trials")
        return cached_result
    
    # If not cached, call the search function
    logger.info(f"search_trials: Performing search for search_expr='{search_expr}', max_results={max_results}, page_token={page_token}")
    result = _search_trials(search_expr, fields_str, max_results, page_token_str)
    
    # Store result in cache with default TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=None)
    
    return result


def _get_full_study(nct_id: str):
    """
    Get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT0000")
        
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
        logger.info(f"_get_full_study: Attempting to fetch full study details for NCT ID: {nct_id}")
        
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
        logger.info(f"_get_full_study: Calling ClinicalTrials.gov API v2 for NCT ID: {nct_id}")
        response = requests.get(study_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        logger.info(f"_get_full_study: API call completed for NCT ID: {nct_id}")
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


def get_full_study(nct_id: str):
    """
    Get complete study record for a specific trial
    
    Args:
        nct_id: NCT ID of the clinical trial (e.g., "NCT00")
        
    Returns:
        Dictionary containing the full study record
        
    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    # Generate cache key
    cache_key_data = {
        "function": "get_full_study",
        "nct_id": nct_id
    }
    
    # Try to get cached result
    cached_result = clinical_trials_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        logger.info("Cache HIT for get_full_study")
        return cached_result
    
    # If not cached, call the search function
    logger.info(f"get_full_study: Fetching study for NCT ID {nct_id}")
    result = _get_full_study(nct_id)
    
    # Store result in cache with default TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=None)
    
    return result


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
@click.option('--nct-ids', help='Comma-separated list of NCT IDs to process concurrently')
@click.option('--limit', '-l', default=10, help='Maximum number of results to return')
@click.option('--count', '--calculate-count', is_flag=True, help='Calculate total number of studies and pages')
@click.option('--export', '-e', type=click.Path(), help='Export results to JSON file')
@click.option('--process-criteria', '-p', is_flag=True, help='Process eligibility criteria with NLP engine')
@click.option('--process-trial', '-t', is_flag=True, help='Process complete trial with NLP engine')
@click.option('--model', '-m', help='Model to use for mCODE processing (e.g., "gpt-4o", "deepseek-coder")')
@click.option('--prompt', help='Prompt to use for mCODE processing (e.g., "direct_mcode_simple", "direct_mcode_comprehensive")')
@click.option('--concurrent', is_flag=True, help='Use concurrent processing for improved performance')
@click.option('--workers', default=5, help='Number of concurrent workers (default: 5)')
@click.option('--batch-size', default=10, help='Batch size for concurrent processing (default: 10)')
@click.option('--progress', is_flag=True, default=True, help='Show progress updates for concurrent processing')
def main(condition, nct_id, nct_ids, limit, count, export, process_criteria, process_trial, model, prompt, concurrent, workers, batch_size, progress):
    """
    Clinical Trial Data Fetcher for mCODE Translator
    
    Examples:
      # Basic operations
      python fetcher.py --condition "breast cancer" --limit 10
      python fetcher.py --nct-id NCT00000000
      python fetcher.py --condition "lung cancer" --export results.json
      
      # Sequential mCODE processing
      python fetcher.py --nct-id NCT000000 --process-criteria
      python fetcher.py --condition "breast cancer" --process-trial --model gpt-4o
      
      # Concurrent processing (NEW!)
      python fetcher.py --condition "breast cancer" --concurrent --workers 10 --process-trial
      python fetcher.py --nct-ids "NCT001,NCT002,NCT003" --concurrent --workers 5
      python fetcher.py --condition "lung cancer" --concurrent --process-criteria --batch-size 20
      
      # Performance and statistics
      python fetcher.py --condition "breast cancer" --count
      python fetcher.py --condition "cancer" --concurrent --workers 8 --progress
    """
    
    try:
        if nct_id:
            # Fetch a specific trial by NCT ID
            click.echo(f"Fetching trial {nct_id}...")
            logger.info(f"main: Fetching trial {nct_id}")
            result = get_full_study(nct_id)
            logger.info(f"main: Fetched trial {nct_id} successfully")
            display_single_study(result, export, process_criteria, process_trial, model, prompt)
        elif nct_ids:
            # Process multiple NCT IDs concurrently
            nct_id_list = [nct_id.strip() for nct_id in nct_ids.split(',')]
            click.echo(f"Processing {len(nct_id_list)} trials concurrently...")
            logger.info(f"main: Processing {len(nct_id_list)} trials with {workers} workers")
            
            # Run concurrent processing
            result = asyncio.run(run_concurrent_trial_processing(
                nct_ids=nct_id_list,
                workers=workers,
                batch_size=batch_size,
                process_criteria=process_criteria,
                process_trial=process_trial,
                model=model,
                prompt=prompt,
                export_path=export,
                progress_updates=progress
            ))
            
            # Display summary
            display_concurrent_results(result)
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
            elif concurrent and (process_criteria or process_trial):
                # Use concurrent processing for search and mCODE processing
                click.echo(f"Using concurrent processing with {workers} workers...")
                logger.info(f"main: Concurrent search and processing for '{condition}' with {workers} workers")
                
                result = asyncio.run(run_concurrent_search_and_process(
                    condition=condition,
                    limit=limit,
                    workers=workers,
                    batch_size=batch_size,
                    process_criteria=process_criteria,
                    process_trial=process_trial,
                    model=model,
                    prompt=prompt,
                    export_path=export,
                    progress_updates=progress
                ))
                
                # Display summary
                display_concurrent_results(result)
            else:
                # Traditional sequential processing
                logger.info(f"main: Searching for trials matching '{condition}' with limit {limit}")
                result = search_trials(condition, None, limit, None)
                logger.info(f"main: Search completed successfully")
                display_search_results(result, export)
        else:
            # Show help if no arguments provided
            click.echo("Please specify either a condition to search for, an NCT ID to fetch, or a list of NCT IDs.")
            click.echo("Use --help for more information.")
            sys.exit(1)
            
    except ClinicalTrialsAPIError as e:
        click.echo(f"API Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected Error: {str(e)}", err=True)
        sys.exit(1)


def display_single_study(result, export_path=None, process_criteria=False, process_trial=False, model=None, prompt=None):
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
            # Create a configured mCODE pipeline 
            prompt_to_use = prompt or "direct_mcode"
            pipeline = McodePipeline(prompt_name=prompt_to_use, model_name=model)
            
            # Process the complete trial through the pipeline
            # This will raise exceptions for missing assets or configuration issues
            pipeline_result = pipeline.process_clinical_trial(result)
            
            # Add results to the study data
            if 'McodeResults' not in result:
                result['McodeResults'] = {}
            
            result['McodeResults'] = {
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
            raise ClinicalTrialsAPIError(f"STRICT mCODE pipeline processing failed: {str(e)}") from e
    
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
                            
                            # Process eligibility criteria with mCODE pipeline
                            prompt_to_use = prompt or "direct_mcode"
                            pipeline_result = process_eligibility_criteria_with_mcode(criteria_text, prompt_name=prompt_to_use, model_name=model)
                            
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
                            if 'McodeResults' not in result:
                                result['McodeResults'] = {}
                            
                            result['McodeResults'] = {
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
                            raise ClinicalTrialsAPIError(f"STRICT mCODE criteria processing failed: {str(e)}") from e
    
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


def _calculate_total_studies(search_expr: str, fields_str: str, page_size: int):
    """
    Calculate the total number of studies matching the search expression
    
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
        logger.info(f"_calculate_total_studies: Search '{search_expr}' statistics")
        logger.info(f"   📊 Total studies: {total_studies:,}")
        logger.info(f"   📄 Total pages: {total_pages:,} (at {page_size} studies/page)")
        
        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


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
    # Convert fields to string for caching
    fields_str = ','.join(fields) if fields else ""
    
    # Generate cache key
    cache_key_data = {
        "function": "calculate_total_studies",
        "search_expr": search_expr,
        "fields_str": fields_str,
        "page_size": page_size
    }
    
    # Try to get cached result
    cached_result = clinical_trials_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        logger.info("Cache HIT for calculate_total_studies")
        return cached_result
    
    # If not cached, call the calculate function
    logger.info(f"calculate_total_studies: Performing calculation for search_expr='{search_expr}', page_size={page_size}")
    result = _calculate_total_studies(search_expr, fields_str, page_size)
    
    # Store result in cache with default TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=None)
    
    return result


# Concurrent processing functions
async def run_concurrent_search_and_process(
    condition: str,
    limit: int = 100,
    workers: int = 5,
    batch_size: int = 10,
    process_criteria: bool = False,
    process_trial: bool = False,
    model: str = None,
    prompt: str = None,
    export_path: str = None,
    progress_updates: bool = True
):
    """Run concurrent search and processing using the ConcurrentFetcher"""
    from src.pipeline.concurrent_fetcher import concurrent_search_and_process
    
    return await concurrent_search_and_process(
        condition=condition,
        limit=limit,
        max_workers=workers,
        batch_size=batch_size,
        process_criteria=process_criteria,
        process_trials=process_trial,
        model_name=model,
        prompt_name=prompt or "direct_mcode",
        export_path=export_path,
        progress_updates=progress_updates
    )


async def run_concurrent_trial_processing(
    nct_ids: list,
    workers: int = 5,
    batch_size: int = 10,
    process_criteria: bool = False,
    process_trial: bool = False,
    model: str = None,
    prompt: str = None,
    export_path: str = None,
    progress_updates: bool = True
):
    """Run concurrent processing of specific trial NCT IDs"""
    from src.pipeline.concurrent_fetcher import concurrent_process_trials
    
    return await concurrent_process_trials(
        nct_ids=nct_ids,
        max_workers=workers,
        batch_size=batch_size,
        process_criteria=process_criteria,
        process_trials=process_trial,
        model_name=model,
        prompt_name=prompt or "direct_mcode",
        export_path=export_path,
        progress_updates=progress_updates
    )


def display_concurrent_results(result):
    """Display results from concurrent processing"""
    from src.pipeline.concurrent_fetcher import ConcurrentProcessingResult
    
    click.echo("\n" + "="*60)
    click.echo("📊 CONCURRENT PROCESSING RESULTS")
    click.echo("="*60)
    
    # Summary statistics
    click.echo(f"📋 Total trials processed: {result.total_trials}")
    click.echo(f"✅ Successful: {result.successful_trials}")
    click.echo(f"❌ Failed: {result.failed_trials}")
    
    if result.total_trials > 0:
        success_rate = (result.successful_trials / result.total_trials) * 100
        click.echo(f"📈 Success rate: {success_rate:.1f}%")
    
    # Performance metrics
    click.echo(f"⏱️  Total duration: {result.duration_seconds:.2f} seconds")
    
    if result.duration_seconds > 0:
        processing_rate = result.total_trials / result.duration_seconds
        click.echo(f"🚀 Processing rate: {processing_rate:.1f} trials/second")
    
    # Task queue statistics
    if result.task_stats:
        stats = result.task_stats
        click.echo(f"👥 Workers used: {stats.get('workers_running', 'N/A')}")
        click.echo(f"📊 Queue completion rate: {stats.get('completion_rate', 0.0)*100:.1f}%")
    
    # Sample results
    if result.results:
        click.echo(f"\n🔍 Sample successful trials:")
        for i, trial in enumerate(result.results[:3]):  # Show first 3
            nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')
            title = trial.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'No title')
            click.echo(f"  {i+1}. {nct_id}: {title[:80]}{'...' if len(title) > 80 else ''}")
            
            # Show mCODE results if available
            if 'McodeResults' in trial:
                mcode_results = trial['McodeResults']
                entities_count = len(mcode_results.get('extracted_entities', []))
                mappings_count = len(mcode_results.get('mcode_mappings', []))
                click.echo(f"     🔬 mCODE: {entities_count} entities, {mappings_count} mappings")
    
    # Sample errors
    if result.errors:
        click.echo(f"\n❌ Sample errors:")
        for i, error in enumerate(result.errors[:3]):  # Show first 3 errors
            nct_id = error.get('nct_id', 'Unknown')
            error_msg = error.get('error', 'Unknown error')
            click.echo(f"  {i+1}. {nct_id}: {error_msg[:100]}{'...' if len(error_msg) > 100 else ''}")
    
    click.echo("="*60)


if __name__ == '__main__':
    main()
