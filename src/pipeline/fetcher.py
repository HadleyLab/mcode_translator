"""
Core Clinical Trial Fetcher Functions

Essential functions for fetching clinical trial data from ClinicalTrials.gov API.
Used by concurrent processing and CLI tools.
"""

import time
import requests
import json
from typing import Dict, Any, List, Optional

from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.utils.api_manager import UnifiedAPIManager

# Get logger instance
logger = get_logger(__name__)

# Initialize API manager and cache
api_manager = UnifiedAPIManager()
clinical_trials_cache = api_manager.get_cache("clinical_trials")


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
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
        # Parse fields string back to list
        fields = fields_str.split(',') if fields_str else None
        
        # Use clinical trials fields
        CLINICAL_TRIALS_FIELDS = [
            "NCTId", "BriefTitle", "Condition", "OverallStatus",
            "BriefSummary", "StartDate", "CompletionDate"
        ]
        api_fields = fields if fields else CLINICAL_TRIALS_FIELDS
        
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
            logger.info(f"API returned {studies_count} studies for search '{search_expr}'")
            result['pagination'] = {
                'max_results': max_results
            }
            
            # Add nextPageToken if it exists in the response
            if 'nextPageToken' in result:
                result['nextPageToken'] = result['nextPageToken']
        
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
    logger.info(f"Performing search for '{search_expr}', max_results={max_results}")
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
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
        logger.info(f"Fetching full study details for NCT ID: {nct_id}")
        
        # Build the API URL from config
        base_url = config.get_clinical_trials_base_url()
        study_url = f"{base_url}/{nct_id}"
        
        params = {"format": "json"}
        
        # Make the API request
        response = requests.get(study_url, params=params, timeout=config.get_request_timeout())
        response.raise_for_status()
        result = response.json()
        
        logger.info(f"API call completed for NCT ID: {nct_id}")
        
        # Check if we got a valid response
        if result is None:
            raise ValueError(f"No response received from API for NCT ID {nct_id}")
        
        if not isinstance(result, dict):
            raise ValueError(f"Invalid response format from API for NCT ID {nct_id}: expected dict, got {type(result)}")
        
        return result
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
    
    # If not cached, call the function
    logger.info(f"Fetching study for NCT ID {nct_id}")
    result = _get_full_study(nct_id)
    
    # Store result in cache with default TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=None)
    
    return result


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
    config = Config()
    
    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())
        
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
        
        logger.info(f"Search '{search_expr}' statistics: {total_studies:,} studies, {total_pages:,} pages")
        
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
    logger.info(f"Calculating total studies for '{search_expr}', page_size={page_size}")
    result = _calculate_total_studies(search_expr, fields_str, page_size)
    
    # Store result in cache with default TTL
    clinical_trials_cache.set_by_key(result, cache_key_data, ttl=None)
    
    return result


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
        logger.info("Processing eligibility criteria with mCODE pipeline")
        logger.info(f"Criteria length: {len(criteria_text)} characters")
        
        # Import here to avoid circular imports
        from src.pipeline.mcode_pipeline import McodePipeline
        
        # Create pipeline
        pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)
        
        # Create mock trial structure with just the eligibility criteria
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
        
        logger.info(f"mCODE processing completed - mapped {len(result.mcode_mappings)} mCODE elements")
        
        return result
        
    except Exception as e:
        logger.error(f"mCODE criteria processing failed: {str(e)}")
        raise