"""
Simple Clinical Trials Fetcher Utility
Provides basic functions for fetching clinical trial data from ClinicalTrials.gov API.
"""

import json
import time
from typing import Any, Dict, List, Optional

import requests

from src.utils.api_manager import APIManager
from src.utils.config import Config
from src.utils.logging_config import get_logger

# Get logger instance
logger = get_logger(__name__)

# Initialize API manager and cache
api_manager = APIManager()
clinical_trials_cache = api_manager.get_cache("clinical_trials")


class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass


def search_trials(
    search_expr: str, fields=None, max_results: int = 100, page_token: str = None
):
    """
    Search for clinical trials matching the expression

    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve (default: None for all fields)
        max_results: Maximum number of results to return (default: 100)
        page_token: Page token for pagination (default: None)

    Returns:
        Dictionary containing search results

    Raises:
        ClinicalTrialsAPIError: If there's an error with the API request
    """
    config = Config()

    try:
        # Rate limiting
        time.sleep(config.get_rate_limit_delay())

        # Parse fields string back to list
        fields_list = fields if fields else None

        # Build the API URL from config
        base_url = config.get_clinical_trials_base_url()
        params = {
            "format": "json",
            "query.term": search_expr,
            "pageSize": max_results,
            "fields": ",".join(fields_list) if fields_list else "",
        }

        # Add page token if provided
        if page_token and page_token != "None":
            params["pageToken"] = page_token

        # Make the API request
        response = requests.get(
            base_url, params=params, timeout=config.get_request_timeout()
        )
        response.raise_for_status()
        result = response.json()

        # Add pagination metadata to the result
        if isinstance(result, dict):
            studies_count = len(result.get("studies", []))
            logger.info(
                f"API returned {studies_count} studies for search '{search_expr}'"
            )
            result["pagination"] = {"max_results": max_results}

            # Add nextPageToken if it exists in the response
            if "nextPageToken" in result:
                result["nextPageToken"] = result["nextPageToken"]

        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")


def get_full_study(nct_id: str):
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
        response = requests.get(
            study_url, params=params, timeout=config.get_request_timeout()
        )
        response.raise_for_status()
        result = response.json()

        logger.info(f"API call completed for NCT ID: {nct_id}")

        # Check if we got a valid response
        if result is None:
            raise ValueError(f"No response received from API for NCT ID {nct_id}")

        if not isinstance(result, dict):
            raise ValueError(
                f"Invalid response format from API for NCT ID {nct_id}: expected dict, got {type(result)}"
            )

        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(
            f"API request failed for NCT ID {nct_id}: {str(e)}"
        )


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
            "countTotal": "true",  # Request total count
        }

        # Make the API request
        response = requests.get(
            base_url, params=params, timeout=config.get_request_timeout()
        )
        response.raise_for_status()
        result = response.json()

        # Extract the total count from the response
        total_studies = result.get("totalCount", 0)

        # Calculate pages
        total_pages = (
            (total_studies + page_size - 1) // page_size if total_studies > 0 else 0
        )

        result = {
            "total_studies": total_studies,
            "total_pages": total_pages,
            "page_size": page_size,
        }

        logger.info(
            f"Search '{search_expr}' statistics: {total_studies:,} studies, {total_pages:,} pages"
        )

        return result
    except Exception as e:
        raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")