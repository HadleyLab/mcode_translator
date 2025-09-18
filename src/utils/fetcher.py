"""
Simple Clinical Trials Fetcher Utility
Provides basic functions for fetching clinical trial data from ClinicalTrials.gov API.
"""

import json
import time
from typing import Any, Dict, List, Optional

import requests

from src.utils.api_manager import APIManager
from src.utils.concurrency import TaskQueue, create_task
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
        }

        # Only add fields parameter if fields are specified
        if fields_list:
            params["fields"] = ",".join(fields_list)

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


def get_full_studies_batch(nct_ids: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """
    Fetch multiple clinical trial studies concurrently using batch processing.

    Args:
        nct_ids: List of NCT IDs to fetch
        max_workers: Maximum number of concurrent workers

    Returns:
        Dictionary mapping NCT IDs to their study data or error messages

    Raises:
        ClinicalTrialsAPIError: If there's a critical error with batch processing
    """
    if not nct_ids:
        return {}

    logger.info(f"🔄 Batch fetching {len(nct_ids)} studies with {max_workers} concurrent workers")

    # Create tasks for concurrent processing
    tasks = []
    for nct_id in nct_ids:
        task = create_task(
            task_id=f"fetch_{nct_id}",
            func=_fetch_single_study_with_error_handling,
            nct_id=nct_id
        )
        tasks.append(task)

    # Execute tasks concurrently
    task_queue = TaskQueue(max_workers=max_workers, name="ClinicalTrialsFetcher")
    task_results = task_queue.execute_tasks(tasks)

    # Process results
    results = {}
    successful = 0
    failed = 0

    for result in task_results:
        nct_id = result.task_id.replace("fetch_", "")

        if result.success:
            results[nct_id] = result.result
            successful += 1
        else:
            logger.warning(f"Failed to fetch {nct_id}: {result.error}")
            results[nct_id] = {"error": str(result.error)}
            failed += 1

    logger.info(f"📊 Batch fetch complete: {successful} successful, {failed} failed")
    return results


def _fetch_single_study_with_error_handling(nct_id: str) -> Dict[str, Any]:
    """
    Fetch a single study with proper error handling for concurrent processing.

    Args:
        nct_id: NCT ID to fetch

    Returns:
        Study data dictionary

    Raises:
        Exception: If the fetch fails
    """
    try:
        return get_full_study(nct_id)
    except Exception as e:
        logger.debug(f"Error fetching {nct_id}: {e}")
        raise


def search_trials_parallel(
    search_expr: str,
    fields=None,
    max_results: int = 1000,
    page_size: int = 100,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Search for clinical trials with parallel pagination for large result sets.

    Args:
        search_expr: Search expression (e.g., "breast cancer")
        fields: List of fields to retrieve
        max_results: Maximum total results to return
        page_size: Results per page (API limit is typically 100)
        max_workers: Maximum concurrent workers for pagination

    Returns:
        Dictionary containing all search results with pagination metadata
    """
    config = Config()

    # First, get total count to determine pagination needs
    try:
        total_info = calculate_total_studies(search_expr, fields, page_size)
        total_studies = total_info["total_studies"]

        if total_studies == 0:
            return {"studies": [], "totalCount": 0, "pagination": {"total_pages": 0}}

        # Limit max_results to total available
        actual_max_results = min(max_results, total_studies)
        total_pages = (actual_max_results + page_size - 1) // page_size

        logger.info(f"🔄 Parallel search: {total_studies} total studies, fetching {actual_max_results} with {total_pages} pages")

        # Create tasks for concurrent pagination
        tasks = []
        for page in range(total_pages):
            start_index = page * page_size
            current_page_size = min(page_size, actual_max_results - start_index)

            task = create_task(
                task_id=f"page_{page}",
                func=_search_single_page,
                search_expr=search_expr,
                fields=fields,
                max_results=current_page_size,
                page_token=None  # We'll handle pagination differently
            )
            tasks.append(task)

        # Execute pagination tasks concurrently
        task_queue = TaskQueue(max_workers=max_workers, name="ClinicalTrialsPagination")
        task_results = task_queue.execute_tasks(tasks)

        # Aggregate results
        all_studies = []
        successful_pages = 0
        failed_pages = 0

        for result in task_results:
            if result.success:
                page_data = result.result
                studies = page_data.get("studies", [])
                all_studies.extend(studies)
                successful_pages += 1
            else:
                logger.warning(f"Page fetch failed: {result.error}")
                failed_pages += 1

        # Create final result
        result = {
            "studies": all_studies,
            "totalCount": len(all_studies),
            "pagination": {
                "total_pages": total_pages,
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "page_size": page_size,
                "max_results_requested": max_results,
                "actual_results": len(all_studies)
            }
        }

        logger.info(f"📊 Parallel search complete: {len(all_studies)} studies from {successful_pages}/{total_pages} pages")
        return result

    except Exception as e:
        logger.error(f"Parallel search failed: {e}")
        raise ClinicalTrialsAPIError(f"Parallel search failed: {str(e)}")


def _search_single_page(
    search_expr: str,
    fields=None,
    max_results: int = 100,
    page_token: str = None
) -> Dict[str, Any]:
    """
    Search a single page of results (helper for parallel processing).

    Args:
        search_expr: Search expression
        fields: Fields to retrieve
        max_results: Results per page
        page_token: Pagination token

    Returns:
        Page results
    """
    return search_trials(search_expr, fields, max_results, page_token)


def search_multiple_queries(
    search_queries: List[str],
    fields=None,
    max_results_per_query: int = 100,
    max_workers: int = 4
) -> Dict[str, Dict[str, Any]]:
    """
    Execute multiple search queries concurrently.

    Args:
        search_queries: List of search expressions to execute
        fields: Fields to retrieve for each query
        max_results_per_query: Maximum results per individual query
        max_workers: Maximum concurrent workers

    Returns:
        Dictionary mapping search queries to their results
    """
    if not search_queries:
        return {}

    logger.info(f"🔄 Executing {len(search_queries)} concurrent search queries")

    # Create tasks for concurrent queries
    tasks = []
    for query in search_queries:
        task = create_task(
            task_id=f"query_{hash(query) % 10000}",
            func=search_trials,
            search_expr=query,
            fields=fields,
            max_results=max_results_per_query
        )
        tasks.append(task)

    # Execute query tasks concurrently
    task_queue = TaskQueue(max_workers=max_workers, name="ClinicalTrialsMultiQuery")
    task_results = task_queue.execute_tasks(tasks)

    # Process results
    results = {}
    successful = 0
    failed = 0

    for i, result in enumerate(task_results):
        query = search_queries[i]

        if result.success:
            results[query] = result.result
            successful += 1
        else:
            logger.warning(f"Query failed '{query}': {result.error}")
            results[query] = {"error": str(result.error)}
            failed += 1

    logger.info(f"📊 Multi-query search complete: {successful} successful, {failed} failed")
    return results


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