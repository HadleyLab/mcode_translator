import requests
import time
import json
import os
import hashlib
from typing import Dict, List, Optional
from .config import Config
from .cache import CacheManager


class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrialsAPI errors"""
    pass


class ClinicalTrialsAPI:
    """
    Client for interacting with clinicaltrials.gov API
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the ClinicalTrialsAPI client
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config or Config()
        self.base_url = self.config.api_base_url
        self.api_key = self.config.get_api_key()
        self.session = requests.Session()
        self.rate_limit_delay = self.config.rate_limit_delay
        self.cache_manager = CacheManager(self.config)
        
        # Set headers for all requests
        self.session.headers.update({
            "User-Agent": "mCODE Translator/1.0"
        })
    
    def search_trials(self, search_expr: str, fields: Optional[List[str]] = None,
                      max_results: int = 100) -> Dict:
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
        # Create cache key
        cache_key_data = f"search:{search_expr}:{','.join(fields) if fields else 'all'}:{max_results}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
        
        # Try to get from cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Build the URL for the studies endpoint (v2 API)
        url = f"{self.base_url}/studies"
        
        # Prepare query parameters
        params = {
            "query.term": search_expr,
            "pageSize": max_results,
            "format": "json"
        }
        
        # Add fields parameter if specified
        if fields:
            params["fields"] = ",".join(fields)
        
        # Add API key if provided
        if self.api_key:
            params["apiKey"] = self.api_key
            
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Make the request
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Cache the result
            self.cache_manager.set(cache_key, result)
            
            return result
        except requests.exceptions.RequestException as e:
            raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ClinicalTrialsAPIError(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            raise ClinicalTrialsAPIError(f"Unexpected error: {str(e)}")
    
    def get_full_study(self, nct_id: str) -> Dict:
        """
        Get complete study record for a specific trial
        
        Args:
            nct_id: NCT ID of the clinical trial (e.g., "NCT00000000")
            
        Returns:
            Dictionary containing the full study record
            
        Raises:
            ClinicalTrialsAPIError: If there's an error with the API request
        """
        # Create cache key
        cache_key = f"full_study:{nct_id}"
        
        # Try to get from cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Build the URL for the studies endpoint with specific NCT ID (v2 API)
        url = f"{self.base_url}/studies/{nct_id}"
        
        # Prepare query parameters
        params = {
            "format": "json"
        }
        
        # Add API key if provided
        if self.api_key:
            params["apiKey"] = self.api_key
            
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            # Make the request
            response = self.session.get(url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # Cache the result
            self.cache_manager.set(cache_key, result)
            
            return result
        except requests.exceptions.RequestException as e:
            raise ClinicalTrialsAPIError(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ClinicalTrialsAPIError(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            raise ClinicalTrialsAPIError(f"Unexpected error: {str(e)}")


class ClinicalTrial:
    """
    Data model for a clinical trial
    """
    
    def __init__(self, nct_id: str, title: str, eligibility_criteria: str):
        self.nct_id = nct_id
        self.title = title
        self.eligibility_criteria = eligibility_criteria
        self.conditions = []
        self.interventions = []
        self.gender = None
        self.min_age = None
        self.max_age = None
        self.healthy_volunteers = None


class EligibilityCriteria:
    """
    Data model for eligibility criteria
    """
    
    def __init__(self, text: str):
        self.text = text
        self.inclusion_criteria = []
        self.exclusion_criteria = []
        self.structured_elements = {}