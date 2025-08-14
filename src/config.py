import os
from typing import Optional


class Config:
    """
    Configuration class for the mCODE Translator
    """
    
    def __init__(self):
        # API Configuration
        self.api_base_url = os.getenv("CLINICALTRIALS_API_BASE_URL", "https://clinicaltrials.gov/api/v2")
        self.api_key = os.getenv("CLINICALTRIALS_API_KEY", None)
        
        # Cache Configuration
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        
        # Rate Limiting
        self.rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
        
        # Request Configuration
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Create cache directory if it doesn't exist
        if self.cache_enabled and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_api_key(self) -> Optional[str]:
        """
        Get the API key for clinicaltrials.gov
        
        Returns:
            API key or None if not set
        """
        return self.api_key
    
    def is_cache_enabled(self) -> bool:
        """
        Check if caching is enabled
        
        Returns:
            True if caching is enabled, False otherwise
        """
        return self.cache_enabled