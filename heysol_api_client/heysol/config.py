"""
Configuration management for the HeySol API client.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class HeySolConfig:
    """
    Configuration class for HeySol API client.
    """

    api_key: Optional[str] = None
    base_url: str = "https://core.heysol.ai/api/v1/mcp"
    source: str = "heysol-python-client"

    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_redirect_uri: Optional[str] = None
    oauth2_scope: str = "openid profile email api"

    @classmethod
    def from_env(cls) -> "HeySolConfig":
        """
        Create configuration from environment variables.
        """
        return cls(
            api_key=os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY"),
            base_url=os.getenv("COREAI_BASE_URL", "https://core.heysol.ai/api/v1/mcp"),
            source=os.getenv("COREAI_SOURCE", "heysol-python-client"),
            oauth2_client_id=os.getenv("COREAI_OAUTH2_CLIENT_ID"),
            oauth2_client_secret=os.getenv("COREAI_OAUTH2_CLIENT_SECRET"),
            oauth2_redirect_uri=os.getenv("COREAI_OAUTH2_REDIRECT_URI"),
            oauth2_scope=os.getenv("COREAI_OAUTH2_SCOPE", "openid profile email api"),
        )