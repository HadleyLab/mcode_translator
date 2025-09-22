"""
Configuration management for the HeySol API client.
"""

import os
from typing import Optional
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, will rely on environment variables only


# Default configuration constants
DEFAULT_BASE_URL = "https://core.heysol.ai/api/v1"
DEFAULT_MCP_URL = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"
DEFAULT_PROFILE_URL = "https://core.heysol.ai/api/profile"
DEFAULT_SOURCE = "heysol-python-client"


@dataclass
class HeySolConfig:
    """
    Configuration class for HeySol API client.
    """

    api_key: Optional[str] = None
    base_url: str = DEFAULT_BASE_URL
    source: str = DEFAULT_SOURCE
    mcp_url: str = DEFAULT_MCP_URL
    profile_url: str = DEFAULT_PROFILE_URL


    @classmethod
    def from_env(cls) -> "HeySolConfig":
        """
        Create configuration from environment variables.
        """
        return cls(
            api_key=os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY"),
            base_url=os.getenv("HEYSOL_BASE_URL") or os.getenv("COREAI_BASE_URL") or DEFAULT_BASE_URL,
            source=os.getenv("HEYSOL_SOURCE") or os.getenv("COREAI_SOURCE") or DEFAULT_SOURCE,
            mcp_url=os.getenv("HEYSOL_MCP_URL") or os.getenv("COREAI_MCP_URL") or DEFAULT_MCP_URL,
            profile_url=os.getenv("HEYSOL_PROFILE_URL") or os.getenv("COREAI_PROFILE_URL") or DEFAULT_PROFILE_URL,
        )