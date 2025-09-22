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


@dataclass
class HeySolConfig:
    """
    Configuration class for HeySol API client.
    """

    api_key: Optional[str] = None
    base_url: str = "https://core.heysol.ai/api/v1"
    source: str = "heysol-python-client"
    mcp_url: str = "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"


    @classmethod
    def from_env(cls) -> "HeySolConfig":
        """
        Create configuration from environment variables.
        """
        return cls(
            api_key=os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY"),
            base_url=os.getenv("HEYSOL_BASE_URL") or os.getenv("COREAI_BASE_URL", "https://core.heysol.ai/api/v1"),
            source=os.getenv("HEYSOL_SOURCE") or os.getenv("COREAI_SOURCE", "heysol-python-client"),
            mcp_url=os.getenv("HEYSOL_MCP_URL") or os.getenv("COREAI_MCP_URL", "https://core.heysol.ai/api/v1/mcp?source=Kilo-Code"),
        )