"""
Minimal HeySol configuration for mCODE Translator CLI.
"""

import os
from typing import Optional


class HeySolConfig:
    """Minimal HeySol configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://core.heysol.ai/api/v1",
        mcp_url: str = "https://core.heysol.ai/api/v1/mcp",
        source: str = "mcode-translator",
        timeout: int = 60,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.mcp_url = mcp_url
        self.source = source
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "HeySolConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv("HEYSOL_API_KEY"),
            base_url=os.getenv("HEYSOL_BASE_URL", "https://core.heysol.ai/api/v1"),
            mcp_url=os.getenv("HEYSOL_MCP_URL", "https://core.heysol.ai/api/v1/mcp"),
            source=os.getenv("HEYSOL_SOURCE", "mcode-translator"),
            timeout=int(os.getenv("HEYSOL_TIMEOUT", "60")),
        )
