"""
OncoCore Client - Unified interface for HeySol API client.

This module provides a unified interface for interacting with OncoCore
through the HeySol API client for oncology-specific memory storage and retrieval.
"""

# Import the HeySol client components
from heysol.clients import HeySolAPIClient, HeySolMCPClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError, ValidationError

# Re-export for convenience
__all__ = [
    "OncoCoreClient",
    "OncoCoreError",
    "CoreMemoryClient",
    "CoreMemoryError",
]
"""
HeySol API client implementation with both direct API and MCP support.

This client provides a unified interface that can use either direct API calls
or MCP (Model Context Protocol) operations based on availability and preference.
"""

from typing import Any, Dict, Optional

from heysol.clients import HeySolAPIClient, HeySolMCPClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError, ValidationError


class OncoCoreClient:
    """
    Unified client for interacting with HeySol services.

    This client provides both direct API access and MCP protocol support,
    automatically choosing the best method based on availability and operation type.

    Features:
    - Direct API operations for predictable, REST-based interactions
    - MCP operations for dynamic tool access and enhanced features
    - Automatic fallback between methods
    - Unified interface for both approaches
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[HeySolConfig] = None,
        skip_mcp_init: bool = False,
        prefer_mcp: bool = False,
    ):
        """
        Initialize the HeySol unified client.

        Args:
            api_key: HeySol API key (required for authentication)
            base_url: Base URL for the API (optional, uses config default if not provided)
            config: HeySolConfig object (optional, overrides individual parameters)
            skip_mcp_init: Skip MCP session initialization (useful for testing)
            prefer_mcp: Prefer MCP operations when both are available
        """
        # Use provided config or load from environment
        if config is None:
            config = HeySolConfig.from_env()

        # Use provided values or fall back to config
        if api_key is None:
            api_key = config.api_key
        if not base_url:
            base_url = config.base_url

        # Validate authentication
        if not api_key:
            raise ValidationError("API key is required")

        self.api_key = api_key
        self.base_url = base_url
        self.source = config.source
        self.timeout = config.timeout
        self.prefer_mcp = prefer_mcp

        # Initialize API client (always available)
        self.api_client = HeySolAPIClient(
            api_key=api_key,
            base_url=base_url,
            config=config
        )

        # Initialize MCP client (optional)
        self.mcp_client: Optional[HeySolMCPClient] = None
        if not skip_mcp_init:
            try:
                self.mcp_client = HeySolMCPClient(
                    api_key=api_key,
                    mcp_url=config.mcp_url,
                    config=config
                )
            except Exception:
                # MCP initialization failed, continue with API-only mode
                self.mcp_client = None

    @classmethod
    def from_env(cls, skip_mcp_init: bool = False, prefer_mcp: bool = False) -> "HeySolClient":
        """
        Create client from environment variables.

        Args:
            skip_mcp_init: Skip MCP session initialization (useful for testing)
            prefer_mcp: Prefer MCP operations when both are available
        """
        config = HeySolConfig.from_env()
        return cls(config=config, skip_mcp_init=skip_mcp_init, prefer_mcp=prefer_mcp)

    def is_mcp_available(self) -> bool:
        """Check if MCP is available and initialized."""
        return self.mcp_client is not None and self.mcp_client.is_mcp_available()

    def get_preferred_access_method(self, operation: str = "", mcp_tool_name: Optional[str] = None) -> str:
        """
        Determine the preferred access method for a given operation.

        Args:
            operation: The operation being performed (for context)
            mcp_tool_name: Specific MCP tool name to check

        Returns:
            "mcp" or "direct_api"
        """
        if self.prefer_mcp and self.is_mcp_available():
            if mcp_tool_name:
                return "mcp" if mcp_tool_name in self.mcp_client.tools else "direct_api"
            return "mcp"
        else:
            return "direct_api"

    def ensure_mcp_available(self, tool_name: Optional[str] = None) -> None:
        """Ensure MCP is available and optionally check for specific tool."""
        if not self.is_mcp_available():
            raise HeySolError("MCP is not available. Please check your MCP configuration.")

        if tool_name and tool_name not in self.mcp_client.tools:
            raise HeySolError(
                f"MCP tool '{tool_name}' is not available. Available tools: {list(self.mcp_client.tools.keys())}"
            )

    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available MCP tools."""
        if self.mcp_client:
            return self.mcp_client.get_available_tools()
        return {}

    def get_client_info(self) -> Dict[str, Any]:
        """Get information about both API and MCP clients."""
        return {
            "api_client": {
                "available": True,
                "base_url": self.api_client.base_url,
            },
            "mcp_client": {
                "available": self.is_mcp_available(),
                "session_id": self.mcp_client.session_id if self.mcp_client else None,
                "tools_count": len(self.mcp_client.tools) if self.mcp_client else 0,
                "mcp_url": self.mcp_client.mcp_url if self.mcp_client else None,
            },
            "preferred_method": "mcp" if self.prefer_mcp else "direct_api",
        }

    def ingest(
        self, message: str, source: Optional[str] = None, space_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingest data into CORE Memory."""
        method = self.get_preferred_access_method("ingest", "memory_ingest")
        if method == "mcp" and self.mcp_client:
            return self.mcp_client.ingest_via_mcp(message, source, space_id, session_id)
        else:
            return self.api_client.ingest(message, space_id, session_id)

    def search(
        self,
        query: str,
        space_ids: Optional[list] = None,
        limit: int = 10,
        include_invalidated: bool = False,
    ) -> Dict[str, Any]:
        """Search for memories in CORE Memory."""
        method = self.get_preferred_access_method("search", "memory_search")
        if method == "mcp" and self.mcp_client:
            return self.mcp_client.search_via_mcp(
                query=query,
                space_ids=space_ids,
                limit=limit
            )
        else:
            return self.api_client.search(query, space_ids, limit, include_invalidated)

    def get_spaces(self) -> list:
        """Get available memory spaces."""
        method = self.get_preferred_access_method("get_spaces", "memory_get_spaces")
        if method == "mcp" and self.mcp_client:
            return self.mcp_client.get_memory_spaces_via_mcp()
        else:
            return self.api_client.get_spaces()

    def create_space(self, name: str, description: str = "") -> str:
        """Create a new memory space."""
        # Space creation is typically API-only
        return self.api_client.create_space(name, description)

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the current user's profile."""
        method = self.get_preferred_access_method("get_user_profile", "get_user_profile")
        if method == "mcp" and self.mcp_client:
            return self.mcp_client.get_user_profile_via_mcp()
        else:
            return self.api_client.get_user_profile()

    # Memory endpoints
    def search_knowledge_graph(
        self, query: str, space_id: Optional[str] = None, limit: int = 10, depth: int = 2
    ) -> Dict[str, Any]:
        """Search the knowledge graph for related concepts and entities."""
        # Knowledge graph search is API-only for now
        return self.api_client.search_knowledge_graph(query, space_id, limit, depth)

    def add_data_to_ingestion_queue(
        self,
        data: Any,
        space_id: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add data to the ingestion queue for processing."""
        return self.api_client.add_data_to_ingestion_queue(data, space_id, priority, tags, metadata)

    def get_episode_facts(
        self, episode_id: str, limit: int = 100, offset: int = 0, include_metadata: bool = True
    ) -> list:
        """Get episode facts from CORE Memory."""
        return self.api_client.get_episode_facts(episode_id, limit, offset, include_metadata)

    def get_ingestion_logs(
        self,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list:
        """Get ingestion logs from CORE Memory."""
        return self.api_client.get_ingestion_logs(space_id, limit, offset, status, start_date, end_date)

    def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """Get a specific ingestion log by ID."""
        return self.api_client.get_specific_log(log_id)

    def check_ingestion_status(
        self, run_id: Optional[str] = None, space_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check the status of data ingestion processing."""
        return self.api_client.check_ingestion_status(run_id, space_id)

    # Spaces endpoints
    def bulk_space_operations(
        self,
        intent: str,
        space_id: Optional[str] = None,
        statement_ids: Optional[list] = None,
        space_ids: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Perform bulk operations on spaces."""
        return self.api_client.bulk_space_operations(intent, space_id, statement_ids, space_ids)

    def get_space_details(
        self, space_id: str, include_stats: bool = True, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get detailed information about a specific space."""
        return self.api_client.get_space_details(space_id, include_stats, include_metadata)

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update properties of an existing space."""
        return self.api_client.update_space(space_id, name, description, metadata)

    def delete_space(self, space_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a space."""
        return self.api_client.delete_space(space_id, confirm)

    # Webhook endpoints
    def register_webhook(
        self, url: str, events: Optional[list] = None, secret: str = ""
    ) -> Dict[str, Any]:
        """Register a new webhook."""
        return self.api_client.register_webhook(url, events, secret)

    def list_webhooks(
        self,
        space_id: Optional[str] = None,
        active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """List all webhooks."""
        return self.api_client.list_webhooks(space_id, active, limit, offset)

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook details."""
        return self.api_client.get_webhook(webhook_id)

    def update_webhook(
        self, webhook_id: str, url: str, events: list, secret: str = "", active: bool = True
    ) -> Dict[str, Any]:
        """Update webhook properties."""
        return self.api_client.update_webhook(webhook_id, url, events, secret, active)

    def delete_webhook(self, webhook_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a webhook."""
        return self.api_client.delete_webhook(webhook_id, confirm)

    def delete_log_entry(self, log_id: str) -> Dict[str, Any]:
        """Delete a log entry from CORE Memory."""
        return self.api_client.delete_log_entry(log_id)

    # MCP-specific operations
    def delete_logs_by_source(
        self, source: str, space_id: Optional[str] = None, confirm: bool = False
    ) -> Dict[str, Any]:
        """Delete all logs with a specific source using MCP."""
        if not self.mcp_client:
            raise HeySolError("MCP client not available")
        return self.mcp_client.delete_logs_by_source(source, space_id, confirm)

    def get_logs_by_source(
        self, source: str, space_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get all logs with a specific source."""
        if self.mcp_client:
            return self.mcp_client.get_logs_by_source(source, space_id, limit)
        else:
            return {
                "logs": [],
                "total_count": 0,
                "source": source,
                "space_id": space_id,
                "note": "MCP client not available for log retrieval by source",
            }

    # Direct access to sub-clients for advanced usage
    @property
    def api(self) -> HeySolAPIClient:
        """Direct access to the API client for advanced operations."""
        return self.api_client

    @property
    def mcp(self) -> Optional[HeySolMCPClient]:
        """Direct access to the MCP client for advanced operations."""
        return self.mcp_client

    def close(self) -> None:
        """Close the client and clean up resources."""
        if self.api_client:
            self.api_client.close()
        if self.mcp_client:
            self.mcp_client.close()


# Aliases for backward compatibility
CoreMemoryClient = OncoCoreClient
CoreMemoryError = HeySolError
