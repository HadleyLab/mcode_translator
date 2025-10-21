"""
OncoCore Client - Unified interface for HeySol API client.

This module provides a unified interface for interacting with OncoCore
through the HeySol API client for oncology-specific memory storage and retrieval.
"""

from typing import Any, Dict, List, Optional

# Import the HeySol client components
from heysol.clients import HeySolAPIClient, HeySolMCPClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError, ValidationError
from shared.models import SearchResult

# Re-export for convenience
__all__ = [
    "OncoCoreClient",
    "HeySolError",
]


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
        self.api_client = HeySolAPIClient(api_key=api_key, base_url=base_url, config=config)

        # Initialize MCP client (optional)
        self.mcp_client: Optional[HeySolMCPClient] = None
        if not skip_mcp_init:
            self.mcp_client = HeySolMCPClient(
                api_key=api_key, mcp_url=config.mcp_url, config=config
            )

    @classmethod
    def from_env(cls, skip_mcp_init: bool = False, prefer_mcp: bool = False) -> "OncoCoreClient":
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

    def get_preferred_access_method(
        self, operation: str = "", mcp_tool_name: Optional[str] = None
    ) -> str:
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
                if self.mcp_client and mcp_tool_name in self.mcp_client.tools:
                    return "mcp"
                return "direct_api"
            return "mcp"
        else:
            return "direct_api"

    def ensure_mcp_available(self, tool_name: Optional[str] = None) -> None:
        """Ensure MCP is available and optionally check for specific tool."""
        if not self.is_mcp_available():
            raise HeySolError("MCP is not available. Please check your MCP configuration.")

        if self.mcp_client and tool_name and tool_name not in self.mcp_client.tools:
            raise HeySolError(
                f"MCP tool '{tool_name}' is not available. Available tools: {list(self.mcp_client.tools.keys())}"
            )

    def get_available_tools(self) -> Dict[str, Any]:
        """Get information about available MCP tools."""
        if self.mcp_client:
            tools = self.mcp_client.get_available_tools()
            return tools if isinstance(tools, dict) else {}
        return {}

    def get_tool_names(self, refresh: bool = False) -> List[str]:
        """Return the list of available MCP tools."""
        if not self.mcp_client:
            return []
        if refresh:
            self.mcp_client.refresh_tools()
        tool_names = self.mcp_client.get_tool_names()
        return tool_names if isinstance(tool_names, list) else []

    def refresh_mcp_tools(self) -> List[str]:
        """Refresh the cached MCP tool metadata and return the updated names."""
        if not self.mcp_client:
            return []
        self.mcp_client.refresh_tools()
        tool_names = self.mcp_client.get_tool_names()
        return tool_names if isinstance(tool_names, list) else []

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Invoke an MCP tool by name."""
        if not self.mcp_client:
            raise HeySolError("MCP client not available")
        self.ensure_mcp_available(tool_name)
        result = self.mcp_client.call_tool(tool_name, **kwargs)
        return result if isinstance(result, dict) else {}

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
        self,
        message: str,
        source: Optional[str] = None,
        space_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest data into CORE Memory."""
        method = self.get_preferred_access_method("ingest", "memory_ingest")
        if method == "mcp" and self.mcp_client:
            result = self.mcp_client.ingest_via_mcp(message, source, space_id, session_id)
            return result if isinstance(result, dict) else {}
        else:
            result = self.api_client.ingest(message, space_id, session_id)
            return result if isinstance(result, dict) else {}

    def search(
        self,
        query: str,
        space_ids: Optional[List[str]] = None,
        limit: int = 10,
        include_invalidated: bool = False,
    ) -> SearchResult:
        """Search for memories in CORE Memory."""
        method = self.get_preferred_access_method("search", "memory_search")
        if method == "mcp" and self.mcp_client:
            result_data = self.mcp_client.search_via_mcp(
                query=query, space_ids=space_ids, limit=limit
            )
        else:
            result_data = self.api_client.search(
                query, space_ids, limit, include_invalidated
            )

        if isinstance(result_data, dict):
            return SearchResult(**result_data)
        # Assuming the other possible type is SearchResult, as per the underlying client
        return result_data

    def get_spaces(self) -> List[Dict[str, Any]]:
        """Get available memory spaces."""
        method = self.get_preferred_access_method("get_spaces", "memory_get_spaces")
        if method == "mcp" and self.mcp_client:
            spaces = self.mcp_client.get_memory_spaces_via_mcp()
            return spaces if isinstance(spaces, list) else []
        else:
            spaces = self.api_client.get_spaces()
            return spaces if isinstance(spaces, list) else []

    def create_space(self, name: str, description: str = "") -> str:
        """Create a new memory space."""
        # Space creation is typically API-only
        space_id = self.api_client.create_space(name, description)
        return space_id if isinstance(space_id, str) else ""

    def get_user_profile(self) -> Dict[str, Any]:
        """Get the current user's profile."""
        method = self.get_preferred_access_method("get_user_profile", "get_user_profile")
        if method == "mcp" and self.mcp_client:
            profile = self.mcp_client.get_user_profile_via_mcp()
            return profile if isinstance(profile, dict) else {}
        else:
            profile = self.api_client.get_user_profile()
            return profile if isinstance(profile, dict) else {}

    # Memory endpoints
    def search_knowledge_graph(
        self,
        query: str,
        space_id: Optional[str] = None,
        limit: int = 10,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Search the knowledge graph for related concepts and entities."""
        # Knowledge graph search is API-only for now
        graph = self.api_client.search_knowledge_graph(query, space_id, limit, depth)
        return graph if isinstance(graph, dict) else {}

    def add_data_to_ingestion_queue(
        self,
        data: Any,
        space_id: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add data to the ingestion queue for processing."""
        response = self.api_client.add_data_to_ingestion_queue(
            data, space_id, priority, tags, metadata
        )
        return response if isinstance(response, dict) else {}

    def get_episode_facts(
        self,
        episode_id: str,
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get episode facts from CORE Memory."""
        facts = self.api_client.get_episode_facts(episode_id, limit, offset, include_metadata)
        facts = self.api_client.get_episode_facts(
            episode_id, limit, offset, include_metadata
        )
        return facts if isinstance(facts, list) else []

    def get_ingestion_logs(
        self,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get ingestion logs from CORE Memory."""
        logs = self.api_client.get_ingestion_logs(
            space_id, limit, offset, status, start_date, end_date
        )
        return logs if isinstance(logs, list) else []

    def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """Get a specific ingestion log by ID."""
        log = self.api_client.get_specific_log(log_id)
        return log if isinstance(log, dict) else {}

    def check_ingestion_status(
        self, run_id: Optional[str] = None, space_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check the status of data ingestion processing."""
        status = self.api_client.check_ingestion_status(run_id, space_id)
        return status if isinstance(status, dict) else {}

    # Spaces endpoints
    def bulk_space_operations(
        self,
        intent: str,
        space_id: Optional[str] = None,
        statement_ids: Optional[List[str]] = None,
        space_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform bulk operations on spaces."""
        response = self.api_client.bulk_space_operations(intent, space_id, statement_ids, space_ids)
        return response if isinstance(response, dict) else {}

    def get_space_details(
        self, space_id: str, include_stats: bool = True, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get detailed information about a specific space."""
        details = self.api_client.get_space_details(space_id, include_stats, include_metadata)
        return details if isinstance(details, dict) else {}

    def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update properties of an existing space."""
        response = self.api_client.update_space(space_id, name, description, metadata)
        return response if isinstance(response, dict) else {}

    def delete_space(self, space_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a space."""
        response = self.api_client.delete_space(space_id, confirm)
        return response if isinstance(response, dict) else {}

    # Webhook endpoints
    def register_webhook(
        self, url: str, events: Optional[List[str]] = None, secret: str = ""
    ) -> Dict[str, Any]:
        """Register a new webhook."""
        webhook = self.api_client.register_webhook(url, events, secret)
        return webhook if isinstance(webhook, dict) else {}

    def list_webhooks(
        self,
        space_id: Optional[str] = None,
        active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all webhooks."""
        webhooks = self.api_client.list_webhooks(space_id, active, limit, offset)
        return webhooks if isinstance(webhooks, list) else []

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook details."""
        webhook = self.api_client.get_webhook(webhook_id)
        return webhook if isinstance(webhook, dict) else {}

    def update_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: str = "",
        active: bool = True,
    ) -> Dict[str, Any]:
        """Update webhook properties."""
        webhook = self.api_client.update_webhook(webhook_id, url, events, secret, active)
        return webhook if isinstance(webhook, dict) else {}

    def delete_webhook(self, webhook_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a webhook."""
        response = self.api_client.delete_webhook(webhook_id, confirm)
        return response if isinstance(response, dict) else {}

    def delete_log_entry(self, log_id: str) -> Dict[str, Any]:
        """Delete a log entry from CORE Memory."""
        response = self.api_client.delete_log_entry(log_id)
        return response if isinstance(response, dict) else {}

    # MCP-specific operations
    def delete_logs_by_source(
        self, source: str, space_id: Optional[str] = None, confirm: bool = False
    ) -> Dict[str, Any]:
        """Delete all logs with a specific source using MCP."""
        if not self.mcp_client:
            raise HeySolError("MCP client not available")
        result = self.mcp_client.delete_logs_by_source(source, space_id, confirm)
        return result if isinstance(result, dict) else {}

    def get_logs_by_source(
        self, source: str, space_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get all logs with a specific source."""
        if self.mcp_client:
            logs = self.mcp_client.get_logs_by_source(source, space_id, limit)
            return logs if isinstance(logs, dict) else {}
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
