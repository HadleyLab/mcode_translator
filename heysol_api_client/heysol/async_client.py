"""
Async HeySol API client implementation.

This module provides an async version of the HeySol API client using aiohttp
for concurrent operations and improved performance.
"""

import asyncio
import json
import uuid
import time
import logging
from typing import Any, Dict, Optional, Union, List
from urllib.parse import urljoin

import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientSession

from .config import HeySolConfig
from .exceptions import (
    HeySolError,
    AuthenticationError,
    RateLimitError,
    APIError,
    ValidationError,
    ConnectionError,
    ServerError,
    NotFoundError,
)


class AsyncHeySolClient:
    """
    Async client for interacting with the HeySol API.

    This client provides an async interface to the HeySol API with support
    for concurrent operations, MCP protocol, authentication, and robust error handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[HeySolConfig] = None,
        base_url: Optional[str] = None,
        source: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the async HeySol API client.

        Args:
            api_key: HeySol API key (optional, can be loaded from config)
            config: HeySolConfig instance (optional)
            base_url: Base URL for the API (optional, can be loaded from config)
            source: Source identifier for requests (optional, can be loaded from config)
            timeout: Request timeout in seconds (optional, can be loaded from config)
            max_retries: Maximum number of retries (optional, can be loaded from config)

        Raises:
            AuthenticationError: If no API key is provided and cannot be loaded from config
        """
        # Load configuration
        if config is None:
            config = HeySolConfig.from_env()

        # Override config values with provided parameters
        if api_key is not None:
            config.api_key = api_key
        if base_url is not None:
            config.base_url = base_url
        if source is not None:
            config.source = source
        if timeout is not None:
            config.timeout = timeout
        if max_retries is not None:
            config.max_retries = max_retries

        # Validate required configuration
        if not config.api_key:
            raise AuthenticationError(
                "API key is required. Provide it as a parameter or set COREAI_API_KEY environment variable."
            )

        self.config = config
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Any] = {}
        self._rate_limit_remaining = config.rate_limit_per_minute
        self._rate_limit_reset_time = time.time() + 60  # Reset every minute
        self._request_count = 0
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Setup logging
        self._setup_logging()

        # Setup aiohttp session
        self._session: Optional[ClientSession] = None

        # Initialize MCP session
        self._initialize_session_sync()

        self.logger.info("Async HeySol API client initialized successfully")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)

        # Set log level
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.config.log_to_file and self.config.log_file_path:
            file_handler = logging.FileHandler(self.config.log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    async def _ensure_session(self) -> ClientSession:
        """Ensure aiohttp session is created and available."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_requests,
                limit_per_host=self.config.max_concurrent_requests // 2,
            )

            self._session = ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self._headers()
            )

        return self._session

    def _headers(self) -> Dict[str, str]:
        """Create headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream, */*",
            "User-Agent": f"heysol-python-client/{self.config.source}",
        }

        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        return headers

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        if not self.config.rate_limit_enabled:
            return

        current_time = time.time()

        # Reset rate limit counter if a minute has passed
        if current_time >= self._rate_limit_reset_time:
            self._rate_limit_remaining = self.config.rate_limit_per_minute
            self._rate_limit_reset_time = current_time + 60

        if self._rate_limit_remaining <= 0:
            reset_in = int(self._rate_limit_reset_time - current_time)
            raise RateLimitError(
                f"Rate limit exceeded. Resets in {reset_in} seconds.",
                retry_after=reset_in
            )

        self._rate_limit_remaining -= 1

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> aiohttp.ClientResponse:
        """
        Make an async HTTP request with error handling and retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (relative to base URL)
            data: Request body data
            params: Query parameters
            stream: Whether to stream the response

        Returns:
            aiohttp.ClientResponse: The HTTP response object

        Raises:
            HeySolError: For various API and network errors
        """
        url = urljoin(self.config.base_url, endpoint.lstrip("/"))

        # Check rate limit
        self._check_rate_limit()

        session = await self._ensure_session()

        try:
            self.logger.debug(f"Making async {method} request to {url}")

            async with self._semaphore:  # Limit concurrent requests
                response = await session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=self._headers(),
                )

                # Handle different HTTP status codes
                if response.status == 401:
                    raise AuthenticationError("Invalid API key or authentication failed")
                elif response.status == 404:
                    raise NotFoundError("Requested resource not found")
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    retry_after = int(retry_after) if retry_after else None
                    raise RateLimitError("Rate limit exceeded", retry_after=retry_after)
                elif response.status >= 500:
                    raise ServerError(f"Server error: {response.status}")
                elif response.status >= 400:
                    raise APIError(f"Client error: {response.status}")

                return response

        except asyncio.TimeoutError:
            raise ConnectionError("Request timeout")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Client error: {e}")

    async def _parse_json_response(self, response: aiohttp.ClientResponse) -> Any:
        """
        Parse JSON response from the API.

        Args:
            response: HTTP response object

        Returns:
            Parsed JSON data

        Raises:
            APIError: If response cannot be parsed as JSON
        """
        try:
            return await response.json()
        except json.JSONDecodeError as e:
            raise APIError(f"Failed to parse JSON response: {e}")

    async def _parse_mcp_response(self, response: aiohttp.ClientResponse) -> Any:
        """
        Parse MCP protocol response.

        Args:
            response: HTTP response object

        Returns:
            Parsed MCP response data

        Raises:
            HeySolError: If MCP response cannot be parsed or contains errors
        """
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()

        if content_type == "application/json":
            return await self._parse_json_response(response)
        elif content_type == "text/event-stream":
            # Handle Server-Sent Events (SSE) - simplified for async
            text = await response.text()
            if text.startswith("data:"):
                try:
                    return json.loads(text[5:].strip())
                except json.JSONDecodeError:
                    raise APIError("Failed to parse SSE data")
            else:
                raise APIError("No data found in SSE stream")
        else:
            raise APIError(f"Unexpected Content-Type: {content_type}")

    def _initialize_session_sync(self) -> None:
        """Initialize MCP session synchronously (called from __init__)."""
        # This is a simplified sync version for initialization
        # The full async initialization would be too complex for __init__

        try:
            # Create a temporary sync session for initialization
            import requests
            session = requests.Session()
            session.timeout = self.config.timeout

            headers = self._headers()

            # Initialize session
            init_payload = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {"tools": True},
                    "clientInfo": {
                        "name": "heysol-python-client",
                        "version": "1.0.0"
                    },
                },
            }

            response = session.post(
                self.config.base_url,
                json=init_payload,
                headers=headers,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                self.session_id = response.headers.get("Mcp-Session-Id")
                self.logger.info("MCP session initialized (sync)")

            session.close()

        except Exception as e:
            self.logger.warning(f"Failed to initialize MCP session: {e}")

    async def _call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Any:
        """
        Call an MCP tool asynchronously.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            stream: Whether to stream the response

        Returns:
            Tool call result

        Raises:
            APIError: If tool is not available or call fails
        """
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            },
        }

        response = await self._make_request("POST", "", data=payload, stream=stream)
        return await self._parse_mcp_response(response)

    async def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get available memory spaces asynchronously.

        Returns:
            List of available spaces

        Raises:
            HeySolError: If the request fails
        """
        try:
            return await self._call_tool("memory_get_spaces")
        except APIError as e:
            raise HeySolError(f"Failed to get spaces: {e}")

    async def create_space(self, name: str, description: str = "") -> str:
        """
        Create a new memory space asynchronously.

        Args:
            name: Name of the space to create
            description: Optional description for the space

        Returns:
            Space ID of the created space

        Raises:
            HeySolError: If space creation fails
        """
        if not name:
            raise ValidationError("Space name is required")

        # Use direct API call for space creation
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces"

        payload = {"name": name, "description": description}

        try:
            response = await self._make_request("POST", url, data=payload)
            data = await self._parse_json_response(response)

            space_id = data.get("space", {}).get("id") or data.get("id")
            if not space_id:
                raise APIError(f"Unexpected create_space response: {data}")

            self.logger.info(f"Created space '{name}' with ID: {space_id}")
            return space_id

        except Exception as e:
            raise HeySolError(f"Failed to create space '{name}': {e}")

    async def bulk_space_operations(
        self,
        operations: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform bulk operations on spaces asynchronously.

        Args:
            operations: List of space operations to perform
            batch_size: Number of operations to process in each batch

        Returns:
            List of operation results

        Raises:
            HeySolError: If bulk operations fail
        """
        # Use direct API call for bulk space operations
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/bulk"

        payload = {
            "operations": operations,
            "batch_size": batch_size,
        }

        try:
            response = await self._make_request("POST", url, data=payload)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to perform bulk space operations: {e}")

    async def get_space_details(
        self,
        space_id: str,
        include_stats: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific space asynchronously.

        Args:
            space_id: ID of the space to get details for
            include_stats: Whether to include usage statistics
            include_metadata: Whether to include metadata

        Returns:
            Dictionary with detailed space information

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for space details
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}/details"

        params = {
            "include_stats": include_stats,
            "include_metadata": include_metadata,
        }

        try:
            response = await self._make_request("GET", url, params=params)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to get space details: {e}")

    async def update_space(
        self,
        space_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_public: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Update properties of an existing space asynchronously.

        Args:
            space_id: ID of the space to update
            name: Optional new name for the space
            description: Optional new description
            metadata: Optional new metadata
            is_public: Optional new public status

        Returns:
            Dictionary with updated space information

        Raises:
            HeySolError: If the update fails
        """
        # Use direct API call for space updates
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        payload = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if metadata is not None:
            payload["metadata"] = metadata
        if is_public is not None:
            payload["is_public"] = is_public

        try:
            response = await self._make_request("PUT", url, data=payload)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to update space: {e}")

    async def delete_space(
        self,
        space_id: str,
        confirm: bool = False,
        cascade: bool = False,
    ) -> Dict[str, Any]:
        """
        Delete a space and optionally all its contents asynchronously.

        Args:
            space_id: ID of the space to delete
            confirm: Whether to confirm the deletion (required for safety)
            cascade: Whether to delete all contents in the space

        Returns:
            Dictionary with deletion result

        Raises:
            HeySolError: If the deletion fails
        """
        if not confirm:
            raise ValidationError("Space deletion requires confirmation (confirm=True)")

        # Use direct API call for space deletion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/spaces/{space_id}"

        params = {
            "cascade": cascade,
        }

        try:
            response = await self._make_request("DELETE", url, params=params)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to delete space: {e}")

    async def get_or_create_space(self, name: str, description: str = "") -> str:
        """
        Get existing space or create a new one asynchronously.

        Args:
            name: Name of the space
            description: Description for new space if created

        Returns:
            Space ID

        Raises:
            HeySolError: If the operation fails
        """
        try:
            spaces = await self.get_spaces()

            # Look for existing space
            for space in spaces:
                if space.get("name") == name and space.get("writable", True):
                    space_id = space.get("id") or space.get("spaceId")
                    if space_id:
                        self.logger.info(f"Found existing space '{name}' with ID: {space_id}")
                        return space_id

            # Create new space if not found
            self.logger.info(f"Space '{name}' not found, creating new space")
            return await self.create_space(name, description)

        except Exception as e:
            raise HeySolError(f"Failed to get or create space '{name}': {e}")

    async def ingest(
        self,
        message: str,
        space_id: Optional[str] = None,
        source: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest data into CORE Memory asynchronously.

        Args:
            message: The message to ingest
            space_id: Optional space ID to store the message in
            source: Source identifier for the message
            priority: Priority level ('low', 'normal', 'high')
            tags: Optional list of tags for the ingestion

        Returns:
            Dictionary with the ingestion result

        Raises:
            HeySolError: If ingestion fails
        """
        if not message:
            raise ValidationError("Message is required for ingestion")

        # Prepare ingestion arguments
        ingest_args = {
            "message": message,
            "source": source or self.config.source,
            "priority": priority,
        }

        if tags:
            ingest_args["tags"] = tags

        # Use provided space_id or get the default space ID
        if space_id:
            ingest_args["spaceId"] = space_id
        else:
            # Use clinical trials space as default
            try:
                clinical_trials_space_id = await self.get_or_create_space(
                    self.config.default_spaces.get("clinical_trials", "Clinical Trials")
                )
                ingest_args["spaceId"] = clinical_trials_space_id
            except Exception:
                # Continue without space_id if default space creation fails
                self.logger.warning("Could not determine default space ID, proceeding without it")

        try:
            result = await self._call_tool("memory_ingest", ingest_args, stream=True)
            self.logger.info("Successfully ingested message into CORE Memory")
            return result

        except Exception as e:
            raise HeySolError(f"Failed to ingest message: {e}")

    async def search(
        self,
        query: str,
        space_id: Optional[str] = None,
        limit: int = 10,
        valid_at: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for memories in CORE Memory asynchronously.

        Args:
            query: Search query
            space_id: Optional space ID to search in
            limit: Maximum number of results to return
            valid_at: Optional timestamp for time-based search
            start_time: Optional start time for time range search
            end_time: Optional end time for time range search

        Returns:
            Dictionary with search results containing 'episodes' and 'facts' arrays

        Raises:
            HeySolError: If search fails
        """
        if not query:
            raise ValidationError("Search query is required")

        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        # Prepare search arguments
        search_args = {
            "query": query,
            "limit": limit,
        }

        if space_id:
            search_args["spaceId"] = space_id
        if valid_at:
            search_args["validAt"] = valid_at
        if start_time:
            search_args["startTime"] = start_time
        if end_time:
            search_args["endTime"] = end_time

        try:
            result = await self._call_tool("memory_search", search_args)

            # Ensure result has expected structure
            if not isinstance(result, dict):
                result = {"episodes": [], "facts": []}

            if "episodes" not in result:
                result["episodes"] = []
            if "facts" not in result:
                result["facts"] = []

            self.logger.info(f"Search completed. Found {len(result['episodes'])} episodes and {len(result['facts'])} facts")
            return result

        except Exception as e:
            raise HeySolError(f"Failed to search memory: {e}")

    async def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the current user's profile asynchronously.

        Returns:
            Dictionary containing user profile information

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for user profile
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/profile"

        try:
            response = await self._make_request("GET", url)
            profile = await self._parse_json_response(response)
            self.logger.info("Successfully retrieved user profile")
            return profile

        except Exception as e:
            raise HeySolError(f"Failed to get user profile: {e}")

    async def close(self) -> None:
        """Close the aiohttp session and clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("Async HeySol API client session closed")

    async def get_ingestion_logs(
        self,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get ingestion logs from CORE Memory asynchronously.

        Args:
            space_id: Optional space ID to filter logs
            limit: Maximum number of logs to return (default: 100)
            offset: Offset for pagination (default: 0)
            status: Optional status filter (e.g., 'success', 'failed', 'pending')
            start_date: Optional start date filter (ISO 8601 format)
            end_date: Optional end date filter (ISO 8601 format)

        Returns:
            List of ingestion log entries

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for log retrieval
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs"

        params = {
            "limit": limit,
            "offset": offset,
        }

        if space_id:
            params["space_id"] = space_id
        if status:
            params["status"] = status
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            response = await self._make_request("GET", url, params=params)
            logs = await self._parse_json_response(response)
            return logs if isinstance(logs, list) else []
        except Exception as e:
            raise HeySolError(f"Failed to get ingestion logs: {e}")

    async def get_specific_log(self, log_id: str) -> Dict[str, Any]:
        """
        Get a specific ingestion log by ID asynchronously.

        Args:
            log_id: The ID of the ingestion log to retrieve

        Returns:
            Dictionary containing the specific log details

        Raises:
            HeySolError: If the request fails
        """
        if not log_id:
            raise ValidationError("Log ID is required")

        # Use direct API call for specific log retrieval
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs/{log_id}"

        try:
            response = await self._make_request("GET", url)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to get log {log_id}: {e}")

    async def delete_log_entry(self, log_id: str) -> Dict[str, Any]:
        """
        Delete a specific ingestion log entry asynchronously.

        Args:
            log_id: The ID of the ingestion log entry to delete

        Returns:
            Dictionary containing the deletion confirmation

        Raises:
            HeySolError: If the request fails
        """
        if not log_id:
            raise ValidationError("Log ID is required")

        # Use direct API call for log deletion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/logs/{log_id}"

        try:
            response = await self._make_request("DELETE", url)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to delete log {log_id}: {e}")

    async def add_data_to_ingestion_queue(
        self,
        data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        space_id: Optional[str] = None,
        priority: str = "normal",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add data to the ingestion queue for processing asynchronously.

        Args:
            data: Data to ingest (string, dict, or list of dicts)
            space_id: Optional space ID to store the data in
            priority: Priority level ('low', 'normal', 'high')
            tags: Optional list of tags for the data
            metadata: Optional metadata for the ingestion

        Returns:
            Dictionary with ingestion queue result

        Raises:
            HeySolError: If ingestion fails
        """
        # Use direct API call for queue ingestion
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/ingestion/queue"

        payload = {
            "data": data,
            "priority": priority,
        }

        if space_id:
            payload["space_id"] = space_id
        if tags:
            payload["tags"] = tags
        if metadata:
            payload["metadata"] = metadata

        try:
            response = await self._make_request("POST", url, data=payload)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to add data to ingestion queue: {e}")

    async def get_episode_facts(
        self,
        episode_id: Optional[str] = None,
        space_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get episode facts from CORE Memory asynchronously.

        Args:
            episode_id: Optional episode ID to filter facts
            space_id: Optional space ID to filter facts
            limit: Maximum number of facts to return (default: 100)
            offset: Offset for pagination (default: 0)
            include_metadata: Whether to include metadata in response

        Returns:
            List of episode facts

        Raises:
            HeySolError: If the request fails
        """
        # Use direct API call for episode facts
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/episodes/facts"

        params = {
            "limit": limit,
            "offset": offset,
            "include_metadata": include_metadata,
        }

        if episode_id:
            params["episode_id"] = episode_id
        if space_id:
            params["space_id"] = space_id

        try:
            response = await self._make_request("GET", url, params=params)
            facts = await self._parse_json_response(response)
            return facts if isinstance(facts, list) else []
        except Exception as e:
            raise HeySolError(f"Failed to get episode facts: {e}")

    async def search_knowledge_graph(
        self,
        query: str,
        space_id: Optional[str] = None,
        limit: int = 10,
        depth: int = 2,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Search the knowledge graph for related concepts and entities asynchronously.

        Args:
            query: Search query for knowledge graph
            space_id: Optional space ID to search in
            limit: Maximum number of results to return
            depth: Depth of graph traversal (1-5)
            include_metadata: Whether to include metadata in results

        Returns:
            Dictionary with knowledge graph search results

        Raises:
            HeySolError: If search fails
        """
        if not query:
            raise ValidationError("Search query is required")

        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        if depth < 1 or depth > 5:
            raise ValidationError("Depth must be between 1 and 5")

        # Use direct API call for knowledge graph search
        base_url = self.config.base_url.split("/api/v1/mcp")[0]
        url = f"{base_url}/api/v1/knowledge-graph/search"

        params = {
            "q": query,
            "limit": limit,
            "depth": depth,
            "include_metadata": include_metadata,
        }

        if space_id:
            params["space_id"] = space_id

        try:
            response = await self._make_request("GET", url, params=params)
            return await self._parse_json_response(response)
        except Exception as e:
            raise HeySolError(f"Failed to search knowledge graph: {e}")

    async def __aenter__(self) -> "AsyncHeySolClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()