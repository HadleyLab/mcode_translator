"""
Unit tests for onco_core_memory module.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.utils.onco_core_memory import OncoCoreClient


class TestOncoCoreClientBase:
    """Base class for OncoCoreClient tests with common patterns."""

    def setup_mock_config(self, mock_config):
        """Helper to setup mock config with common values."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"
        return mock_config_instance

    def setup_mock_mcp_instance(self, mock_mcp_client):
        """Helper to setup mock MCP instance."""
        mock_mcp_instance = MagicMock()
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_client.return_value = mock_mcp_instance
        return mock_mcp_instance

    def setup_mock_api_instance(self, mock_api_client):
        """Helper to setup mock API instance."""
        mock_api_instance = MagicMock()
        mock_api_client.return_value = mock_api_instance
        return mock_api_instance

    def assert_client_initialization(self, client, expected_api_key="test_key", expected_base_url="https://api.example.com"):
        """Helper to assert client initialization."""
        assert client.api_key == expected_api_key
        assert client.base_url == expected_base_url

    def assert_mcp_preference(self, client, prefer_mcp, expected_method):
        """Helper to assert MCP preference behavior."""
        assert client.get_preferred_access_method() == expected_method


class TestOncoCoreClient(TestOncoCoreClientBase):
    """Test suite for OncoCoreClient."""

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_init_with_api_key(self, mock_mcp_client, mock_api_client, mock_config):
        """Test initialization with API key."""
        mock_config_instance = self.setup_mock_config(mock_config)
        mock_config_instance.base_url = "https://api.example.com"
        mock_config_instance.source = "test"
        mock_config_instance.timeout = 30
        mock_config_instance.mcp_url = "https://mcp.example.com"

        client = OncoCoreClient(api_key="test_key")

        self.assert_client_initialization(client)
        mock_api_client.assert_called_once()
        mock_mcp_client.assert_called_once()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_init_without_api_key_raises_error(self, mock_mcp_client, mock_api_client, mock_config):
        """Test initialization without API key raises error."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = None

        with pytest.raises(Exception):  # ValidationError
            OncoCoreClient()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_init_skip_mcp_init(self, mock_mcp_client, mock_api_client, mock_config):
        """Test initialization with skip_mcp_init=True."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient(api_key="test_key", skip_mcp_init=True)

        assert client.mcp_client is None
        mock_mcp_client.assert_not_called()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_init_mcp_init_fails(self, mock_mcp_client, mock_api_client, mock_config):
        """Test initialization when MCP init fails."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"
        mock_mcp_client.side_effect = Exception("MCP failed")

        client = OncoCoreClient(api_key="test_key")

        assert client.mcp_client is None

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_from_env(self, mock_mcp_client, mock_api_client, mock_config):
        """Test from_env class method."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient.from_env()

        assert isinstance(client, OncoCoreClient)
        mock_config.from_env.assert_called()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_is_mcp_available_true(self, mock_mcp_client, mock_api_client, mock_config):
        """Test is_mcp_available returns True when MCP is available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        assert client.is_mcp_available() is True

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_is_mcp_available_false(self, mock_mcp_client, mock_api_client, mock_config):
        """Test is_mcp_available returns False when MCP is not available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient(api_key="test_key", skip_mcp_init=True)

        assert client.is_mcp_available() is False

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_preferred_access_method_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_preferred_access_method returns MCP when preferred."""
        self.setup_mock_config(mock_config)
        self.setup_mock_mcp_instance(mock_mcp_client)

        client = OncoCoreClient(api_key="test_key", prefer_mcp=True)

        self.assert_mcp_preference(client, True, "mcp")

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_preferred_access_method_direct_api(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_preferred_access_method returns direct_api when MCP not preferred."""
        self.setup_mock_config(mock_config)

        client = OncoCoreClient(api_key="test_key", prefer_mcp=False)

        self.assert_mcp_preference(client, False, "direct_api")

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_ensure_mcp_available_raises_error(self, mock_mcp_client, mock_api_client, mock_config):
        """Test ensure_mcp_available raises error when MCP not available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient(api_key="test_key", skip_mcp_init=True)

        with pytest.raises(Exception):  # HeySolError
            client.ensure_mcp_available()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_available_tools(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_available_tools."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_available_tools.return_value = {"tool1": "desc1"}
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_available_tools()
        assert result == {"tool1": "desc1"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_client_info(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_client_info."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.base_url = "https://api.example.com"
        mock_api_client.return_value = mock_api_instance

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.session_id = "session123"
        mock_mcp_instance.tools = {"tool1": "desc"}
        mock_mcp_instance.mcp_url = "https://mcp.example.com"
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_client_info()
        assert result["api_client"]["available"] is True
        assert result["mcp_client"]["available"] is True

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_ingest_via_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test ingest using MCP."""
        self.setup_mock_config(mock_config)

        mock_mcp_instance = self.setup_mock_mcp_instance(mock_mcp_client)
        mock_mcp_instance.tools = ["memory_ingest"]
        mock_mcp_instance.ingest_via_mcp.return_value = {"status": "success"}

        client = OncoCoreClient(api_key="test_key", prefer_mcp=True)

        result = client.ingest("test message")
        assert result == {"status": "success"}
        mock_mcp_instance.ingest_via_mcp.assert_called_once()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_ingest_via_api(self, mock_mcp_client, mock_api_client, mock_config):
        """Test ingest using API."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.ingest.return_value = {"status": "success"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=False)

        result = client.ingest("test message")
        assert result == {"status": "success"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_search_via_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test search using MCP."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_instance.tools = ["memory_search"]
        mock_mcp_instance.search_via_mcp.return_value = {"results": []}
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=True)

        result = client.search("test query")
        assert result == {"results": []}
        mock_mcp_instance.search_via_mcp.assert_called_once()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_search_via_api(self, mock_mcp_client, mock_api_client, mock_config):
        """Test search using API."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.search.return_value = {"results": []}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=False)

        result = client.search("test query")
        assert result == {"results": []}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_spaces_via_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_spaces using MCP."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_instance.tools = ["memory_get_spaces"]
        mock_mcp_instance.get_memory_spaces_via_mcp.return_value = ["space1", "space2"]
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=True)

        result = client.get_spaces()
        assert result == ["space1", "space2"]
        mock_mcp_instance.get_memory_spaces_via_mcp.assert_called_once()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_spaces_via_api(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_spaces using API."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_spaces.return_value = ["space1", "space2"]
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=False)

        result = client.get_spaces()
        assert result == ["space1", "space2"]

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_create_space(self, mock_mcp_client, mock_api_client, mock_config):
        """Test create_space."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.create_space.return_value = "space_id_123"
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.create_space("test space", "description")
        assert result == "space_id_123"

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_user_profile_via_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_user_profile using MCP."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.is_mcp_available.return_value = True
        mock_mcp_instance.tools = ["get_user_profile"]
        mock_mcp_instance.get_user_profile_via_mcp.return_value = {"name": "test"}
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=True)

        result = client.get_user_profile()
        assert result == {"name": "test"}
        mock_mcp_instance.get_user_profile_via_mcp.assert_called_once()

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_user_profile_via_api(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_user_profile using API."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_user_profile.return_value = {"name": "test"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key", prefer_mcp=False)

        result = client.get_user_profile()
        assert result == {"name": "test"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_search_knowledge_graph(self, mock_mcp_client, mock_api_client, mock_config):
        """Test search_knowledge_graph."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.search_knowledge_graph.return_value = {"results": []}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.search_knowledge_graph("test query")
        assert result == {"results": []}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_add_data_to_ingestion_queue(self, mock_mcp_client, mock_api_client, mock_config):
        """Test add_data_to_ingestion_queue."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.add_data_to_ingestion_queue.return_value = {"status": "queued"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.add_data_to_ingestion_queue("test data")
        assert result == {"status": "queued"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_episode_facts(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_episode_facts."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_episode_facts.return_value = [{"fact": "test"}]
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_episode_facts("episode123")
        assert result == [{"fact": "test"}]

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_ingestion_logs(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_ingestion_logs."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_ingestion_logs.return_value = [{"log": "test"}]
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_ingestion_logs()
        assert result == [{"log": "test"}]

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_specific_log(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_specific_log."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_specific_log.return_value = {"log": "details"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_specific_log("log123")
        assert result == {"log": "details"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_check_ingestion_status(self, mock_mcp_client, mock_api_client, mock_config):
        """Test check_ingestion_status."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.check_ingestion_status.return_value = {"status": "complete"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.check_ingestion_status()
        assert result == {"status": "complete"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_bulk_space_operations(self, mock_mcp_client, mock_api_client, mock_config):
        """Test bulk_space_operations."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.bulk_space_operations.return_value = {"result": "success"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.bulk_space_operations("test_intent")
        assert result == {"result": "success"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_space_details(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_space_details."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_space_details.return_value = {"name": "test_space"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_space_details("space123")
        assert result == {"name": "test_space"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_update_space(self, mock_mcp_client, mock_api_client, mock_config):
        """Test update_space."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.update_space.return_value = {"status": "updated"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.update_space("space123", name="new_name")
        assert result == {"status": "updated"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_delete_space(self, mock_mcp_client, mock_api_client, mock_config):
        """Test delete_space."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.delete_space.return_value = {"status": "deleted"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.delete_space("space123")
        assert result == {"status": "deleted"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_register_webhook(self, mock_mcp_client, mock_api_client, mock_config):
        """Test register_webhook."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.register_webhook.return_value = {"id": "webhook123"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.register_webhook("https://example.com/webhook")
        assert result == {"id": "webhook123"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_list_webhooks(self, mock_mcp_client, mock_api_client, mock_config):
        """Test list_webhooks."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.list_webhooks.return_value = [{"id": "webhook1"}]
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.list_webhooks()
        assert result == [{"id": "webhook1"}]

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_webhook(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_webhook."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.get_webhook.return_value = {"id": "webhook123", "url": "https://example.com"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_webhook("webhook123")
        assert result == {"id": "webhook123", "url": "https://example.com"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_update_webhook(self, mock_mcp_client, mock_api_client, mock_config):
        """Test update_webhook."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.update_webhook.return_value = {"status": "updated"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.update_webhook("webhook123", "https://new-url.com", ["event1"])
        assert result == {"status": "updated"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_delete_webhook(self, mock_mcp_client, mock_api_client, mock_config):
        """Test delete_webhook."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.delete_webhook.return_value = {"status": "deleted"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.delete_webhook("webhook123")
        assert result == {"status": "deleted"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_delete_log_entry(self, mock_mcp_client, mock_api_client, mock_config):
        """Test delete_log_entry."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_instance.delete_log_entry.return_value = {"status": "deleted"}
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.delete_log_entry("log123")
        assert result == {"status": "deleted"}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_delete_logs_by_source(self, mock_mcp_client, mock_api_client, mock_config):
        """Test delete_logs_by_source."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.delete_logs_by_source.return_value = {"deleted_count": 5}
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.delete_logs_by_source("test_source")
        assert result == {"deleted_count": 5}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_delete_logs_by_source_no_mcp_raises_error(self, mock_mcp_client, mock_api_client, mock_config):
        """Test delete_logs_by_source raises error when MCP not available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient(api_key="test_key", skip_mcp_init=True)

        with pytest.raises(Exception):  # HeySolError
            client.delete_logs_by_source("test_source")

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_logs_by_source_with_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_logs_by_source with MCP available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.get_logs_by_source.return_value = {"logs": [{"id": "log1"}]}
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        result = client.get_logs_by_source("test_source")
        assert result == {"logs": [{"id": "log1"}]}

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_get_logs_by_source_without_mcp(self, mock_mcp_client, mock_api_client, mock_config):
        """Test get_logs_by_source without MCP available."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        client = OncoCoreClient(api_key="test_key", skip_mcp_init=True)

        result = client.get_logs_by_source("test_source")
        assert result["logs"] == []
        assert result["source"] == "test_source"
        assert "note" in result

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_api_property(self, mock_mcp_client, mock_api_client, mock_config):
        """Test api property."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_api_client.return_value = mock_api_instance

        client = OncoCoreClient(api_key="test_key")

        assert client.api == mock_api_instance

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_mcp_property(self, mock_mcp_client, mock_api_client, mock_config):
        """Test mcp property."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_mcp_instance = MagicMock()
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        assert client.mcp == mock_mcp_instance

    @patch('src.utils.onco_core_memory.HeySolConfig')
    @patch('src.utils.onco_core_memory.HeySolAPIClient')
    @patch('src.utils.onco_core_memory.HeySolMCPClient')
    def test_close(self, mock_mcp_client, mock_api_client, mock_config):
        """Test close method."""
        mock_config_instance = MagicMock()
        mock_config.from_env.return_value = mock_config_instance
        mock_config_instance.api_key = "test_key"

        mock_api_instance = MagicMock()
        mock_mcp_instance = MagicMock()
        mock_api_client.return_value = mock_api_instance
        mock_mcp_client.return_value = mock_mcp_instance

        client = OncoCoreClient(api_key="test_key")

        client.close()

        mock_api_instance.close.assert_called_once()
        mock_mcp_instance.close.assert_called_once()