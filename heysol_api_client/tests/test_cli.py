#!/usr/bin/env python3
"""
Unit Tests for HeySol API Client - CLI Interface

Tests the command-line interface functionality including argument parsing,
command execution, and error handling.
"""

import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner
import click

import sys
from pathlib import Path

# Add the parent directory to the Python path to import the CLI
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import cli
from scripts.common import create_client


class TestCLI:
    """Unit tests for CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "HeySol API Client CLI" in result.output
        assert "Setup:" in result.output

    def test_cli_profile_help(self, runner):
        """Test profile command help."""
        result = runner.invoke(cli, ["profile", "--help"])
        assert result.exit_code == 0
        assert "User profile operations" in result.output

    def test_cli_spaces_help(self, runner):
        """Test spaces command help."""
        result = runner.invoke(cli, ["spaces", "--help"])
        assert result.exit_code == 0
        assert "Memory space operations" in result.output

    def test_cli_memory_help(self, runner):
        """Test memory command help."""
        result = runner.invoke(cli, ["memory", "--help"])
        assert result.exit_code == 0
        assert "Memory operations" in result.output

    def test_cli_logs_help(self, runner):
        """Test logs command help."""
        result = runner.invoke(cli, ["logs", "--help"])
        assert result.exit_code == 0
        assert "Log operations" in result.output

    def test_cli_tools_help(self, runner):
        """Test tools command help."""
        result = runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0
        assert "MCP tools operations" in result.output

    def test_cli_webhooks_help(self, runner):
        """Test webhooks command help."""
        result = runner.invoke(cli, ["webhooks", "--help"])
        assert result.exit_code == 0
        assert "Webhook operations" in result.output

    @patch("scripts.profile.create_client")
    def test_cli_profile_get_success(self, mock_create_client, runner):
        """Test successful profile get command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.get_user_profile.return_value = {"name": "Test User", "email": "test@example.com"}
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 0
        assert '"name": "Test User"' in result.output
        assert '"email": "test@example.com"' in result.output
        mock_client.get_user_profile.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("scripts.profile.create_client")
    def test_cli_profile_get_error(self, mock_create_client, runner):
        """Test profile get command with error."""
        # Mock client to raise exception
        mock_client = Mock()
        mock_client.get_user_profile.side_effect = Exception("API Error")
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 1
        assert "API Error" in str(result.exception)
        mock_client.close.assert_called_once()

    @patch("scripts.spaces.create_client")
    def test_cli_spaces_list_success(self, mock_create_client, runner):
        """Test successful spaces list command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.get_spaces.return_value = [
            {"id": "space-1", "name": "Test Space 1"},
            {"id": "space-2", "name": "Test Space 2"}
        ]
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["spaces", "list", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 0
        assert '"id": "space-1"' in result.output
        assert '"name": "Test Space 1"' in result.output
        mock_client.get_spaces.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("scripts.spaces.create_client")
    def test_cli_spaces_create_success(self, mock_create_client, runner):
        """Test successful spaces create command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.create_space.return_value = "space-123"
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "spaces", "create", "Test Space",
            "--description", "A test space",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"space_id": "space-123"' in result.output
        assert '"name": "Test Space"' in result.output
        mock_client.create_space.assert_called_once_with("Test Space", "A test space")
        mock_client.close.assert_called_once()

    def test_cli_spaces_create_missing_name(self, runner):
        """Test spaces create command with missing name."""
        result = runner.invoke(cli, ["spaces", "create"])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    @patch("scripts.memory.create_client")
    def test_cli_memory_ingest_success(self, mock_create_client, runner):
        """Test successful memory ingest command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.ingest.return_value = {"success": True, "id": "episode-123"}
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "ingest", "Test message",
            "--space-id", "space-123",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        assert '"id": "episode-123"' in result.output
        mock_client.ingest.assert_called_once_with(
            message="Test message",
            space_id="space-123",
            session_id=None
        )
        mock_client.close.assert_called_once()

    @patch("scripts.memory.create_client")
    def test_cli_memory_ingest_from_file(self, mock_create_client, runner, tmp_path):
        """Test memory ingest command with file input."""
        # Create a temporary file
        test_file = tmp_path / "test_message.txt"
        test_file.write_text("Message from file")

        # Mock client and response
        mock_client = Mock()
        mock_client.ingest.return_value = {"success": True, "id": "episode-456"}
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "ingest",
            "--file", str(test_file),
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        mock_client.ingest.assert_called_once_with(
            message="Message from file",
            space_id=None,
            session_id=None
        )
        mock_client.close.assert_called_once()

    def test_cli_memory_ingest_missing_message(self, runner):
        """Test memory ingest command with missing message."""
        result = runner.invoke(cli, ["memory", "ingest", "--api-key", "test-key"])
        assert result.exit_code == 1
        assert "Message or file is required" in result.output

    @patch("scripts.memory.create_client")
    def test_cli_memory_search_success(self, mock_create_client, runner):
        """Test successful memory search command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.search.return_value = {
            "episodes": [{"content": "Test result", "id": "ep-123"}],
            "total": 1
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "search", "test query",
            "--limit", "10",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"content": "Test result"' in result.output
        assert '"total": 1' in result.output
        mock_client.search.assert_called_once_with(
            query="test query",
            space_ids=None,
            limit=10,
            include_invalidated=False
        )
        mock_client.close.assert_called_once()

    @patch("scripts.logs.create_client")
    def test_cli_logs_list_success(self, mock_create_client, runner):
        """Test successful logs list command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.get_ingestion_logs.return_value = [
            {"id": "log-1", "status": "success", "timestamp": "2024-01-01T00:00:00Z"}
        ]
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "logs", "list",
            "--space-id", "space-123",
            "--limit", "50",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"id": "log-1"' in result.output
        assert '"status": "success"' in result.output
        mock_client.get_ingestion_logs.assert_called_once_with(
            space_id="space-123",
            limit=50,
            offset=0,
            status=None,
            start_date=None,
            end_date=None
        )
        mock_client.close.assert_called_once()

    @patch("scripts.tools.create_client")
    def test_cli_tools_list_success(self, mock_create_client, runner):
        """Test successful tools list command."""
        # Mock client and response
        mock_client = Mock()
        mock_client.get_available_tools.return_value = {
            "memory_ingest": {"name": "memory_ingest", "description": "Ingest data"},
            "memory_search": {"name": "memory_search", "description": "Search data"}
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["tools", "list", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 0
        assert '"memory_ingest"' in result.output
        assert '"memory_search"' in result.output
        mock_client.get_available_tools.assert_called_once()
        mock_client.close.assert_called_once()


    @patch("scripts.profile.create_client")
    def test_cli_keyboard_interrupt(self, mock_create_client, runner):
        """Test CLI handling of keyboard interrupt."""
        # Mock client to raise KeyboardInterrupt
        mock_client = Mock()
        mock_client.get_user_profile.side_effect = KeyboardInterrupt()
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 1
        # Click handles KeyboardInterrupt specially
        assert result.exception is not None
        mock_client.close.assert_called_once()

    @patch("scripts.profile.create_client")
    def test_cli_unexpected_error(self, mock_create_client, runner):
        """Test CLI handling of unexpected errors."""
        # Mock client to raise unexpected exception
        mock_client = Mock()
        mock_client.get_user_profile.side_effect = RuntimeError("Unexpected error")
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 1
        assert "Unexpected error" in str(result.exception)
        mock_client.close.assert_called_once()

    @patch("scripts.profile.create_client")
    def test_cli_pretty_output(self, mock_create_client, runner):
        """Test CLI pretty output formatting."""
        mock_client = Mock()
        mock_client.get_user_profile.return_value = {"name": "Test User", "nested": {"value": 42}}
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--pretty", "--skip-mcp"])
        assert result.exit_code == 0
        # Pretty JSON should have indentation
        assert "  \"name\":" in result.output
        assert "  \"nested\":" in result.output

    @patch("scripts.profile.create_client")
    def test_cli_compact_output(self, mock_create_client, runner):
        """Test CLI compact output formatting."""
        mock_client = Mock()
        mock_client.get_user_profile.return_value = {"name": "Test User", "nested": {"value": 42}}
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, ["profile", "get", "--api-key", "test-key", "--skip-mcp"])
        assert result.exit_code == 0
        # Compact JSON should be all on one line
        assert '"name": "Test User"' in result.output
        assert '"nested": {"value": 42}' in result.output

    def test_cli_invalid_command(self, runner):
        """Test CLI with invalid command."""
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code == 2
        assert "No such command" in result.output

    def test_cli_spaces_invalid_subcommand(self, runner):
        """Test spaces command with invalid subcommand."""
        result = runner.invoke(cli, ["spaces", "invalid-subcommand"])
        assert result.exit_code == 2
        assert "No such command" in result.output

    @patch("scripts.memory.create_client")
    def test_cli_memory_search_graph_success(self, mock_create_client, runner):
        """Test successful memory search-graph command."""
        mock_client = Mock()
        mock_client.search_knowledge_graph.return_value = {
            "results": [{"content": "Graph result", "id": "node-123"}],
            "total": 1
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "search-graph", "test query",
            "--space-id", "space-123",
            "--depth", "3",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"content": "Graph result"' in result.output
        mock_client.search_knowledge_graph.assert_called_once_with(
            query="test query",
            space_id="space-123",
            limit=10,
            depth=3,
            include_metadata=True
        )
        mock_client.close.assert_called_once()

    @patch("scripts.memory.create_client")
    def test_cli_memory_queue_success(self, mock_create_client, runner):
        """Test successful memory queue command."""
        mock_client = Mock()
        mock_client.add_data_to_ingestion_queue.return_value = {
            "success": True,
            "queueId": "queue-123"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "queue", "test data",
            "--space-id", "space-123",
            "--priority", "high",
            "--tags", "tag1",
            "--tags", "tag2",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        assert '"queueId": "queue-123"' in result.output
        mock_client.add_data_to_ingestion_queue.assert_called_once_with(
            data="test data",
            space_id="space-123",
            priority="high",
            tags=["tag1", "tag2"],
            metadata=None
        )
        mock_client.close.assert_called_once()

    @patch("scripts.memory.create_client")
    def test_cli_memory_episode_success(self, mock_create_client, runner):
        """Test successful memory episode command."""
        mock_client = Mock()
        mock_client.get_episode_facts.return_value = [
            {"fact": "Test fact", "confidence": 0.95}
        ]
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "memory", "episode", "episode-123",
            "--limit", "50",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"fact": "Test fact"' in result.output
        mock_client.get_episode_facts.assert_called_once_with(
            episode_id="episode-123",
            limit=50,
            offset=0,
            include_metadata=True
        )
        mock_client.close.assert_called_once()

    @patch("scripts.spaces.create_client")
    def test_cli_spaces_update_success(self, mock_create_client, runner):
        """Test successful spaces update command."""
        mock_client = Mock()
        mock_client.update_space.return_value = {
            "id": "space-123",
            "name": "Updated Space",
            "description": "Updated description"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "spaces", "update", "space-123",
            "--name", "Updated Space",
            "--description", "Updated description",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"name": "Updated Space"' in result.output
        mock_client.update_space.assert_called_once_with(
            space_id="space-123",
            name="Updated Space",
            description="Updated description",
            metadata=None
        )
        mock_client.close.assert_called_once()

    @patch("scripts.spaces.create_client")
    def test_cli_spaces_bulk_ops_success(self, mock_create_client, runner):
        """Test successful spaces bulk-ops command."""
        mock_client = Mock()
        mock_client.bulk_space_operations.return_value = {
            "success": True,
            "affected_spaces": 3
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "spaces", "bulk-ops", "archive",
            "--space-id", "space-123",
            "--statement-ids", "stmt-1",
            "--statement-ids", "stmt-2",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        mock_client.bulk_space_operations.assert_called_once_with(
            intent="archive",
            space_id="space-123",
            statement_ids=["stmt-1", "stmt-2"],
            space_ids=None
        )
        mock_client.close.assert_called_once()

    @patch("scripts.spaces.create_client")
    def test_cli_spaces_delete_success(self, mock_create_client, runner):
        """Test successful spaces delete command."""
        mock_client = Mock()
        mock_client.delete_space.return_value = {
            "success": True,
            "deleted_space": "space-123"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "spaces", "delete", "space-123",
            "--confirm",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        mock_client.delete_space.assert_called_once_with(space_id="space-123", confirm=True)
        mock_client.close.assert_called_once()

    def test_cli_spaces_delete_missing_confirm(self, runner):
        """Test spaces delete command without confirm flag."""
        result = runner.invoke(cli, [
            "spaces", "delete", "space-123",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 1
        assert "Space deletion requires --confirm flag for safety" in result.output

    @patch("scripts.logs.create_client")
    def test_cli_logs_delete_success(self, mock_create_client, runner):
        """Test successful logs delete command."""
        mock_client = Mock()
        mock_client.delete_logs_by_source.return_value = {
            "message": "Deleted 5 logs with source 'test-source'",
            "deleted_count": 5,
            "total_found": 5
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "logs", "delete", "test-source",
            "--confirm",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"deleted_count": 5' in result.output
        mock_client.delete_logs_by_source.assert_called_once_with(
            source="test-source",
            space_id=None,
            confirm=True
        )
        mock_client.close.assert_called_once()

    def test_cli_logs_delete_missing_confirm(self, runner):
        """Test logs delete command without confirm flag."""
        result = runner.invoke(cli, [
            "logs", "delete", "test-source",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 1
        assert "Deletion requires --confirm flag for safety" in result.output

    @patch("scripts.logs.create_client")
    def test_cli_logs_delete_entry_success(self, mock_create_client, runner):
        """Test successful logs delete-entry command."""
        mock_client = Mock()
        mock_client.delete_log_entry.return_value = {
            "success": True,
            "deleted_log": "log-123"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "logs", "delete-entry", "log-123",
            "--confirm",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        mock_client.delete_log_entry.assert_called_once_with(log_id="log-123")
        mock_client.close.assert_called_once()

    @patch("scripts.logs.create_client")
    def test_cli_logs_get_success(self, mock_create_client, runner):
        """Test successful logs get command."""
        mock_client = Mock()
        mock_client.get_specific_log.return_value = {
            "id": "log-123",
            "status": "success",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "logs", "get", "log-123",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"id": "log-123"' in result.output
        assert '"status": "success"' in result.output
        mock_client.get_specific_log.assert_called_once_with(log_id="log-123")
        mock_client.close.assert_called_once()

    @patch("scripts.logs.create_client")
    def test_cli_logs_list_with_source_filter(self, mock_create_client, runner):
        """Test logs list command with source filtering."""
        mock_client = Mock()
        # Mock logs with different sources
        mock_client.get_ingestion_logs.return_value = [
            {"id": "log-1", "source": "source-a", "status": "success"},
            {"id": "log-2", "source": "source-b", "status": "success"},
            {"id": "log-3", "source": "source-a", "status": "error"}
        ]
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "logs", "list",
            "--source", "source-a",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        # Should only show logs from source-a
        assert '"id": "log-1"' in result.output
        assert '"id": "log-3"' in result.output
        assert '"id": "log-2"' not in result.output  # source-b should be filtered out

    @patch("scripts.webhooks.create_client")
    def test_cli_webhooks_create_success(self, mock_create_client, runner):
        """Test successful webhooks create command."""
        mock_client = Mock()
        mock_client.register_webhook.return_value = {
            "id": "webhook-123",
            "url": "https://example.com/webhook",
            "secret": "secret123"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "webhooks", "create", "https://example.com/webhook",
            "--secret", "secret123",
            "--events", "memory.created",
            "--events", "memory.updated",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"id": "webhook-123"' in result.output
        mock_client.register_webhook.assert_called_once_with(
            url="https://example.com/webhook",
            events=["memory.created", "memory.updated"],
            secret="secret123"
        )
        mock_client.close.assert_called_once()

    @patch("scripts.webhooks.create_client")
    def test_cli_webhooks_get_success(self, mock_create_client, runner):
        """Test successful webhooks get command."""
        mock_client = Mock()
        mock_client.get_webhook.return_value = {
            "id": "webhook-123",
            "url": "https://example.com/webhook",
            "active": True
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "webhooks", "get", "webhook-123",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"id": "webhook-123"' in result.output
        assert '"active": true' in result.output
        mock_client.get_webhook.assert_called_once_with(webhook_id="webhook-123")
        mock_client.close.assert_called_once()

    @patch("scripts.webhooks.create_client")
    def test_cli_webhooks_list_success(self, mock_create_client, runner):
        """Test successful webhooks list command."""
        mock_client = Mock()
        mock_client.list_webhooks.return_value = [
            {"id": "webhook-1", "url": "https://example.com/1"},
            {"id": "webhook-2", "url": "https://example.com/2"}
        ]
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "webhooks", "list",
            "--space-id", "space-123",
            "--active", "true",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"id": "webhook-1"' in result.output
        assert '"id": "webhook-2"' in result.output
        mock_client.list_webhooks.assert_called_once_with(
            space_id="space-123",
            active=True,
            limit=100,
            offset=0
        )
        mock_client.close.assert_called_once()

    @patch("scripts.webhooks.create_client")
    def test_cli_webhooks_update_success(self, mock_create_client, runner):
        """Test successful webhooks update command."""
        mock_client = Mock()
        mock_client.update_webhook.return_value = {
            "id": "webhook-123",
            "url": "https://new-url.com",
            "active": False
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "webhooks", "update", "webhook-123", "https://new-url.com",
            "--events", "memory.deleted",
            "--secret", "new-secret",
            "--inactive",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"url": "https://new-url.com"' in result.output
        assert '"active": false' in result.output
        mock_client.update_webhook.assert_called_once_with(
            webhook_id="webhook-123",
            url="https://new-url.com",
            events=["memory.deleted"],
            secret="new-secret",
            active=False
        )
        mock_client.close.assert_called_once()

    @patch("scripts.webhooks.create_client")
    def test_cli_webhooks_delete_success(self, mock_create_client, runner):
        """Test successful webhooks delete command."""
        mock_client = Mock()
        mock_client.delete_webhook.return_value = {
            "success": True,
            "deleted_webhook": "webhook-123"
        }
        mock_client.close = Mock()
        mock_create_client.return_value = mock_client

        result = runner.invoke(cli, [
            "webhooks", "delete", "webhook-123",
            "--confirm",
            "--api-key", "test-key",
            "--skip-mcp"
        ])
        assert result.exit_code == 0
        assert '"success": true' in result.output
        mock_client.delete_webhook.assert_called_once_with(webhook_id="webhook-123", confirm=True)
        mock_client.close.assert_called_once()

    def test_cli_webhooks_delete_missing_confirm(self, runner):
        """Test webhooks delete command without confirm flag."""
        result = runner.invoke(cli, [
            "webhooks", "delete", "webhook-123",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 1
        assert "Webhook deletion requires --confirm flag for safety" in result.output

    def test_cli_webhooks_create_missing_secret(self, runner):
        """Test webhooks create command with missing secret."""
        result = runner.invoke(cli, [
            "webhooks", "create", "https://example.com/webhook",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing option" in result.output or "required" in result.output

    def test_cli_webhooks_update_missing_events(self, runner):
        """Test webhooks update command with missing events."""
        result = runner.invoke(cli, [
            "webhooks", "update", "webhook-123", "https://new-url.com",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing option" in result.output or "required" in result.output

    def test_cli_memory_queue_missing_data(self, runner):
        """Test memory queue command with missing data."""
        result = runner.invoke(cli, [
            "memory", "queue",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 1
        assert "Data or file is required" in result.output

    def test_cli_memory_episode_missing_id(self, runner):
        """Test memory episode command with missing episode ID."""
        result = runner.invoke(cli, [
            "memory", "episode",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_logs_get_missing_id(self, runner):
        """Test logs get command with missing log ID."""
        result = runner.invoke(cli, [
            "logs", "get",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_spaces_update_missing_id(self, runner):
        """Test spaces update command with missing space ID."""
        result = runner.invoke(cli, [
            "spaces", "update",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_spaces_bulk_ops_missing_intent(self, runner):
        """Test spaces bulk-ops command with missing intent."""
        result = runner.invoke(cli, [
            "spaces", "bulk-ops",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_spaces_delete_missing_id(self, runner):
        """Test spaces delete command with missing space ID."""
        result = runner.invoke(cli, [
            "spaces", "delete",
            "--confirm",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_webhooks_get_missing_id(self, runner):
        """Test webhooks get command with missing webhook ID."""
        result = runner.invoke(cli, [
            "webhooks", "get",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_webhooks_update_missing_url(self, runner):
        """Test webhooks update command with missing URL."""
        result = runner.invoke(cli, [
            "webhooks", "update", "webhook-123",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output

    def test_cli_webhooks_delete_missing_id(self, runner):
        """Test webhooks delete command with missing webhook ID."""
        result = runner.invoke(cli, [
            "webhooks", "delete",
            "--confirm",
            "--api-key", "test-key"
        ])
        assert result.exit_code == 2  # Click parameter error
        assert "Missing argument" in result.output