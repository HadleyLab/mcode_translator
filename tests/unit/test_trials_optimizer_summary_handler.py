#!/usr/bin/env python3
"""
Unit tests for trials_optimizer SummaryHandler.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.cli.trials_optimizer import SummaryHandler


class TestSummaryHandler:
    """Test the SummaryHandler class."""

    def test_summary_handler_on_created(self):
        """Test SummaryHandler file creation event handling."""
        # Mock workflow
        mock_workflow = MagicMock()

        # Create handler
        handler = SummaryHandler(mock_workflow)

        # Mock event for JSON file creation
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "optimization_runs/result.json"

        # Call on_created
        with patch("builtins.print") as mock_print:
            handler.on_created(mock_event)

            # Verify print was called and workflow method was invoked
            mock_print.assert_called_with("\n📊 Real-time summary updated:")
            mock_workflow.summarize_benchmark_validations.assert_called_once()

    def test_summary_handler_on_created_directory(self):
        """Test SummaryHandler ignores directory creation events."""
        # Mock workflow
        mock_workflow = MagicMock()

        # Create handler
        handler = SummaryHandler(mock_workflow)

        # Mock event for directory creation
        mock_event = MagicMock()
        mock_event.is_directory = True
        mock_event.src_path = "optimization_runs/new_dir"

        # Call on_created
        handler.on_created(mock_event)

        # Verify workflow method was NOT called
        mock_workflow.summarize_benchmark_validations.assert_not_called()

    def test_summary_handler_on_created_non_json(self):
        """Test SummaryHandler ignores non-JSON file creation events."""
        # Mock workflow
        mock_workflow = MagicMock()

        # Create handler
        handler = SummaryHandler(mock_workflow)

        # Mock event for non-JSON file creation
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "optimization_runs/result.txt"

        # Call on_created
        handler.on_created(mock_event)

        # Verify workflow method was NOT called
        mock_workflow.summarize_benchmark_validations.assert_not_called()