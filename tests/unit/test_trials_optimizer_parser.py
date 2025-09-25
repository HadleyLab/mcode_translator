#!/usr/bin/env python3
"""
Unit tests for trials_optimizer parser functionality.
"""

import argparse


from src.cli.trials_optimizer import create_parser


class TestTrialsOptimizerParser:
    """Test the trials_optimizer parser."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Optimize mCODE translation parameters" in parser.description

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "trials_file" in actions
        assert "cv_folds" in actions
        assert "prompts" in actions
        assert "models" in actions
        assert "list_prompts" in actions
        assert "list_models" in actions
        assert "save_mcode_elements" in actions

    def test_create_parser_trials_file_argument(self):
        """Test the trials_file argument configuration."""
        parser = create_parser()

        # Find the trials_file action
        trials_file_action = None
        for action in parser._actions:
            if action.dest == "trials_file":
                trials_file_action = action
                break

        assert trials_file_action is not None
        assert "Path to NDJSON file containing trial data" in trials_file_action.help

    def test_create_parser_cv_folds_argument(self):
        """Test the cv_folds argument configuration."""
        parser = create_parser()

        # Find the cv_folds action
        cv_folds_action = None
        for action in parser._actions:
            if action.dest == "cv_folds":
                cv_folds_action = action
                break

        assert cv_folds_action is not None
        assert cv_folds_action.type == int

    def test_create_parser_list_prompts_argument(self):
        """Test the list_prompts argument configuration."""
        parser = create_parser()

        # Find the list_prompts action
        list_prompts_action = None
        for action in parser._actions:
            if action.dest == "list_prompts":
                list_prompts_action = action
                break

        assert list_prompts_action is not None
        assert "List available prompt templates" in list_prompts_action.help

    def test_create_parser_list_models_argument(self):
        """Test the list_models argument configuration."""
        parser = create_parser()

        # Find the list_models action
        list_models_action = None
        for action in parser._actions:
            if action.dest == "list_models":
                list_models_action = action
                break

        assert list_models_action is not None
        assert "List available models" in list_models_action.help