"""
Shared CLI utilities for mCODE translator.

This module provides common argument patterns and CLI utilities
used across all command-line interfaces.
"""

import argparse
from typing import List, Optional


class McodeCLI:
    """Shared CLI utilities and argument patterns."""

    @staticmethod
    def add_core_args(parser: argparse.ArgumentParser) -> None:
        """Add core arguments used by all CLI scripts."""
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose (DEBUG) logging",
        )

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Set logging level (default: INFO)",
        )

        parser.add_argument(
            "--config", help="Path to configuration file (overrides default)"
        )

    @staticmethod
    def add_memory_args(parser: argparse.ArgumentParser) -> None:
        """Add core memory storage arguments."""
        parser.add_argument(
            "--ingest",
            action="store_true",
            help="Store results in CORE Memory",
        )

        parser.add_argument(
            "--memory-source",
            default="mcode_translator",
            help="Source identifier for CORE Memory storage (default: mcode_translator)",
        )


    @staticmethod
    def add_fetcher_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments specific to data fetching workflows."""
        parser.add_argument("-o", "--output", help="Output file path for fetched data")

        parser.add_argument(
            "--format",
            choices=["json", "jsonl"],
            default="json",
            help="Output format (default: json)",
        )

    @staticmethod
    def add_concurrency_args(parser: argparse.ArgumentParser) -> None:
        """Add unified concurrency arguments for all components."""
        concurrency_group = parser.add_argument_group("concurrency options")

        concurrency_group.add_argument(
            "--workers",
            type=int,
            default=0,
            help="Number of concurrent workers (0 = sequential, >0 = concurrent)",
        )

        concurrency_group.add_argument(
            "--worker-pool",
            choices=["fetcher", "processor", "optimizer", "custom"],
            default="custom",
            help="Use predefined worker pool sizes (fetcher=4, processor=8, optimizer=2)",
        )

        concurrency_group.add_argument(
            "--max-queue-size",
            type=int,
            default=1000,
            help="Maximum number of tasks in queue (default: 1000)",
        )

        concurrency_group.add_argument(
            "--task-timeout",
            type=float,
            help="Timeout for individual task execution (seconds)",
        )

    @staticmethod
    def add_processor_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments specific to data processing workflows."""
        parser.add_argument(
            "-m",
            "--model",
            default="deepseek-coder",
            help="LLM model to use for processing (default: deepseek-coder)",
        )

        parser.add_argument(
            "-p",
            "--prompt",
            default="direct_mcode_evidence_based_concise",
            help="Prompt template to use (default: direct_mcode_evidence_based_concise)",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,
            help="Batch size for processing (default: 10)",
        )

        # Add unified concurrency arguments
        McodeCLI.add_concurrency_args(parser)

    @staticmethod
    def add_optimizer_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments specific to optimization workflows."""
        parser.add_argument(
            "--max-combinations",
            type=int,
            help="Maximum number of prompt×model combinations to test",
        )

        parser.add_argument(
            "--save-config",
            help="Path to save optimal settings configuration file",
        )

    @staticmethod
    def setup_logging(args: argparse.Namespace) -> None:
        """Setup logging based on CLI arguments."""
        import logging

        from src.utils.logging_config import setup_logging

        log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
        setup_logging(level=log_level)

    @staticmethod
    def create_config(args: argparse.Namespace) -> "Config":
        """Create configuration instance from CLI arguments."""
        from src.utils.config import Config

        if args.config:
            # Custom config handling would go here
            pass

        return Config()
