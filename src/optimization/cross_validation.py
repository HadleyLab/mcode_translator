"""
Cross Validation Module - Handles k-fold cross validation for optimization.

This module provides utilities for creating k-fold splits and managing
cross validation processes for mCODE translation optimization.
"""

from typing import List, Dict
from src.utils.logging_config import get_logger


class CrossValidator:
    """
    Handles cross validation operations for optimization workflows.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def create_kfold_splits(self, n_samples: int, n_folds: int) -> List[List[int]]:
        """
        Create k-fold cross validation splits.

        Args:
            n_samples: Total number of samples
            n_folds: Number of folds

        Returns:
            List of lists, where each inner list contains indices for that fold's validation set
        """
        indices = list(range(n_samples))
        fold_sizes = [n_samples // n_folds] * n_folds
        remainder = n_samples % n_folds

        # Distribute remainder across first few folds
        for i in range(remainder):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for size in fold_sizes:
            folds.append(indices[start : start + size])
            start += size

        return folds

    def generate_combinations(
        self, prompts: List[str], models: List[str], max_combinations: int
    ) -> List[Dict[str, str]]:
        """Generate combinations of prompts and models to test."""
        combinations = []

        for prompt in prompts:
            for model in models:
                if max_combinations > 0 and len(combinations) >= max_combinations:
                    break
                combinations.append({"prompt": prompt, "model": model})

        return combinations

    def validate_combination(
        self,
        prompt: str,
        model: str,
        available_prompts: List[str],
        available_models: List[str],
    ) -> bool:
        """
        Validate that a prompt×model combination is valid.

        Args:
            prompt: Prompt template name
            model: Model name
            available_prompts: List of available prompts
            available_models: List of available models

        Returns:
            bool: True if combination is valid
        """
        return prompt in available_prompts and model in available_models
