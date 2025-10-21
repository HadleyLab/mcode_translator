"""
Ensemble processing module for unified ensemble architecture.

This module provides a unified ensemble architecture that can be reused
for both matching and trials processing, with shared consensus methods,
expert weight management, and common data structures.
"""

from .base_ensemble_engine import BaseEnsembleEngine

__all__ = ["BaseEnsembleEngine"]
