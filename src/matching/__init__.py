"""
Triple-Engine Matching System for Patient-Trial Matching.

This package contains the implementations of the three matching engines:
- RegexRulesEngine: For deterministic, pattern-based matching.
- LLMMatchingEngine: for AI-driven, nuanced matching.
- CoreMemoryGraphEngine: For matching based on the CORE Memory knowledge graph.
"""

from . import tools
from .base import MatchingEngineBase
from .ensemble_decision_engine import EnsembleDecisionEngine
from .llm_engine import LLMMatchingEngine
from .memory_engine import CoreMemoryGraphEngine
from .regex_engine import RegexRulesEngine

__all__ = [
    "MatchingEngineBase",
    "RegexRulesEngine",
    "LLMMatchingEngine",
    "CoreMemoryGraphEngine",
    "EnsembleDecisionEngine",
    "tools",
]
