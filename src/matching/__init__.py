"""
Triple-Engine Matching System for Patient-Trial Matching.

This package contains the implementations of the three matching engines:
- RegexRulesEngine: For deterministic, pattern-based matching.
- LLMMatchingEngine: for AI-driven, nuanced matching.
- CoreMemoryGraphEngine: For matching based on the CORE Memory knowledge graph.
"""

from .base import MatchingEngineBase
from .llm_engine import LLMMatchingEngine
from .memory_engine import CoreMemoryGraphEngine
from .regex_engine import RegexRulesEngine
from . import tools

__all__ = [
    "MatchingEngineBase",
    "RegexRulesEngine",
    "LLMMatchingEngine",
    "CoreMemoryGraphEngine",
    "tools",
]