"""
Services package for mCODE translator with engine-based organization.

This package contains specialized service classes organized by processing engine:
- LLM engine services for AI-powered processing
- Regex engine services for pattern-based processing
- Shared services for engine-agnostic functionality
"""

# LLM Engine Services
from .llm.engine import LLMEngine

# Regex Engine Services
from .regex.engine import RegexEngine
from .regex.pattern_manager import PatternManager

# Shared/Engine-Agnostic Services
from .clinical_note_generator import ClinicalNoteGenerator
from .demographics_extractor import DemographicsExtractor
from .fhir_extractors import FHIRResourceExtractors
from .processing_service import ProcessingService, ProcessingEngine
from .summarizer import McodeSummarizer
from .trial_processor import (
    McodeTrialProcessor,
    RegexTrialProcessor,
    LLMTrialProcessor,
    BaseTrialProcessor,
    TrialProcessor,
    ProcessingResult
)

__all__ = [
    # LLM Engine
    "LLMEngine",

    # Regex Engine
    "RegexEngine",
    "PatternManager",

    # Shared Services
    "ClinicalNoteGenerator",
    "DemographicsExtractor",
    "FHIRResourceExtractors",
    "ProcessingService",
    "ProcessingEngine",
    "McodeSummarizer",

    # Trial Processor Framework
    "McodeTrialProcessor",
    "RegexTrialProcessor",
    "LLMTrialProcessor",
    "BaseTrialProcessor",
    "TrialProcessor",
    "ProcessingResult",
]
