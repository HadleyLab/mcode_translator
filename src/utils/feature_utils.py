"""
Feature Utilities for mCODE Translator

This module provides functions for standardizing the structure of NLP feature
extraction results, including biomarkers, features, and variants.
"""

from typing import Any, Dict, List, Optional


def standardize_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize feature extraction results.

    Args:
        features: List of feature dictionaries

    Returns:
        Standardized list of features
    """
    standardized = []
    for feature in features:
        standardized.append({
            "type": feature.get("type", "unknown"),
            "value": feature.get("value", ""),
            "confidence": feature.get("confidence", 0.0),
            "source": feature.get("source", "nlp"),
            "normalized": feature.get("normalized", feature.get("value", ""))
        })
    return standardized


def standardize_biomarkers(biomarkers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize biomarker extraction results.

    Args:
        biomarkers: List of biomarker dictionaries

    Returns:
        Standardized list of biomarkers
    """
    standardized = []
    for biomarker in biomarkers:
        standardized.append({
            "name": biomarker.get("name", ""),
            "value": biomarker.get("value", ""),
            "type": biomarker.get("type", "biomarker"),
            "status": biomarker.get("status", "unknown"),
            "confidence": biomarker.get("confidence", 0.0),
            "source": biomarker.get("source", "nlp")
        })
    return standardized


def standardize_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize genetic variant extraction results.

    Args:
        variants: List of variant dictionaries

    Returns:
        Standardized list of variants
    """
    standardized = []
    for variant in variants:
        standardized.append({
            "gene": variant.get("gene", ""),
            "mutation": variant.get("mutation", ""),
            "type": variant.get("type", "variant"),
            "classification": variant.get("classification", "unknown"),
            "confidence": variant.get("confidence", 0.0),
            "source": variant.get("source", "nlp")
        })
    return standardized