"""
Unit tests for feature_utils module.
"""

import pytest
from src.utils.feature_utils import (
    standardize_features,
    standardize_biomarkers,
    standardize_variants,
)


class TestStandardizeFeatures:
    """Test standardize_features function."""

    def test_standardize_features_complete(self):
        """Test standardizing features with complete data."""
        features = [
            {
                "type": "gene",
                "value": "BRCA1",
                "confidence": 0.95,
                "source": "extractor",
                "normalized": "BRCA1",
            }
        ]

        result = standardize_features(features)

        expected = [
            {
                "type": "gene",
                "value": "BRCA1",
                "confidence": 0.95,
                "source": "extractor",
                "normalized": "BRCA1",
            }
        ]

        assert result == expected

    def test_standardize_features_minimal(self):
        """Test standardizing features with minimal data."""
        features = [{"type": "protein"}]

        result = standardize_features(features)

        expected = [
            {
                "type": "protein",
                "value": "",
                "confidence": 0.0,
                "source": "nlp",
                "normalized": "",
            }
        ]

        assert result == expected

    def test_standardize_features_empty(self):
        """Test standardizing empty features list."""
        result = standardize_features([])
        assert result == []

    def test_standardize_features_missing_fields(self):
        """Test standardizing features with missing fields."""
        features = [{"value": "test"}]

        result = standardize_features(features)

        expected = [
            {
                "type": "unknown",
                "value": "test",
                "confidence": 0.0,
                "source": "nlp",
                "normalized": "test",
            }
        ]

        assert result == expected

    def test_standardize_features_normalized_present(self):
        """Test normalized field when explicitly set."""
        features = [{"value": "BRCA1", "normalized": ""}]

        result = standardize_features(features)

        assert result[0]["normalized"] == ""  # normalized is explicitly set to empty

    def test_standardize_features_normalized_fallback(self):
        """Test normalized field fallback to value when not present."""
        features = [{"value": "BRCA1"}]  # normalized not present

        result = standardize_features(features)

        assert result[0]["normalized"] == "BRCA1"


class TestStandardizeBiomarkers:
    """Test standardize_biomarkers function."""

    def test_standardize_biomarkers_complete(self):
        """Test standardizing biomarkers with complete data."""
        biomarkers = [
            {
                "name": "HER2",
                "value": "positive",
                "type": "receptor",
                "status": "overexpressed",
                "confidence": 0.88,
                "source": "pathology",
            }
        ]

        result = standardize_biomarkers(biomarkers)

        expected = [
            {
                "name": "HER2",
                "value": "positive",
                "type": "receptor",
                "status": "overexpressed",
                "confidence": 0.88,
                "source": "pathology",
            }
        ]

        assert result == expected

    def test_standardize_biomarkers_minimal(self):
        """Test standardizing biomarkers with minimal data."""
        biomarkers = [{"name": "TP53"}]

        result = standardize_biomarkers(biomarkers)

        expected = [
            {
                "name": "TP53",
                "value": "",
                "type": "biomarker",
                "status": "unknown",
                "confidence": 0.0,
                "source": "nlp",
            }
        ]

        assert result == expected

    def test_standardize_biomarkers_empty(self):
        """Test standardizing empty biomarkers list."""
        result = standardize_biomarkers([])
        assert result == []

    def test_standardize_biomarkers_missing_fields(self):
        """Test standardizing biomarkers with missing fields."""
        biomarkers = [{"value": "negative"}]

        result = standardize_biomarkers(biomarkers)

        expected = [
            {
                "name": "",
                "value": "negative",
                "type": "biomarker",
                "status": "unknown",
                "confidence": 0.0,
                "source": "nlp",
            }
        ]

        assert result == expected


class TestStandardizeVariants:
    """Test standardize_variants function."""

    def test_standardize_variants_complete(self):
        """Test standardizing variants with complete data."""
        variants = [
            {
                "gene": "BRCA1",
                "mutation": "c.68_69delAG",
                "type": "deletion",
                "classification": "pathogenic",
                "confidence": 0.92,
                "source": "sequencing",
            }
        ]

        result = standardize_variants(variants)

        expected = [
            {
                "gene": "BRCA1",
                "mutation": "c.68_69delAG",
                "type": "deletion",
                "classification": "pathogenic",
                "confidence": 0.92,
                "source": "sequencing",
            }
        ]

        assert result == expected

    def test_standardize_variants_minimal(self):
        """Test standardizing variants with minimal data."""
        variants = [{"gene": "EGFR"}]

        result = standardize_variants(variants)

        expected = [
            {
                "gene": "EGFR",
                "mutation": "",
                "type": "variant",
                "classification": "unknown",
                "confidence": 0.0,
                "source": "nlp",
            }
        ]

        assert result == expected

    def test_standardize_variants_empty(self):
        """Test standardizing empty variants list."""
        result = standardize_variants([])
        assert result == []

    def test_standardize_variants_missing_fields(self):
        """Test standardizing variants with missing fields."""
        variants = [{"mutation": "p.Val600Glu"}]

        result = standardize_variants(variants)

        expected = [
            {
                "gene": "",
                "mutation": "p.Val600Glu",
                "type": "variant",
                "classification": "unknown",
                "confidence": 0.0,
                "source": "nlp",
            }
        ]

        assert result == expected


class TestIntegration:
    """Test integration scenarios."""

    def test_all_functions_empty_lists(self):
        """Test all functions with empty lists."""
        assert standardize_features([]) == []
        assert standardize_biomarkers([]) == []
        assert standardize_variants([]) == []

    def test_standardize_features_multiple_items(self):
        """Test standardizing multiple features."""
        features = [
            {"type": "gene", "value": "BRCA1"},
            {"type": "protein", "value": "HER2", "confidence": 0.8},
        ]

        result = standardize_features(features)

        assert len(result) == 2
        assert result[0]["type"] == "gene"
        assert result[0]["value"] == "BRCA1"
        assert result[1]["type"] == "protein"
        assert result[1]["value"] == "HER2"
        assert result[1]["confidence"] == 0.8

    def test_standardize_biomarkers_multiple_items(self):
        """Test standardizing multiple biomarkers."""
        biomarkers = [
            {"name": "HER2", "status": "positive"},
            {"name": "ER", "status": "negative", "confidence": 0.9},
        ]

        result = standardize_biomarkers(biomarkers)

        assert len(result) == 2
        assert result[0]["name"] == "HER2"
        assert result[0]["status"] == "positive"
        assert result[1]["name"] == "ER"
        assert result[1]["status"] == "negative"
        assert result[1]["confidence"] == 0.9

    def test_standardize_variants_multiple_items(self):
        """Test standardizing multiple variants."""
        variants = [
            {"gene": "BRCA1", "classification": "pathogenic"},
            {"gene": "TP53", "mutation": "R175H", "confidence": 0.85},
        ]

        result = standardize_variants(variants)

        assert len(result) == 2
        assert result[0]["gene"] == "BRCA1"
        assert result[0]["classification"] == "pathogenic"
        assert result[1]["gene"] == "TP53"
        assert result[1]["mutation"] == "R175H"
        assert result[1]["confidence"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])
