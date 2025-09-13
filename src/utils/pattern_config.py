"""Configuration of regex patterns for clinical text processing.

Loads patterns from centralized modular configuration for better maintainability.
"""

import re
from typing import Any, Dict

from .config import Config


class PatternManager:
    """Manages regex patterns loaded from centralized configuration."""

    def __init__(self):
        self.config = Config()
        self.patterns_config = self.config.get_patterns_config()
        self._compiled_patterns = {}

    def _compile_pattern(self, pattern_config: Dict[str, Any]) -> re.Pattern:
        """Compile a regex pattern from configuration."""
        pattern = pattern_config["pattern"]
        flags = 0

        if pattern_config.get("flags"):
            flag_str = pattern_config["flags"]
            if "IGNORECASE" in flag_str:
                flags |= re.IGNORECASE
            if "MULTILINE" in flag_str:
                flags |= re.MULTILINE
            if "DOTALL" in flag_str:
                flags |= re.DOTALL

        return re.compile(pattern, flags)

    def get_biomarker_patterns(self) -> Dict[str, re.Pattern]:
        """Get compiled biomarker regex patterns."""
        if "biomarker" not in self._compiled_patterns:
            patterns = {}
            biomarker_config = self.patterns_config["patterns"]["biomarker_patterns"]
            for name, config in biomarker_config.items():
                patterns[name] = self._compile_pattern(config)
            self._compiled_patterns["biomarker"] = patterns

        return self._compiled_patterns["biomarker"]

    def get_genomic_patterns(self) -> Dict[str, re.Pattern]:
        """Get compiled genomic variant patterns."""
        if "genomic" not in self._compiled_patterns:
            patterns = {}
            genomic_config = self.patterns_config["patterns"]["genomic_patterns"]
            for name, config in genomic_config.items():
                patterns[name] = self._compile_pattern(config)
            self._compiled_patterns["genomic"] = patterns

        return self._compiled_patterns["genomic"]

    def get_condition_patterns(self) -> Dict[str, re.Pattern]:
        """Get compiled condition patterns."""
        if "condition" not in self._compiled_patterns:
            patterns = {}
            condition_config = self.patterns_config["patterns"]["condition_patterns"]
            for name, config in condition_config.items():
                patterns[name] = self._compile_pattern(config)
            self._compiled_patterns["condition"] = patterns

        return self._compiled_patterns["condition"]

    def get_demographic_patterns(self) -> Dict[str, re.Pattern]:
        """Get compiled demographic patterns."""
        if "demographic" not in self._compiled_patterns:
            patterns = {}
            demographic_config = self.patterns_config["patterns"][
                "demographic_patterns"
            ]
            for name, config in demographic_config.items():
                patterns[name] = self._compile_pattern(config)
            self._compiled_patterns["demographic"] = patterns

        return self._compiled_patterns["demographic"]

    def get_all_patterns(self) -> Dict[str, Dict[str, re.Pattern]]:
        """Get all compiled patterns organized by category."""
        return {
            "biomarker": self.get_biomarker_patterns(),
            "genomic": self.get_genomic_patterns(),
            "condition": self.get_condition_patterns(),
            "demographic": self.get_demographic_patterns(),
        }


# Global pattern manager instance
pattern_manager = PatternManager()

# Legacy compatibility - expose commonly used patterns
BIOMARKER_PATTERNS = pattern_manager.get_biomarker_patterns()
GENE_PATTERN = pattern_manager.get_genomic_patterns().get("gene_pattern")
VARIANT_PATTERN = pattern_manager.get_genomic_patterns().get("variant_pattern")
COMPLEX_VARIANT_PATTERN = pattern_manager.get_genomic_patterns().get(
    "complex_variant_pattern"
)
STAGE_PATTERN = pattern_manager.get_condition_patterns().get("stage_pattern")
CANCER_TYPE_PATTERN = pattern_manager.get_condition_patterns().get(
    "cancer_type_pattern"
)
CONDITION_PATTERN = pattern_manager.get_condition_patterns().get("condition_pattern")
ECOG_PATTERN = pattern_manager.get_demographic_patterns().get("ecog_pattern")
GENDER_PATTERN = pattern_manager.get_demographic_patterns().get("gender_pattern")
AGE_PATTERN = pattern_manager.get_demographic_patterns().get("age_pattern")
