"""
Shared BenchmarkResult class to avoid circular imports.
This class contains the result structure for benchmark runs.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils import Loggable


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_variant_id: str = ""
    api_config_name: str = ""
    test_case_id: str = ""
    pipeline_type: str = ""  # Type of pipeline used (McodePipeline or NlpMcodePipeline)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    error_message: str = ""

    # Performance metrics - no defaults, will be calculated or remain None
    entities_extracted: Optional[int] = None
    entities_mapped: Optional[int] = None
    extraction_completeness: Optional[float] = None
    mapping_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    compliance_score: Optional[float] = None
    token_usage: Optional[int] = None

    # Raw results
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    mcode_mappings: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prompt_variant_id": self.prompt_variant_id,
            "api_config_name": self.api_config_name,
            "test_case_id": self.test_case_id,
            "pipeline_type": self.pipeline_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "entities_extracted": self.entities_extracted,
            "entities_mapped": self.entities_mapped,
            "extraction_completeness": self.extraction_completeness,
            "mapping_accuracy": self.mapping_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "compliance_score": self.compliance_score,
            "token_usage": self.token_usage,
            # Include raw results for validation recalculation
            "extracted_entities": self.extracted_entities,
            "mcode_mappings": self.mcode_mappings,
            "validation_results": self.validation_results,
        }

    def calculate_metrics(
        self,
        expected_entities: List[Dict[str, Any]] = None,
        expected_mappings: List[Dict[str, Any]] = None,
        framework: Loggable = None,
    ) -> None:
        """Calculate performance metrics based on results using gold standard validation - STRICT implementation"""
        if framework is None:
            raise ValueError("Framework parameter is required for metric calculation")

        framework.logger.debug("calculate_metrics method called")
        if not self.success:
            framework.logger.debug("self.success is False, returning early")
            return

        # Debug logging to understand why mapping validation is not being executed
        framework.logger.debug(
            f"calculate_metrics called with expected_entities={expected_entities is not None}, expected_mappings={expected_mappings is not None}"
        )
        if expected_entities is not None:
            framework.logger.debug(
                f"expected_entities length: {len(expected_entities)}"
            )
        if expected_mappings is not None:
            framework.logger.debug(
                f"expected_mappings length: {len(expected_mappings)}"
            )

        # Basic counts
        self.entities_extracted = len(self.extracted_entities)
        self.entities_mapped = len(self.mcode_mappings)

        # Calculate completeness (if expected entities provided)
        if expected_entities is not None:
            expected_count = len(expected_entities)
            if expected_count > 0:
                self.extraction_completeness = self.entities_extracted / expected_count

        # Use validation results from mCODE mapping
        if self.validation_results:
            self.compliance_score = self.validation_results.get(
                "compliance_score", None
            )

        # Calculate extraction metrics using text-based matching for NLP entities
        framework.logger.debug(
            f"Extraction validation: expected_entities={expected_entities is not None}, extracted_entities={self.extracted_entities is not None}, len={len(self.extracted_entities) if self.extracted_entities else 0}"
        )
        framework.logger.debug(
            f"Before extraction validation: expected_entities is None: {expected_entities is None}"
        )
        framework.logger.debug(
            f"Before extraction validation: self.extracted_entities is None: {self.extracted_entities is None}"
        )
        framework.logger.debug(
            f"Before extraction validation: expected_entities bool: {expected_entities is not None}"
        )
        framework.logger.debug(
            f"Before extraction validation: self.extracted_entities bool: {self.extracted_entities is not None}"
        )
        if expected_entities is not None and self.extracted_entities is not None:
            framework.logger.debug(
                f"Expected entities length: {len(expected_entities)}"
            )
            framework.logger.debug(
                f"Extracted entities length: {len(self.extracted_entities)}"
            )
            if len(expected_entities) > 0 and len(self.extracted_entities) > 0:
                framework.logger.debug(
                    f"Calculating extraction metrics with {len(self.extracted_entities)} extracted and {len(expected_entities)} expected entities"
                )
                # Use fuzzy text matching for extraction metrics to handle different text representations
                true_positives_ext, false_positives_ext, false_negatives_ext = (
                    BenchmarkResult._calculate_fuzzy_text_matches(
                        self.extracted_entities, expected_entities, framework
                    )
                )

                framework.logger.debug(
                    f"True positives: {true_positives_ext}, False positives: {false_positives_ext}, False negatives: {false_negatives_ext}"
                )

                self.precision = (
                    true_positives_ext / (true_positives_ext + false_positives_ext)
                    if (true_positives_ext + false_positives_ext) > 0
                    else None
                )
                self.recall = (
                    true_positives_ext / (true_positives_ext + false_negatives_ext)
                    if (true_positives_ext + false_negatives_ext) > 0
                    else None
                )
                self.f1_score = (
                    2 * (self.precision * self.recall) / (self.precision + self.recall)
                    if (
                        self.precision is not None
                        and self.recall is not None
                        and (self.precision + self.recall) > 0
                    )
                    else None
                )

                precision_str = (
                    f"{self.precision:.3f}" if self.precision is not None else "None"
                )
                recall_str = f"{self.recall:.3f}" if self.recall is not None else "None"
                f1_str = f"{self.f1_score:.3f}" if self.f1_score is not None else "None"
                framework.logger.debug(
                    f"Calculated metrics: precision={precision_str}, recall={recall_str}, f1={f1_str}"
                )
            else:
                # Handle case where one or both lists are empty
                framework.logger.debug(
                    f"One or both lists are empty: expected={len(expected_entities)}, extracted={len(self.extracted_entities)}"
                )
                self.precision = None
                self.recall = None
                self.f1_score = None
        else:
            # Handle case where one or both parameters are None
            framework.logger.debug(
                f"One or both parameters are None: expected_entities={expected_entities is not None}, extracted_entities={self.extracted_entities is not None}"
            )
            self.precision = None
            self.recall = None
            self.f1_score = None

        # Calculate mapping accuracy using mCODE-based matching for mCODE elements
        framework.logger.debug(
            f"Mapping validation: expected_mappings={expected_mappings is not None}, mcode_mappings={self.mcode_mappings is not None}, len={len(self.mcode_mappings) if self.mcode_mappings else 0}"
        )
        framework.logger.debug(
            f"Mapping validation: expected_mappings type={type(expected_mappings)}, mcode_mappings type={type(self.mcode_mappings)}"
        )
        framework.logger.debug(
            f"Mapping validation: expected_mappings value={expected_mappings}, mcode_mappings value={self.mcode_mappings}"
        )
        framework.logger.debug(
            f"Mapping validation: expected_mappings is None: {expected_mappings is None}"
        )
        framework.logger.debug(
            f"Mapping validation: self.mcode_mappings is None: {self.mcode_mappings is None}"
        )
        framework.logger.debug(
            f"Mapping validation: expected_mappings bool: {expected_mappings is not None}"
        )
        framework.logger.debug(
            f"Mapping validation: self.mcode_mappings bool: {self.mcode_mappings is not None}"
        )
        if expected_mappings is not None and self.mcode_mappings is not None:
            framework.logger.debug(
                f"Expected mappings length: {len(expected_mappings)}"
            )
            framework.logger.debug(f"mCODE mappings length: {len(self.mcode_mappings)}")
            if len(expected_mappings) > 0 and len(self.mcode_mappings) > 0:
                framework.logger.debug(
                    f"Calculating mapping metrics with {len(self.mcode_mappings)} mapped and {len(expected_mappings)} expected mappings"
                )
                # Use node-based matching for mCODE metrics (compare all fields including metadata)
                true_positives_map, false_positives_map, false_negatives_map = (
                    BenchmarkResult._calculate_mcode_matches(
                        self.mcode_mappings, expected_mappings, framework, debug=True
                    )
                )

                framework.logger.debug(f"Mapping validation debug info:")
                framework.logger.debug(f"  True positives: {true_positives_map}")
                framework.logger.debug(f"  False positives: {false_positives_map}")
                framework.logger.debug(f"  False negatives: {false_negatives_map}")

                mapping_precision = (
                    true_positives_map / (true_positives_map + false_positives_map)
                    if (true_positives_map + false_positives_map) > 0
                    else None
                )
                mapping_recall = (
                    true_positives_map / (true_positives_map + false_negatives_map)
                    if (true_positives_map + false_negatives_map) > 0
                    else None
                )

                self.mapping_accuracy = (
                    2
                    * (mapping_precision * mapping_recall)
                    / (mapping_precision + mapping_recall)
                    if (
                        mapping_precision is not None
                        and mapping_recall is not None
                        and (mapping_precision + mapping_recall) > 0
                    )
                    else None
                )

                mapping_precision_str = (
                    f"{mapping_precision:.3f}"
                    if mapping_precision is not None
                    else "None"
                )
                mapping_recall_str = (
                    f"{mapping_recall:.3f}" if mapping_recall is not None else "None"
                )
                mapping_accuracy_str = (
                    f"{self.mapping_accuracy:.3f}"
                    if self.mapping_accuracy is not None
                    else "None"
                )
                framework.logger.debug(
                    f"  Mapping accuracy calculation: precision={mapping_precision_str}, recall={mapping_recall_str}, accuracy={mapping_accuracy_str}"
                )
            else:
                # Handle case where one or both lists are empty
                framework.logger.debug(
                    f"One or both mapping lists are empty: expected={len(expected_mappings)}, mcode={len(self.mcode_mappings)}"
                )
                self.mapping_accuracy = None
        else:
            # Handle case where one or both parameters are None
            framework.logger.debug(
                f"One or both mapping parameters are None: expected_mappings={expected_mappings is not None}, mcode_mappings={self.mcode_mappings is not None}"
            )
            self.mapping_accuracy = None

        precision_final_str = (
            f"{self.precision:.3f}" if self.precision is not None else "None"
        )
        recall_final_str = f"{self.recall:.3f}" if self.recall is not None else "None"
        f1_final_str = f"{self.f1_score:.3f}" if self.f1_score is not None else "None"
        mapping_accuracy_final_str = (
            f"{self.mapping_accuracy:.3f}"
            if self.mapping_accuracy is not None
            else "None"
        )
        framework.logger.debug(
            f"Final metrics: precision={precision_final_str}, recall={recall_final_str}, f1={f1_final_str}, mapping_accuracy={mapping_accuracy_final_str}"
        )

    @staticmethod
    def _calculate_fuzzy_text_matches(
        extracted_entities: List[Dict[str, Any]],
        expected_entities: List[Dict[str, Any]],
        framework: Loggable = None,
        debug: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Calculate text matches using fuzzy matching to handle different text representations

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        extracted_texts = [entity.get("text", "") for entity in extracted_entities]
        expected_texts = [entity.get("text", "") for entity in expected_entities]

        # Track matches
        matched_extracted = set()
        matched_expected = set()

        # First pass: exact matches
        for i, extracted_text in enumerate(extracted_texts):
            for j, expected_text in enumerate(expected_texts):
                if extracted_text == expected_text:
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(
                            f"   ✅ Exact match: '{extracted_text}' -> '{expected_text}'"
                        )

        # Second pass: fuzzy matches for remaining entities
        for i, extracted_text in enumerate(extracted_texts):
            if i in matched_extracted:
                continue

            for j, expected_text in enumerate(expected_texts):
                if j in matched_expected:
                    continue

                # Check if extracted text contains expected text (partial match)
                if expected_text.lower() in extracted_text.lower():
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(
                            f"   ✅ Partial match: '{extracted_text}' contains '{expected_text}'"
                        )
                    continue

                # Check if expected text contains extracted text (partial match)
                if extracted_text.lower() in expected_text.lower():
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(
                            f"   ✅ Partial match: '{expected_text}' contains '{extracted_text}'"
                        )
                    continue

                # Check for combined entities (e.g., "Pregnancy or breastfeeding" should match both)
                combined_match = False
                if " or " in extracted_text.lower():
                    parts = [
                        part.strip() for part in extracted_text.lower().split(" or ")
                    ]
                    if expected_text.lower() in parts:
                        matched_extracted.add(i)
                        matched_expected.add(j)
                        combined_match = True
                        if debug and framework:
                            framework.logger.warning(
                                f"   ✅ Combined entity match: '{extracted_text}' contains '{expected_text}'"
                            )

                if not combined_match and " or " in expected_text.lower():
                    parts = [
                        part.strip() for part in expected_text.lower().split(" or ")
                    ]
                    if extracted_text.lower() in parts:
                        matched_extracted.add(i)
                        matched_expected.add(j)
                        if debug and framework:
                            framework.logger.warning(
                                f"   ✅ Combined entity match: '{expected_text}' contains '{extracted_text}'"
                            )

        # Calculate metrics
        true_positives = len(matched_expected)
        false_positives = len(extracted_entities) - len(matched_extracted)
        false_negatives = len(expected_entities) - len(matched_expected)

        if debug and framework:
            framework.logger.warning(f"   Total extracted: {len(extracted_entities)}")
            framework.logger.warning(f"   Total expected: {len(expected_entities)}")
            framework.logger.warning(f"   Matched extracted: {len(matched_extracted)}")
            framework.logger.warning(f"   Matched expected: {len(matched_expected)}")
            framework.logger.warning(f"   Unmatched extracted: {false_positives}")
            framework.logger.warning(f"   Unmatched expected: {false_negatives}")

            # Log some unmatched examples for debugging
            unmatched_extracted = [
                extracted_texts[i]
                for i in range(len(extracted_texts))
                if i not in matched_extracted
            ]
            unmatched_expected = [
                expected_texts[j]
                for j in range(len(expected_texts))
                if j not in matched_expected
            ]

            if unmatched_extracted:
                framework.logger.warning(
                    f"   Unmatched extracted examples: {unmatched_extracted[:3]}"
                )
            if unmatched_expected:
                framework.logger.warning(
                    f"   Unmatched expected examples: {unmatched_expected[:3]}"
                )

        return true_positives, false_positives, false_negatives

    @staticmethod
    def _calculate_mcode_matches(
        actual_mappings: List[Dict[str, Any]],
        expected_mappings: List[Dict[str, Any]],
        framework: Loggable = None,
        debug: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Calculate mCODE matches using node-based matching to handle clinical concept comparison
        Compares all fields including mcode_element, value, source_entity_index, confidence, and mapping_rationale

        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # Track matches
        matched_actual = set()
        matched_expected = set()

        # First pass: exact matches on all fields
        for i, actual_mapping in enumerate(actual_mappings):
            for j, expected_mapping in enumerate(expected_mappings):
                if j in matched_expected:
                    continue

                # Compare all relevant fields for exact match
                actual_element = actual_mapping.get("mcode_element", "").lower()
                expected_element = expected_mapping.get("mcode_element", "").lower()
                actual_value_raw = actual_mapping.get("value", "")
                expected_value_raw = expected_mapping.get("value", "")

                actual_value = (
                    json.dumps(actual_value_raw)
                    if isinstance(actual_value_raw, dict)
                    else str(actual_value_raw).lower()
                )
                expected_value = (
                    json.dumps(expected_value_raw)
                    if isinstance(expected_value_raw, dict)
                    else str(expected_value_raw).lower()
                )
                actual_source_idx = actual_mapping.get("source_entity_index", -1)
                expected_source_idx = expected_mapping.get("source_entity_index", -1)
                actual_confidence = actual_mapping.get("confidence", 0.0)
                expected_confidence = expected_mapping.get("confidence", 0.0)
                actual_rationale = actual_mapping.get("mapping_rationale", "").lower()
                expected_rationale = expected_mapping.get(
                    "mapping_rationale", ""
                ).lower()

                # Exact match on all fields
                if (
                    actual_element == expected_element
                    and actual_value == expected_value
                    and actual_source_idx == expected_source_idx
                    and abs(actual_confidence - expected_confidence) < 0.01
                    and actual_rationale == expected_rationale
                ):

                    matched_actual.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(
                            f"   ✅ Exact mCODE match: '{actual_element}'='{actual_value}' -> '{expected_element}'='{expected_value}'"
                        )
                    continue

        # Second pass: relaxed matching for clinical concepts (same mcode_element + value)
        for i, actual_mapping in enumerate(actual_mappings):
            if i in matched_actual:
                continue

            for j, expected_mapping in enumerate(expected_mappings):
                if j in matched_expected:
                    continue

                actual_element = actual_mapping.get("mcode_element", "").lower()
                expected_element = expected_mapping.get("mcode_element", "").lower()
                actual_value_raw = actual_mapping.get("value", "")
                expected_value_raw = expected_mapping.get("value", "")

                actual_value = (
                    json.dumps(actual_value_raw)
                    if isinstance(actual_value_raw, dict)
                    else str(actual_value_raw).lower()
                )
                expected_value = (
                    json.dumps(expected_value_raw)
                    if isinstance(expected_value_raw, dict)
                    else str(expected_value_raw).lower()
                )

                # Match on mcode_element and value (clinical concept match)
                if (
                    actual_element == expected_element
                    and actual_value == expected_value
                ):
                    matched_actual.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(
                            f"   ✅ Clinical concept match: '{actual_element}'='{actual_value}' -> '{expected_element}'='{expected_value}'"
                        )
                    continue

        # Calculate metrics
        true_positives = len(matched_expected)
        false_positives = len(actual_mappings) - len(matched_actual)
        false_negatives = len(expected_mappings) - len(matched_expected)

        if debug and framework:
            framework.logger.warning(
                f"   Total actual mappings: {len(actual_mappings)}"
            )
            framework.logger.warning(
                f"   Total expected mappings: {len(expected_mappings)}"
            )
            framework.logger.warning(f"   Matched actual: {len(matched_actual)}")
            framework.logger.warning(f"   Matched expected: {len(matched_expected)}")
            framework.logger.warning(f"   Unmatched actual: {false_positives}")
            framework.logger.warning(f"   Unmatched expected: {false_negatives}")

            # Log some unmatched examples for debugging
            unmatched_actual = [
                f"{m.get('mcode_element', '')}={m.get('value', '')}"
                for idx, m in enumerate(actual_mappings)
                if idx not in matched_actual
            ]
            unmatched_expected = [
                f"{m.get('mcode_element', '')}={m.get('value', '')}"
                for idx, m in enumerate(expected_mappings)
                if idx not in matched_expected
            ]

            if unmatched_actual:
                framework.logger.warning(
                    f"   Unmatched actual examples: {unmatched_actual[:3]}"
                )
            if unmatched_expected:
                framework.logger.warning(
                    f"   Unmatched expected examples: {unmatched_expected[:3]}"
                )

        return true_positives, false_positives, false_negatives
