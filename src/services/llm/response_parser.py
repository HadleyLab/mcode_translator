"""
LLM Response Parser - Parse LLM responses into structured mCODE models.

This module provides functionality to parse LLM-generated responses into
validated mCODE ontology models with proper type safety and validation.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.shared.models import (
    # mCODE profile models
    CancerConditionCode,
    ECOGPerformanceStatus,
    McodeElement,
    PipelineResult,
    ProcessingMetadata,
    ReceptorStatus,
    SourceReference,
    TNMStageGroup,
    ValidationResult,
)
from src.utils.logging_config import get_logger


class LLMResponseParser:
    """
    Parser for converting LLM responses into structured mCODE models.

    Handles various response formats (JSON, markdown-wrapped JSON) and
    provides validation and type safety for mCODE elements.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def parse_response(self, raw_response: str, expected_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse LLM response into structured data.

        Args:
            raw_response: Raw response from LLM
            expected_format: Expected format hint (e.g., 'mcode_elements', 'validation_result')

        Returns:
            Parsed and validated response data
        """
        # Clean the response
        cleaned_response = self._clean_response(raw_response)

        # Parse JSON
        parsed_data = self._parse_json(cleaned_response)

        # Validate and structure based on expected format
        if expected_format == 'mcode_elements':
            return self._parse_mcode_elements(parsed_data)
        elif expected_format == 'validation_result':
            return self._parse_validation_result(parsed_data)
        else:
            # Generic parsing
            return parsed_data

    def _clean_response(self, response: str) -> str:
        """Clean and normalize LLM response."""
        # Remove markdown code blocks
        response = re.sub(r'```\w*\n?', '', response)
        response = re.sub(r'```\n?', '', response)

        # Remove common prefixes/suffixes
        response = response.strip()
        response = re.sub(r'^(Here is|Here are|The following|Response:)\s*', '', response, flags=re.IGNORECASE)

        return response

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response string."""
        return json.loads(response)

    def _parse_mcode_elements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse mCODE elements from LLM response."""
        elements = []

        # Handle different response formats
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    element = self._create_mcode_element(item)
                    if element:
                        elements.append(element)
        elif isinstance(data, dict):
            # Check for various keys that might contain elements
            for key in ['mcode_elements', 'elements', 'results', 'data']:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            element = self._create_mcode_element(item)
                            if element:
                                elements.append(element)
                    break
            else:
                # Try to parse the dict itself as an element
                element = self._create_mcode_element(data)
                if element:
                    elements.append(element)

        return {
            "mcode_elements": elements,
            "element_count": len(elements),
            "parsed_successfully": True
        }

    def _create_mcode_element(self, data: Dict[str, Any]) -> Optional[McodeElement]:
        """Create a validated McodeElement from parsed data."""
        # Map common field variations
        element_type = (
            data.get('element_type') or
            data.get('type') or
            data.get('mcode_element') or
            data.get('element_name')
        )

        if not element_type:
            return None

        # Create McodeElement with validation
        element = McodeElement(
            element_type=element_type,
            code=data.get('code'),
            display=data.get('display') or data.get('value'),
            system=data.get('system'),
            confidence_score=data.get('confidence_score') or data.get('confidence'),
            evidence_text=data.get('evidence_text') or data.get('evidence')
        )

        return element

    def _parse_validation_result(self, data: Dict[str, Any]) -> ValidationResult:
        """Parse validation result from LLM response."""
        return ValidationResult(
            compliance_score=data.get('compliance_score', 0.0),
            validation_errors=data.get('validation_errors', []),
            validation_warnings=data.get('validation_warnings', []),
            required_elements_present=data.get('required_elements_present', []),
            missing_elements=data.get('missing_elements', [])
        )

    def create_pipeline_result(
        self,
        raw_response: str,
        processing_metadata: Optional[ProcessingMetadata] = None,
        source_references: Optional[List[SourceReference]] = None
    ) -> PipelineResult:
        """
        Create a complete PipelineResult from LLM response.

        Args:
            raw_response: Raw LLM response
            processing_metadata: Optional processing metadata
            source_references: Optional source references

        Returns:
            Complete PipelineResult with parsed mCODE elements
        """
        # Parse the response
        parsed_data = self.parse_response(raw_response, 'mcode_elements')

        # Extract elements
        mcode_elements = parsed_data.get('mcode_elements', [])

        # Create validation result (basic for now)
        validation_result = ValidationResult(
            compliance_score=0.8 if mcode_elements else 0.0,  # Basic scoring
            validation_errors=[],
            validation_warnings=[],
            required_elements_present=[elem.element_type for elem in mcode_elements],
            missing_elements=[]
        )

        # Ensure metadata exists
        if processing_metadata is None:
            processing_metadata = ProcessingMetadata(
                engine_type="llm_parser",
                entities_count=len(mcode_elements),
                mapped_count=len(mcode_elements)
            )

        return PipelineResult(
            extracted_entities=[],  # Not used in this context
            mcode_mappings=mcode_elements,
            source_references=source_references or [],
            validation_results=validation_result,
            metadata=processing_metadata,
            original_data={"raw_response": raw_response},
            error=None
        )

    def validate_mcode_element(self, element: McodeElement) -> Dict[str, Any]:
        """
        Validate a single mCODE element against known profiles.

        Args:
            element: McodeElement to validate

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Basic validation rules
        if not element.element_type:
            issues.append("Missing element_type")
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Validate specific element types
        if element.element_type == "CancerCondition":
            if element.code and element.code not in [c.value for c in CancerConditionCode]:
                warnings.append(f"Unknown cancer condition code: {element.code}")

        elif element.element_type in ["ERReceptorStatus", "PRReceptorStatus", "HER2ReceptorStatus"]:
            if element.code and element.code not in [r.value for r in ReceptorStatus]:
                warnings.append(f"Unknown receptor status: {element.code}")

        elif element.element_type == "ECOGPerformanceStatus":
            if element.code and element.code not in [e.value for e in ECOGPerformanceStatus]:
                warnings.append(f"Unknown ECOG status: {element.code}")

        elif element.element_type == "TNMStageGroup":
            if element.code and element.code not in [t.value for t in TNMStageGroup]:
                warnings.append(f"Unknown TNM stage: {element.code}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }
