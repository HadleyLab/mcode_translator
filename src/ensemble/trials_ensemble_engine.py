"""
TrialsEnsembleEngine - mCODE-specific ensemble engine for clinical trials.

Extends BaseEnsembleEngine to handle mCODE extraction and validation from clinical trial data.
Provides specialized expert types for trial processing, mCODE validation, and evidence analysis.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from shared.models import McodeElement, McodeValidator, ValidationResult
from utils.config import Config
from utils.logging_config import get_logger

from .base_ensemble_engine import (
    BaseEnsembleEngine,
    ConfidenceCalibration,
    ConsensusMethod,
    EnsembleResult,
    ExpertWeight,
)


@dataclass
class TrialsExpertAssessment:
    """Assessment from a trials-specific expert."""
    expert_type: str
    mcode_elements: List[McodeElement]
    confidence_score: float
    validation_results: ValidationResult
    reasoning: str
    evidence_quality: float
    clinical_accuracy: float


class TrialsEnsembleEngine(BaseEnsembleEngine):
    """
    Ensemble engine specialized for mCODE extraction from clinical trials.

    Extends BaseEnsembleEngine with trials-specific expert types and processing logic:
    - mcode_extractor: Extracts mCODE elements from trial eligibility criteria and descriptions
    - clinical_validator: Validates clinical accuracy and medical appropriateness
    - evidence_analyzer: Analyzes evidence quality and supporting documentation
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        config: Optional[Config] = None,
        consensus_method: Optional[ConsensusMethod] = None,
        confidence_calibration: Optional[ConfidenceCalibration] = None,
        enable_rule_based_integration: bool = True,
        enable_dynamic_weighting: bool = True,
        min_experts: int = 2,
        max_experts: int = 3,
        cache_enabled: bool = True,
        max_retries: int = 3,
        mcode_validator: Optional[McodeValidator] = None
    ) -> None:
        """
        Initialize the trials ensemble engine.

        Args:
            model_name: LLM model to use for expert assessments
            config: Configuration instance
            consensus_method: Method for combining expert opinions
            confidence_calibration: Method for calibrating confidence scores
            enable_rule_based_integration: Whether to integrate rule-based scoring
            enable_dynamic_weighting: Whether to use dynamic expert weighting
            min_experts: Minimum number of experts to use
            max_experts: Maximum number of experts to use
            cache_enabled: Whether to enable caching
            max_retries: Maximum number of retries on failure
            mcode_validator: mCODE validator instance for validation logic
        """
        super().__init__(
            model_name=model_name,
            config=config,
            consensus_method=consensus_method or ConsensusMethod.DYNAMIC_WEIGHTING,
            confidence_calibration=confidence_calibration or ConfidenceCalibration.ISOTONIC_REGRESSION,
            enable_rule_based_integration=enable_rule_based_integration,
            enable_dynamic_weighting=enable_dynamic_weighting,
            min_experts=min_experts,
            max_experts=max_experts,
            cache_enabled=cache_enabled,
            max_retries=max_retries
        )

        self.mcode_validator = mcode_validator or McodeValidator()
        self.logger = get_logger(__name__)

        # Override expert weights with trials-specific experts
        self._initialize_trials_expert_weights()

        self.logger.info(
            "✅ TrialsEnsembleEngine initialized with mCODE-specific experts: "
            "mcode_extractor, clinical_validator, evidence_analyzer, tnm_staging_specialist, "
            "clinical_terminologist, treatment_regimen_specialist, biomarker_specialist"
        )

    def _initialize_trials_expert_weights(self) -> None:
        """Initialize expert weights for trials-specific processing."""
        current_time = time.time()

        self.expert_weights.update({
            "mcode_extractor": ExpertWeight(
                expert_type="mcode_extractor",
                base_weight=1.2,
                reliability_score=0.88,
                specialization_bonus=0.15,
                historical_accuracy=0.85,
                last_updated=current_time
            ),
            "clinical_validator": ExpertWeight(
                expert_type="clinical_validator",
                base_weight=1.1,
                reliability_score=0.92,
                specialization_bonus=0.12,
                historical_accuracy=0.89,
                last_updated=current_time
            ),
            "evidence_analyzer": ExpertWeight(
                expert_type="evidence_analyzer",
                base_weight=0.95,
                reliability_score=0.85,
                specialization_bonus=0.18,
                historical_accuracy=0.82,
                last_updated=current_time
            ),
            "tnm_staging_specialist": ExpertWeight(
                expert_type="tnm_staging_specialist",
                base_weight=1.25,
                reliability_score=0.91,
                specialization_bonus=0.20,
                historical_accuracy=0.87,
                last_updated=current_time
            ),
            "clinical_terminologist": ExpertWeight(
                expert_type="clinical_terminologist",
                base_weight=1.15,
                reliability_score=0.94,
                specialization_bonus=0.22,
                historical_accuracy=0.90,
                last_updated=current_time
            ),
            "treatment_regimen_specialist": ExpertWeight(
                expert_type="treatment_regimen_specialist",
                base_weight=1.05,
                reliability_score=0.89,
                specialization_bonus=0.16,
                historical_accuracy=0.84,
                last_updated=current_time
            ),
            "biomarker_specialist": ExpertWeight(
                expert_type="biomarker_specialist",
                base_weight=1.10,
                reliability_score=0.86,
                specialization_bonus=0.19,
                historical_accuracy=0.81,
                last_updated=current_time
            )
        })

    async def process_ensemble(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> EnsembleResult:
        """
        Process ensemble decision for clinical trial mCODE extraction.

        Args:
            input_data: Trial data containing eligibility criteria, conditions, etc.
            criteria_data: mCODE extraction criteria and validation rules

        Returns:
            EnsembleResult with mCODE extraction results and validation
        """
        start_time = time.time()

        try:
            # Get expert assessments
            expert_assessments = await self._get_expert_assessments(input_data, criteria_data)

            # Get rule-based score for mCODE compliance
            rule_based_score = await self._get_rule_based_score(input_data, criteria_data)

            # Apply consensus method
            ensemble_decision = self._apply_consensus_method(
                expert_assessments, rule_based_score, input_data, criteria_data
            )

            # Calculate processing metadata
            processing_metadata = {
                "engine_type": "trials_ensemble",
                "trial_id": input_data.get("trial_id", "unknown"),
                "processing_time_seconds": time.time() - start_time,
                "experts_used": len(expert_assessments),
                "mcode_elements_extracted": len(ensemble_decision.get("mcode_elements", [])),
                "validation_score": ensemble_decision.get("validation_score", 0.0)
            }

            # Create final result
            result = EnsembleResult(
                is_match=ensemble_decision["is_match"],
                confidence_score=ensemble_decision["confidence_score"],
                consensus_method=self.consensus_method.value,
                expert_assessments=expert_assessments,
                individual_decisions=ensemble_decision["individual_decisions"],
                reasoning=ensemble_decision["reasoning"],
                matched_criteria=ensemble_decision["matched_criteria"],
                unmatched_criteria=ensemble_decision["unmatched_criteria"],
                clinical_notes=ensemble_decision["clinical_notes"],
                consensus_level=ensemble_decision["consensus_level"],
                diversity_score=self._calculate_diversity_score(expert_assessments),
                processing_metadata=processing_metadata,
                rule_based_score=rule_based_score,
                hybrid_confidence=ensemble_decision.get("hybrid_confidence")
            )

            self.logger.info(
                f"✅ Trials ensemble processing complete for trial {input_data.get('trial_id', 'unknown')}: "
                f"match={result.is_match}, confidence={result.confidence_score:.3f}, "
                f"elements={len(ensemble_decision.get('mcode_elements', []))}"
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ Trials ensemble processing failed: {str(e)}")
            raise

    async def _get_expert_assessments(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get assessments from trials-specific experts.

        Args:
            input_data: Trial data for processing
            criteria_data: mCODE extraction criteria

        Returns:
            List of expert assessments with mCODE elements and validation
        """
        assessments = []
        expert_types = [
            "mcode_extractor", "clinical_validator", "evidence_analyzer",
            "tnm_staging_specialist", "clinical_terminologist",
            "treatment_regimen_specialist", "biomarker_specialist"
        ]

        for expert_type in expert_types[:self.max_experts]:  # Limit to max_experts
            try:
                assessment = await self._get_single_expert_assessment(
                    expert_type, input_data, criteria_data
                )
                assessments.append(assessment)

            except Exception as e:
                self.logger.warning(f"⚠️ Expert {expert_type} assessment failed: {str(e)}")
                continue

        if len(assessments) < self.min_experts:
            self.logger.warning(
                f"⚠️ Only {len(assessments)} experts succeeded, minimum required: {self.min_experts}"
            )

        return assessments

    async def _get_single_expert_assessment(
        self,
        expert_type: str,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get assessment from a single trials expert.

        Args:
            expert_type: Type of expert (mcode_extractor, clinical_validator, evidence_analyzer)
            input_data: Trial data
            criteria_data: mCODE criteria

        Returns:
            Expert assessment with mCODE elements and validation
        """
        # This would integrate with LLM services to get expert assessments
        # For now, return mock assessment structure

        assessment_start = time.time()

        # Mock expert assessment - in real implementation, this would call LLM
        if expert_type == "mcode_extractor":
            mcode_elements, confidence = self._extract_mcode_elements(input_data)
            reasoning = f"Extracted {len(mcode_elements)} mCODE elements from trial criteria"
        elif expert_type == "clinical_validator":
            mcode_elements, confidence = self._validate_clinical_accuracy(input_data)
            reasoning = "Validated clinical accuracy of extracted mCODE elements"
        elif expert_type == "evidence_analyzer":
            mcode_elements, confidence = self._analyze_evidence_quality(input_data)
            reasoning = "Analyzed evidence quality and supporting documentation"
        elif expert_type == "tnm_staging_specialist":
            mcode_elements, confidence = self._extract_tnm_staging(input_data)
            reasoning = "Analyzed AJCC TNM staging requirements and classifications"
        elif expert_type == "clinical_terminologist":
            mcode_elements, confidence = self._validate_terminology_codes(input_data)
            reasoning = "Validated SNOMED CT, LOINC, and ICD-O-3 coding standards"
        elif expert_type == "treatment_regimen_specialist":
            mcode_elements, confidence = self._classify_treatment_regimens(input_data)
            reasoning = "Classified cancer treatment protocols and therapeutic approaches"
        elif expert_type == "biomarker_specialist":
            mcode_elements, confidence = self._analyze_biomarkers(input_data)
            reasoning = "Analyzed tumor markers and molecular diagnostic requirements"
        else:
            mcode_elements, confidence = [], 0.0
            reasoning = "Unknown expert type"

        # Perform mCODE validation
        validation_result = self._validate_mcode_elements(mcode_elements, input_data)

        assessment = {
            "expert_type": expert_type,
            "assessment": {
                "is_match": confidence > 0.5,
                "confidence_score": confidence,
                "reasoning": reasoning,
                "matched_criteria": [elem.element_type for elem in mcode_elements],
                "unmatched_criteria": [],  # Would be populated based on validation
                "clinical_notes": f"mCODE extraction by {expert_type}",
                "mcode_elements": [elem.model_dump() for elem in mcode_elements],
                "validation_results": validation_result.model_dump(),
                "processing_time": time.time() - assessment_start
            }
        }

        return assessment

    def _extract_mcode_elements(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Extract mCODE elements from trial data."""
        elements = []
        confidence = 0.0

        # Extract from eligibility criteria - use flattened structure from trials processor
        eligibility = trial_data.get("eligibility_criteria", "")

        if eligibility and isinstance(eligibility, str):
            # Mock extraction - would use LLM in real implementation
            if "breast cancer" in eligibility.lower():
                elements.append(McodeElement(
                    element_type="CancerCondition",
                    code="254837009",
                    display="Breast Cancer",
                    system="http://snomed.info/sct",
                    confidence_score=0.9,
                    evidence_text="Found in eligibility criteria"
                ))
                confidence = 0.85

        return elements, confidence

    def _validate_clinical_accuracy(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Validate clinical accuracy of mCODE elements."""
        # Mock validation - would perform clinical validation
        elements = self._extract_mcode_elements(trial_data)[0]
        confidence = 0.88 if elements else 0.0
        return elements, confidence

    def _analyze_evidence_quality(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Analyze evidence quality for mCODE elements."""
        # Mock evidence analysis
        elements = self._extract_mcode_elements(trial_data)[0]
        confidence = 0.82 if elements else 0.0
        return elements, confidence

    def _extract_tnm_staging(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Extract TNM staging information from trial data."""
        elements = []
        confidence = 0.0

        # Extract from eligibility criteria - use flattened structure from trials processor
        eligibility = trial_data.get("eligibility_criteria", "")

        if eligibility and isinstance(eligibility, str):
            # Mock TNM staging extraction - would use LLM in real implementation
            eligibility_lower = eligibility.lower()
            if any(stage in eligibility_lower for stage in ["stage i", "stage ii", "stage iii", "stage iv", "tnm", "ajcc"]):
                elements.append(McodeElement(
                    element_type="CancerStage",
                    code="385356007",
                    display="TNM Cancer Staging",
                    system="http://snomed.info/sct",
                    confidence_score=0.85,
                    evidence_text="Found TNM staging references in eligibility criteria"
                ))
                confidence = 0.83

        return elements, confidence

    def _validate_terminology_codes(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Validate clinical terminology codes (SNOMED CT, LOINC, ICD-O-3)."""
        # Mock terminology validation - would perform code validation
        elements = self._extract_mcode_elements(trial_data)[0]
        confidence = 0.91 if elements else 0.0
        return elements, confidence

    def _classify_treatment_regimens(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Classify cancer treatment regimens and protocols."""
        elements = []
        confidence = 0.0

        # Extract from eligibility criteria - use flattened structure from trials processor
        eligibility = trial_data.get("eligibility_criteria", "")

        if eligibility and isinstance(eligibility, str):
            # Mock treatment classification - would use LLM in real implementation
            eligibility_lower = eligibility.lower()
            if any(treatment in eligibility_lower for treatment in ["chemotherapy", "radiation", "surgery", "immunotherapy", "hormone therapy"]):
                elements.append(McodeElement(
                    element_type="TreatmentRegimen",
                    code="416377005",
                    display="Cancer Treatment Protocol",
                    system="http://snomed.info/sct",
                    confidence_score=0.80,
                    evidence_text="Found treatment protocol references in eligibility criteria"
                ))
                confidence = 0.78

        return elements, confidence

    def _analyze_biomarkers(self, trial_data: Dict[str, Any]) -> tuple[List[McodeElement], float]:
        """Analyze tumor markers and molecular diagnostics."""
        elements = []
        confidence = 0.0

        # Extract from eligibility criteria - use flattened structure from trials processor
        eligibility = trial_data.get("eligibility_criteria", "")

        if eligibility and isinstance(eligibility, str):
            # Mock biomarker analysis - would use LLM in real implementation
            eligibility_lower = eligibility.lower()
            if any(biomarker in eligibility_lower for biomarker in ["her2", "estrogen", "progesterone", "ki-67", "biomarker", "molecular"]):
                elements.append(McodeElement(
                    element_type="Biomarker",
                    code="118645006",
                    display="Tumor Marker",
                    system="http://snomed.info/sct",
                    confidence_score=0.82,
                    evidence_text="Found biomarker references in eligibility criteria"
                ))
                confidence = 0.79

        return elements, confidence

    def _validate_mcode_elements(
        self,
        elements: List[McodeElement],
        trial_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate mCODE elements against mCODE standards."""
        validation_errors = []
        warnings = []
        compliance_score = 1.0

        # Basic validation
        for element in elements:
            if not element.code:
                validation_errors.append(f"Element {element.element_type} missing code")
                compliance_score -= 0.1

            if not element.system:
                warnings.append(f"Element {element.element_type} missing coding system")

        # Ensure compliance score is between 0 and 1
        compliance_score = max(0.0, min(1.0, compliance_score))

        return ValidationResult(
            compliance_score=compliance_score,
            validation_errors=validation_errors,
            warnings=warnings,
            required_elements_present=[elem.element_type for elem in elements],
            missing_elements=[]
        )

    def _apply_consensus_method(
        self,
        assessments: List[Dict[str, Any]],
        rule_based_score: Optional[float],
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply the configured consensus method for trials processing."""
        if self.consensus_method.value == "dynamic_weighting":
            return self._dynamic_weighting_ensemble(assessments, rule_based_score, input_data, criteria_data)
        else:
            # Use weighted majority vote as default
            base_result = self._weighted_majority_vote_ensemble(assessments, rule_based_score)

            # Add trials-specific fields
            base_result.update({
                "mcode_elements": self._aggregate_mcode_elements(assessments),
                "validation_score": self._calculate_ensemble_validation_score(assessments)
            })

            return base_result

    def _aggregate_mcode_elements(self, assessments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate mCODE elements from all expert assessments."""
        all_elements = []

        for assessment in assessments:
            elements = assessment.get("assessment", {}).get("mcode_elements", [])
            all_elements.extend(elements)

        # Remove duplicates based on element_type and code
        seen = set()
        unique_elements = []

        for element in all_elements:
            key = (element.get("element_type"), element.get("code"))
            if key not in seen:
                seen.add(key)
                unique_elements.append(element)

        return unique_elements

    def _calculate_ensemble_validation_score(self, assessments: List[Dict[str, Any]]) -> float:
        """Calculate overall validation score from expert assessments."""
        if not assessments:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for assessment in assessments:
            validation_results = assessment.get("assessment", {}).get("validation_results", {})
            compliance_score = validation_results.get("compliance_score", 0.0)

            expert_type = assessment.get("expert_type", "unknown")
            weight = self.expert_weights.get(expert_type, self.expert_weights.get("mcode_extractor"))
            if weight:
                effective_weight = weight.base_weight * weight.reliability_score
                total_score += compliance_score * effective_weight
                total_weight += effective_weight

        return total_score / total_weight if total_weight > 0 else 0.0

    async def _get_rule_based_score(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Optional[float]:
        """Get rule-based score for mCODE compliance in trials."""
        score = 0.0
        factors = 0

        # Check for presence of key trial elements
        if input_data.get("eligibility_criteria"):
            score += 0.3
            factors += 1

        if input_data.get("conditions"):
            score += 0.2
            factors += 1

        if input_data.get("phase"):
            score += 0.1
            factors += 1

        # Check for mCODE-relevant keywords - use flattened structure
        eligibility_text = input_data.get("eligibility_criteria", "")
        eligibility_text = eligibility_text.lower() if isinstance(eligibility_text, str) else ""
        mcode_keywords = ["cancer", "tumor", "carcinoma", "neoplasm", "stage", "grade"]
        keyword_matches = sum(1 for keyword in mcode_keywords if keyword in eligibility_text)

        if keyword_matches > 0:
            score += min(0.4, keyword_matches * 0.1)
            factors += 1

        return score / factors if factors > 0 else 0.0

    def _calculate_case_complexity(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> float:
        """Calculate case complexity for trials processing."""
        complexity = 0.0

        # Trial complexity factors - use flattened structure
        eligibility = input_data.get("eligibility_criteria", "")
        eligibility_length = len(eligibility) if isinstance(eligibility, str) else 0

        if eligibility_length > 1000:
            complexity += 0.3
        elif eligibility_length > 500:
            complexity += 0.2

        # Number of conditions
        conditions_count = len(input_data.get("conditions", []))
        if conditions_count > 3:
            complexity += 0.2
        elif conditions_count > 1:
            complexity += 0.1

        # Phase complexity - handle both string and list formats
        phase = input_data.get("phase", "")
        if isinstance(phase, list):
            # Convert list to string for processing
            phase_str = " ".join(str(p) for p in phase).lower()
        elif isinstance(phase, str):
            phase_str = phase.lower()
        else:
            phase_str = str(phase).lower()

        if "iii" in phase_str or "iv" in phase_str:
            complexity += 0.2

        return min(1.0, complexity)
