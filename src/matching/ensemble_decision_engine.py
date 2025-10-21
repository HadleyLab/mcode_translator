"""
EnsembleDecisionEngine - Advanced ensemble scoring mechanism for clinical trial matching.

Combines multiple expert opinions using sophisticated weighted scoring, consensus mechanisms,
and dynamic weighting to provide enhanced confidence scoring and clinical rationale.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from matching.base import MatchingEngineBase
from matching.expert_panel_manager import ExpertPanelManager, ExpertPanelAssessment
from utils.config import Config
from utils.logging_config import get_logger


class ConsensusMethod(Enum):
    """Available consensus methods for ensemble decision making."""
    WEIGHTED_MAJORITY_VOTE = "weighted_majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    BAYESIAN_ENSEMBLE = "bayesian_ensemble"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


class ConfidenceCalibration(Enum):
    """Methods for calibrating confidence scores across experts."""
    ISOTONIC_REGRESSION = "isotonic_regression"
    PLATT_SCALING = "platt_scaling"
    HISTOGRAM_BINNING = "histogram_binning"
    NONE = "none"


@dataclass
class ExpertWeight:
    """Configuration for expert weights and reliability metrics."""
    expert_type: str
    base_weight: float
    reliability_score: float
    specialization_bonus: float
    historical_accuracy: float
    last_updated: float


@dataclass
class EnsembleResult:
    """Comprehensive ensemble decision result."""
    is_match: bool
    confidence_score: float
    consensus_method: str
    expert_assessments: List[Dict[str, Any]]
    individual_decisions: List[Dict[str, Any]]
    reasoning: str
    matched_criteria: List[str]
    unmatched_criteria: List[str]
    clinical_notes: str
    consensus_level: str
    diversity_score: float
    processing_metadata: Dict[str, Any]
    rule_based_score: Optional[float] = None
    hybrid_confidence: Optional[float] = None


class EnsembleDecisionEngine(MatchingEngineBase):
    """
    Advanced ensemble decision engine that combines multiple expert opinions
    with sophisticated weighting and consensus mechanisms.

    Integrates LLM expert assessments with rule-based gold standard logic
    to provide enhanced accuracy and clinical rationale.
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        config: Optional[Config] = None,
        consensus_method: ConsensusMethod = ConsensusMethod.DYNAMIC_WEIGHTING,
        confidence_calibration: ConfidenceCalibration = ConfidenceCalibration.ISOTONIC_REGRESSION,
        enable_rule_based_integration: bool = True,
        enable_dynamic_weighting: bool = True,
        min_experts: int = 2,
        max_experts: int = 3,
        cache_enabled: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the ensemble decision engine.

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
        """
        super().__init__(cache_enabled=cache_enabled, max_retries=max_retries)

        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.config = config or Config()
        self.consensus_method = consensus_method
        self.confidence_calibration = confidence_calibration
        self.enable_rule_based_integration = enable_rule_based_integration
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.min_experts = min_experts
        self.max_experts = max_experts

        # Initialize expert panel manager
        self.expert_panel = ExpertPanelManager(
            model_name=model_name,
            config=self.config,
            max_concurrent_experts=max_experts,
            enable_diversity_selection=True
        )

        # Expert weight management
        self.expert_weights: Dict[str, ExpertWeight] = self._initialize_expert_weights()
        self.historical_performance: Dict[str, Dict[str, float]] = {}

        # Rule-based integration (if enabled)
        self.rule_based_engine = None
        if enable_rule_based_integration:
            self._initialize_rule_based_engine()

        # Confidence calibration models
        self.confidence_calibrators: Dict[str, Any] = {}

        self.logger.info(
            f"✅ EnsembleDecisionEngine initialized: method={consensus_method.value}, "
            f"calibration={confidence_calibration.value}, experts={min_experts}-{max_experts}, "
            f"rule_integration={enable_rule_based_integration}"
        )

    def _initialize_expert_weights(self) -> Dict[str, ExpertWeight]:
        """Initialize expert weights with default values."""
        current_time = time.time()

        return {
            "clinical_reasoning": ExpertWeight(
                expert_type="clinical_reasoning",
                base_weight=1.0,
                reliability_score=0.85,
                specialization_bonus=0.1,
                historical_accuracy=0.82,
                last_updated=current_time
            ),
            "pattern_recognition": ExpertWeight(
                expert_type="pattern_recognition",
                base_weight=0.9,
                reliability_score=0.80,
                specialization_bonus=0.15,
                historical_accuracy=0.78,
                last_updated=current_time
            ),
            "comprehensive_analyst": ExpertWeight(
                expert_type="comprehensive_analyst",
                base_weight=1.1,
                reliability_score=0.90,
                specialization_bonus=0.05,
                historical_accuracy=0.88,
                last_updated=current_time
            )
        }

    def _initialize_rule_based_engine(self) -> None:
        """Initialize rule-based scoring engine for integration."""
        # For now, disable rule-based integration until the engine is implemented
        # This allows the ensemble system to work without the rule-based component
        self.logger.info("ℹ️ Rule-based integration disabled - will be enabled when rule-based engine is available")
        self.rule_based_engine = None

    async def match(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> bool:
        """
        Match patient data against trial criteria using ensemble decision making.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            Boolean match result from ensemble decision
        """
        try:
            ensemble_result = await self._perform_ensemble_assessment(
                patient_data, trial_criteria
            )
            return ensemble_result.is_match

        except Exception as e:
            self.logger.error(f"❌ Ensemble matching failed: {e}")
            return False

    async def _perform_ensemble_assessment(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> EnsembleResult:
        """Perform comprehensive ensemble assessment."""
        start_time = time.time()

        # Get expert panel assessments
        expert_assessments = await self._get_expert_assessments(
            patient_data, trial_criteria
        )

        if not expert_assessments:
            raise ValueError("No valid expert assessments obtained")

        # Get rule-based score if enabled
        rule_based_score = None
        if self.enable_rule_based_integration:
            rule_based_score = await self._get_rule_based_score(
                patient_data, trial_criteria
            )

        # Create ensemble decision based on selected method
        if self.consensus_method == ConsensusMethod.WEIGHTED_MAJORITY_VOTE:
            ensemble_decision = self._weighted_majority_vote_ensemble(
                expert_assessments, rule_based_score
            )
        elif self.consensus_method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            ensemble_decision = self._confidence_weighted_ensemble(
                expert_assessments, rule_based_score
            )
        elif self.consensus_method == ConsensusMethod.BAYESIAN_ENSEMBLE:
            ensemble_decision = self._bayesian_ensemble(
                expert_assessments, rule_based_score
            )
        else:  # DYNAMIC_WEIGHTING
            ensemble_decision = self._dynamic_weighting_ensemble(
                expert_assessments, rule_based_score, patient_data, trial_criteria
            )

        # Apply confidence calibration
        calibrated_confidence = self._calibrate_confidence(
            ensemble_decision["confidence_score"], expert_assessments
        )

        # Create final ensemble result
        processing_time = time.time() - start_time
        ensemble_result = EnsembleResult(
            is_match=ensemble_decision["is_match"],
            confidence_score=calibrated_confidence,
            consensus_method=self.consensus_method.value,
            expert_assessments=[assessment.to_dict() for assessment in expert_assessments],
            individual_decisions=ensemble_decision["individual_decisions"],
            reasoning=ensemble_decision["reasoning"],
            matched_criteria=ensemble_decision["matched_criteria"],
            unmatched_criteria=ensemble_decision["unmatched_criteria"],
            clinical_notes=ensemble_decision["clinical_notes"],
            consensus_level=ensemble_decision["consensus_level"],
            diversity_score=self._calculate_diversity_score(expert_assessments),
            processing_metadata={
                "total_experts": len(expert_assessments),
                "processing_time": processing_time,
                "consensus_method": self.consensus_method.value,
                "rule_based_integrated": rule_based_score is not None,
                "dynamic_weighting": self.enable_dynamic_weighting
            },
            rule_based_score=rule_based_score,
            hybrid_confidence=self._calculate_hybrid_confidence(
                calibrated_confidence, rule_based_score
            )
        )

        self.logger.info(
            f"✅ Ensemble assessment completed in {processing_time:.2f}s: "
            f"match={ensemble_result.is_match}, confidence={calibrated_confidence:.3f}, "
            f"consensus={ensemble_result.consensus_level}"
        )

        return ensemble_result

    async def _get_expert_assessments(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> List[ExpertPanelAssessment]:
        """Get assessments from expert panel."""
        try:
            # Use expert panel to get assessments
            panel_result = await self.expert_panel.assess_with_expert_panel(
                patient_data, trial_criteria
            )

            # Extract individual assessments
            assessments = []
            for assessment_data in panel_result.get("expert_assessments", []):
                assessment = ExpertPanelAssessment(
                    expert_type=assessment_data["expert_type"],
                    assessment=assessment_data["assessment"],
                    processing_time=assessment_data["processing_time"],
                    success=assessment_data["success"],
                    error=assessment_data.get("error")
                )
                assessments.append(assessment)

            return assessments

        except Exception as e:
            self.logger.error(f"❌ Failed to get expert assessments: {e}")
            return []

    async def _get_rule_based_score(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Optional[float]:
        """Get rule-based score for integration."""
        if not self.rule_based_engine:
            # For now, return a simple rule-based score based on basic criteria matching
            # This is a placeholder until the actual rule-based engine is implemented

            # Simple heuristic based on available data
            score = 0.5  # Base score

            # Adjust based on basic criteria
            if patient_data.get("cancer_type") and trial_criteria.get("conditions"):
                if patient_data["cancer_type"].lower() in str(trial_criteria["conditions"]).lower():
                    score += 0.3

            if patient_data.get("age") and trial_criteria.get("minimumAge"):
                try:
                    min_age = int(trial_criteria["minimumAge"].split()[0])
                    if patient_data["age"] >= min_age:
                        score += 0.2
                except:
                    pass

            return min(score, 1.0)

        # This code path is currently unreachable since rule_based_engine is None
        # It will be used when the actual rule-based engine is implemented
        try:  # type: ignore[unreachable]
            # Convert to rule-based engine format
            rule_result = await self.rule_based_engine.match_with_recovery(
                patient_data, trial_criteria
            )

            # Convert boolean result to confidence score (0.0-1.0 scale)
            if rule_result.is_match:
                # Use metadata confidence if available, otherwise default
                return rule_result.metadata.get("confidence", 0.8)
            else:
                return 0.2  # Low confidence for non-matches

        except Exception as e:
            self.logger.error(f"❌ Rule-based scoring failed: {e}")
            return None

    def _weighted_majority_vote_ensemble(
        self,
        assessments: List[ExpertPanelAssessment],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using weighted majority vote."""
        individual_decisions = []
        total_weighted_match = 0.0
        total_weighted_no_match = 0.0
        total_confidence = 0.0

        # Process expert assessments
        for assessment in assessments:
            if not assessment.success:
                continue

            expert_type = assessment.expert_type
            weight = self.expert_weights.get(expert_type, ExpertWeight(expert_type, 1.0, 0.8, 0.0, 0.8, time.time()))

            assessment_data = assessment.assessment
            is_match = assessment_data.get("is_match", False)
            confidence = assessment_data.get("confidence_score", 0.0)

            # Apply expert weighting
            effective_weight = weight.base_weight * weight.reliability_score
            weighted_confidence = confidence * effective_weight

            if is_match:
                total_weighted_match += weighted_confidence
            else:
                total_weighted_no_match += weighted_confidence

            total_confidence += weighted_confidence

            individual_decisions.append({
                "expert_type": expert_type,
                "is_match": is_match,
                "confidence": confidence,
                "weight": effective_weight,
                "weighted_confidence": weighted_confidence,
                "assessment": assessment_data
            })

        # Integrate rule-based score if available
        if rule_based_score is not None:
            rule_weight = 0.3  # Rule-based component weight
            rule_confidence = rule_based_score

            if rule_based_score > 0.5:
                total_weighted_match += rule_confidence * rule_weight
            else:
                total_weighted_no_match += (1 - rule_confidence) * rule_weight

            total_confidence += rule_confidence * rule_weight

            individual_decisions.append({
                "expert_type": "rule_based",
                "is_match": rule_based_score > 0.5,
                "confidence": rule_confidence,
                "weight": rule_weight,
                "weighted_confidence": rule_confidence * rule_weight,
                "assessment": {"method": "rule_based", "score": rule_based_score}
            })

        # Determine final decision
        ensemble_match = total_weighted_match > total_weighted_no_match
        avg_confidence = total_confidence / len(individual_decisions) if individual_decisions else 0.0

        # Aggregate reasoning and criteria
        all_reasoning = []
        all_matched_criteria = []
        all_unmatched_criteria = []
        all_clinical_notes = []

        for decision in individual_decisions:
            assessment_data = decision["assessment"]

            if assessment_data.get("reasoning"):
                expert_prefix = f"[{decision['expert_type'].replace('_', ' ').title()}]"
                all_reasoning.append(f"{expert_prefix}: {assessment_data['reasoning']}")

            all_matched_criteria.extend(assessment_data.get("matched_criteria", []))
            all_unmatched_criteria.extend(assessment_data.get("unmatched_criteria", []))
            if assessment_data.get("clinical_notes"):
                all_clinical_notes.append(assessment_data["clinical_notes"])

        # Remove duplicates while preserving order
        seen_matched = set()
        unique_matched_criteria = [
            criterion for criterion in all_matched_criteria
            if not (criterion in seen_matched or seen_matched.add(criterion))
        ]

        seen_unmatched = set()
        unique_unmatched_criteria = [
            criterion for criterion in all_unmatched_criteria
            if not (criterion in seen_unmatched or seen_unmatched.add(criterion))
        ]

        return {
            "is_match": ensemble_match,
            "confidence_score": avg_confidence,
            "individual_decisions": individual_decisions,
            "reasoning": " | ".join(all_reasoning),
            "matched_criteria": unique_matched_criteria,
            "unmatched_criteria": unique_unmatched_criteria,
            "clinical_notes": " | ".join(all_clinical_notes),
            "consensus_level": self._calculate_consensus_level(individual_decisions)
        }

    def _confidence_weighted_ensemble(
        self,
        assessments: List[ExpertPanelAssessment],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using confidence-weighted scoring."""
        # Similar to weighted majority vote but with different weighting scheme
        # Implementation would focus more on confidence calibration
        return self._weighted_majority_vote_ensemble(assessments, rule_based_score)

    def _bayesian_ensemble(
        self,
        assessments: List[ExpertPanelAssessment],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using Bayesian inference."""
        # Bayesian approach would model expert reliability and update beliefs
        # For now, fall back to weighted majority vote
        self.logger.info("🔄 Bayesian ensemble using weighted majority vote fallback")
        return self._weighted_majority_vote_ensemble(assessments, rule_based_score)

    def _dynamic_weighting_ensemble(
        self,
        assessments: List[ExpertPanelAssessment],
        rule_based_score: Optional[float],
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ensemble decision using dynamic weighting based on context."""
        # Adjust weights based on patient/trial characteristics and expert performance
        dynamic_weights = self._calculate_dynamic_weights(
            assessments, patient_data, trial_criteria
        )

        # Apply dynamic weights to create decision
        individual_decisions = []
        total_weighted_match = 0.0
        total_weighted_no_match = 0.0

        for assessment in assessments:
            if not assessment.success:
                continue

            expert_type = assessment.expert_type
            base_weight = self.expert_weights.get(expert_type, ExpertWeight(expert_type, 1.0, 0.8, 0.0, 0.8, time.time()))
            dynamic_weight = dynamic_weights.get(expert_type, base_weight.base_weight)

            assessment_data = assessment.assessment
            is_match = assessment_data.get("is_match", False)
            confidence = assessment_data.get("confidence_score", 0.0)

            effective_weight = dynamic_weight * base_weight.reliability_score
            weighted_confidence = confidence * effective_weight

            if is_match:
                total_weighted_match += weighted_confidence
            else:
                total_weighted_no_match += weighted_confidence

            individual_decisions.append({
                "expert_type": expert_type,
                "is_match": is_match,
                "confidence": confidence,
                "weight": effective_weight,
                "dynamic_weight": dynamic_weight,
                "weighted_confidence": weighted_confidence,
                "assessment": assessment_data
            })

        # Add rule-based if available
        if rule_based_score is not None:
            rule_weight = 0.3
            rule_confidence = rule_based_score

            if rule_based_score > 0.5:
                total_weighted_match += rule_confidence * rule_weight
            else:
                total_weighted_no_match += (1 - rule_confidence) * rule_weight

            individual_decisions.append({
                "expert_type": "rule_based",
                "is_match": rule_based_score > 0.5,
                "confidence": rule_confidence,
                "weight": rule_weight,
                "weighted_confidence": rule_confidence * rule_weight,
                "assessment": {"method": "rule_based", "score": rule_based_score}
            })

        ensemble_match = total_weighted_match > total_weighted_no_match
        avg_confidence = (total_weighted_match + total_weighted_no_match) / len(individual_decisions) if individual_decisions else 0.0

        # Aggregate results
        all_reasoning = []
        all_matched_criteria = []
        all_unmatched_criteria = []
        all_clinical_notes = []

        for decision in individual_decisions:
            assessment_data = decision["assessment"]

            if assessment_data.get("reasoning"):
                expert_prefix = f"[{decision['expert_type'].replace('_', ' ').title()}]"
                all_reasoning.append(f"{expert_prefix}: {assessment_data['reasoning']}")

            all_matched_criteria.extend(assessment_data.get("matched_criteria", []))
            all_unmatched_criteria.extend(assessment_data.get("unmatched_criteria", []))
            if assessment_data.get("clinical_notes"):
                all_clinical_notes.append(assessment_data["clinical_notes"])

        seen_matched = set()
        unique_matched_criteria = [
            criterion for criterion in all_matched_criteria
            if not (criterion in seen_matched or seen_matched.add(criterion))
        ]

        seen_unmatched = set()
        unique_unmatched_criteria = [
            criterion for criterion in all_unmatched_criteria
            if not (criterion in seen_unmatched or seen_unmatched.add(criterion))
        ]

        return {
            "is_match": ensemble_match,
            "confidence_score": avg_confidence,
            "individual_decisions": individual_decisions,
            "reasoning": " | ".join(all_reasoning),
            "matched_criteria": unique_matched_criteria,
            "unmatched_criteria": unique_unmatched_criteria,
            "clinical_notes": " | ".join(all_clinical_notes),
            "consensus_level": self._calculate_consensus_level(individual_decisions)
        }

    def _calculate_dynamic_weights(
        self,
        assessments: List[ExpertPanelAssessment],
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on context and performance."""
        dynamic_weights = {}

        # Base weights from configuration
        for expert_type, weight in self.expert_weights.items():
            dynamic_weights[expert_type] = weight.base_weight

        # Adjust based on case complexity
        complexity_score = self._calculate_case_complexity(patient_data, trial_criteria)

        if complexity_score > 0.7:
            # High complexity - favor comprehensive analyst
            dynamic_weights["comprehensive_analyst"] = max(
                dynamic_weights.get("comprehensive_analyst", 1.0) * 1.2, 1.5
            )
        elif complexity_score < 0.3:
            # Low complexity - favor pattern recognition
            dynamic_weights["pattern_recognition"] = max(
                dynamic_weights.get("pattern_recognition", 0.9) * 1.1, 1.0
            )

        # Adjust based on historical performance
        for expert_type in dynamic_weights:
            if expert_type in self.historical_performance:
                performance = self.historical_performance[expert_type]
                # Boost weight for well-performing experts
                if performance.get("accuracy", 0.5) > 0.7:
                    dynamic_weights[expert_type] *= 1.1

        return dynamic_weights

    def _calculate_case_complexity(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> float:
        """Calculate case complexity for dynamic weighting."""
        complexity_factors = 0
        total_factors = 0

        # Patient complexity factors
        if patient_data.get("comorbidities"):
            complexity_factors += 1
        total_factors += 1

        if len(patient_data.get("current_medications", [])) > 2:
            complexity_factors += 1
        total_factors += 1

        if patient_data.get("age", 0) > 65:
            complexity_factors += 1
        total_factors += 1

        # Trial complexity factors
        criteria_text = trial_criteria.get("eligibilityCriteria", "")
        if criteria_text and len(criteria_text) > 2000:
            complexity_factors += 1
        total_factors += 1

        if len(trial_criteria.get("conditions", [])) > 1:
            complexity_factors += 1
        total_factors += 1

        return complexity_factors / total_factors if total_factors > 0 else 0.5

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        assessments: List[ExpertPanelAssessment]
    ) -> float:
        """Calibrate confidence score using selected method."""
        if self.confidence_calibration == ConfidenceCalibration.NONE:
            return raw_confidence

        # For now, implement simple calibration based on expert agreement
        if len(assessments) > 1:
            # Calculate agreement level
            decisions = [a.assessment.get("is_match", False) for a in assessments if a.success]
            agreement_ratio = sum(decisions) / len(decisions) if decisions else 0.5

            # Adjust confidence based on agreement
            if agreement_ratio > 0.8:
                calibrated = min(raw_confidence * 1.1, 1.0)
            elif agreement_ratio < 0.3:
                calibrated = raw_confidence * 0.9
            else:
                calibrated = raw_confidence

            return calibrated

        return raw_confidence

    def _calculate_hybrid_confidence(
        self,
        ensemble_confidence: float,
        rule_based_score: Optional[float]
    ) -> Optional[float]:
        """Calculate hybrid confidence combining ensemble and rule-based scores."""
        if rule_based_score is None:
            return None

        # Weighted combination of ensemble and rule-based confidence
        ensemble_weight = 0.7
        rule_weight = 0.3

        return (ensemble_confidence * ensemble_weight) + (rule_based_score * rule_weight)

    def _calculate_consensus_level(self, decisions: List[Dict[str, Any]]) -> str:
        """Calculate consensus level among decisions."""
        if not decisions:
            return "none"

        match_count = sum(1 for d in decisions if d["is_match"])
        total_count = len(decisions)

        agreement_ratio = match_count / total_count

        if agreement_ratio >= 0.8:
            return "high"
        elif agreement_ratio >= 0.6:
            return "moderate"
        else:
            return "low"

    def _calculate_diversity_score(self, assessments: List[ExpertPanelAssessment]) -> float:
        """Calculate diversity score based on expert types used."""
        expert_types = set(assessment.expert_type for assessment in assessments)
        max_possible_types = len(self.expert_weights)

        return len(expert_types) / max_possible_types if max_possible_types > 0 else 0.0

    def update_expert_weights(self, performance_data: Dict[str, Dict[str, float]]):
        """Update expert weights based on performance data."""
        current_time = time.time()

        for expert_type, performance in performance_data.items():
            if expert_type in self.expert_weights:
                weight = self.expert_weights[expert_type]

                # Update historical accuracy
                if "accuracy" in performance:
                    # Exponential moving average of accuracy
                    alpha = 0.3
                    weight.historical_accuracy = (
                        alpha * performance["accuracy"] +
                        (1 - alpha) * weight.historical_accuracy
                    )

                # Update reliability score
                if "reliability" in performance:
                    weight.reliability_score = performance["reliability"]

                weight.last_updated = current_time

        self.logger.info("✅ Expert weights updated based on performance data")

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get status of the ensemble decision engine."""
        return {
            "consensus_method": self.consensus_method.value,
            "confidence_calibration": self.confidence_calibration.value,
            "rule_based_integration": self.enable_rule_based_integration,
            "dynamic_weighting": self.enable_dynamic_weighting,
            "expert_weights": {
                expert_type: {
                    "base_weight": weight.base_weight,
                    "reliability_score": weight.reliability_score,
                    "historical_accuracy": weight.historical_accuracy
                }
                for expert_type, weight in self.expert_weights.items()
            },
            "expert_panel_status": self.expert_panel.get_expert_panel_status() if self.expert_panel else None,
            "rule_based_engine_available": self.rule_based_engine is not None
        }

    def shutdown(self):
        """Shutdown the ensemble decision engine and cleanup resources."""
        self.logger.info("🔄 Shutting down EnsembleDecisionEngine")

        # Shutdown expert panel
        if self.expert_panel:
            self.expert_panel.shutdown()

        self.logger.info("✅ EnsembleDecisionEngine shutdown complete")