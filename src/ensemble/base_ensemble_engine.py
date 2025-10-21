"""
BaseEnsembleEngine - Abstract base class for ensemble processing.

Provides shared consensus method implementations, expert weight management,
and common data structures for ensemble-based decision making.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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


class BaseEnsembleEngine(ABC):
    """
    Abstract base class for ensemble processing engines.

    Provides shared consensus method implementations, expert weight management,
    and common data structures that can be extended for different use cases
    (matching, trials processing, etc.).
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
        Initialize the base ensemble engine.

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
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.config = config or Config()
        self.consensus_method = consensus_method
        self.confidence_calibration = confidence_calibration
        self.enable_rule_based_integration = enable_rule_based_integration
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries

        # Expert weight management
        self.expert_weights: Dict[str, ExpertWeight] = self._initialize_expert_weights()
        self.historical_performance: Dict[str, Dict[str, float]] = {}

        # Confidence calibration models
        self.confidence_calibrators: Dict[str, Any] = {}

        self.logger.info(
            f"✅ BaseEnsembleEngine initialized: method={consensus_method.value}, "
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

    @abstractmethod
    async def process_ensemble(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> EnsembleResult:
        """
        Abstract method for processing ensemble decisions.

        Args:
            input_data: Input data for processing (patient data, trial data, etc.)
            criteria_data: Criteria data for matching/evaluation

        Returns:
            EnsembleResult with comprehensive decision information
        """
        pass

    @abstractmethod
    async def _get_expert_assessments(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to get assessments from expert panel.

        Args:
            input_data: Input data for processing
            criteria_data: Criteria data for matching/evaluation

        Returns:
            List of expert assessments
        """
        pass

    async def _get_rule_based_score(
        self,
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Optional[float]:
        """
        Get rule-based score for integration.

        This is a base implementation that can be overridden by subclasses.
        For now, returns a simple heuristic score.

        Args:
            input_data: Input data for processing
            criteria_data: Criteria data for matching/evaluation

        Returns:
            Rule-based confidence score (0.0-1.0) or None if not available
        """
        # Simple heuristic based on available data
        score = 0.5  # Base score

        # This can be customized by subclasses based on their specific needs
        # For example, matching engines might check cancer type alignment,
        # while trial processors might check different criteria

        return min(score, 1.0)

    def _weighted_majority_vote_ensemble(
        self,
        assessments: List[Dict[str, Any]],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using weighted majority vote."""
        individual_decisions = []
        total_weighted_match = 0.0
        total_weighted_no_match = 0.0
        total_confidence = 0.0

        # Process expert assessments
        for assessment in assessments:
            expert_type = assessment.get("expert_type", "unknown")
            weight = self.expert_weights.get(expert_type, ExpertWeight(expert_type, 1.0, 0.8, 0.0, 0.8, time.time()))

            assessment_data = assessment.get("assessment", {})
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
                total_weighted_no_match += rule_confidence * rule_weight

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
        assessments: List[Dict[str, Any]],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using confidence-weighted scoring."""
        # Similar to weighted majority vote but with different weighting scheme
        # Implementation would focus more on confidence calibration
        return self._weighted_majority_vote_ensemble(assessments, rule_based_score)

    def _bayesian_ensemble(
        self,
        assessments: List[Dict[str, Any]],
        rule_based_score: Optional[float]
    ) -> Dict[str, Any]:
        """Create ensemble decision using Bayesian inference."""
        # Bayesian approach would model expert reliability and update beliefs
        # For now, fall back to weighted majority vote
        self.logger.info("🔄 Bayesian ensemble using weighted majority vote fallback")
        return self._weighted_majority_vote_ensemble(assessments, rule_based_score)

    def _dynamic_weighting_ensemble(
        self,
        assessments: List[Dict[str, Any]],
        rule_based_score: Optional[float],
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ensemble decision using dynamic weighting based on context."""
        # Adjust weights based on input/criteria characteristics and expert performance
        dynamic_weights = self._calculate_dynamic_weights(
            assessments, input_data, criteria_data
        )

        # Apply dynamic weights to create decision
        individual_decisions = []
        total_weighted_match = 0.0
        total_weighted_no_match = 0.0

        for assessment in assessments:
            expert_type = assessment.get("expert_type", "unknown")
            base_weight = self.expert_weights.get(expert_type, ExpertWeight(expert_type, 1.0, 0.8, 0.0, 0.8, time.time()))
            dynamic_weight = dynamic_weights.get(expert_type, base_weight.base_weight)

            assessment_data = assessment.get("assessment", {})
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
        assessments: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on context and performance."""
        dynamic_weights = {}

        # Base weights from configuration
        for expert_type, weight in self.expert_weights.items():
            dynamic_weights[expert_type] = weight.base_weight

        # Adjust based on case complexity
        complexity_score = self._calculate_case_complexity(input_data, criteria_data)

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
        input_data: Dict[str, Any],
        criteria_data: Dict[str, Any]
    ) -> float:
        """Calculate case complexity for dynamic weighting."""
        complexity_factors = 0
        total_factors = 0

        # This is a base implementation - subclasses can override for specific complexity calculations
        # For example, matching might consider patient comorbidities,
        # while trials processing might consider criteria complexity

        # Generic complexity factors
        if len(str(input_data)) > 1000:  # Large input data
            complexity_factors += 1
        total_factors += 1

        if len(str(criteria_data)) > 1000:  # Complex criteria
            complexity_factors += 1
        total_factors += 1

        return complexity_factors / total_factors if total_factors > 0 else 0.5

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        assessments: List[Dict[str, Any]]
    ) -> float:
        """Calibrate confidence score using selected method."""
        if self.confidence_calibration == ConfidenceCalibration.NONE:
            return raw_confidence

        # For now, implement simple calibration based on expert agreement
        if len(assessments) > 1:
            # Calculate agreement level
            decisions = [a.get("assessment", {}).get("is_match", False) for a in assessments]
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

    def _calculate_diversity_score(self, assessments: List[Dict[str, Any]]) -> float:
        """Calculate diversity score based on expert types used."""
        expert_types = set(assessment.get("expert_type", "unknown") for assessment in assessments)
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
        """Get status of the ensemble engine."""
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
            }
        }

    def shutdown(self):
        """Shutdown the ensemble engine and cleanup resources."""
        self.logger.info("🔄 Shutting down BaseEnsembleEngine")
        self.logger.info("✅ BaseEnsembleEngine shutdown complete")
