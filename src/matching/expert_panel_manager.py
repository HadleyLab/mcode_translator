"""
ExpertPanelManager - Manages a panel of specialized LLM experts for clinical trial matching.

Coordinates concurrent expert assessments and implements diversity-aware expert selection
for comprehensive patient-trial matching analysis.
"""

import asyncio
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from matching.clinical_expert_agent import ClinicalExpertAgent
from utils.api_manager import APIManager
from utils.config import Config
from utils.logging_config import get_logger


class ExpertPanelAssessment:
    """Container for individual expert assessment results."""

    def __init__(
        self,
        expert_type: str,
        assessment: Dict[str, Any],
        processing_time: float,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.expert_type = expert_type
        self.assessment = assessment
        self.processing_time = processing_time
        self.success = success
        self.error = error
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expert_type": self.expert_type,
            "assessment": self.assessment,
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp
        }


class ExpertPanelManager:
    """
    Manages a panel of specialized LLM experts for comprehensive clinical trial matching.

    Coordinates concurrent expert assessments with diversity-aware selection
    and ensemble decision making capabilities.
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        config: Optional[Config] = None,
        max_concurrent_experts: int = 3,
        enable_diversity_selection: bool = True
    ):
        """Initialize the expert panel manager.

        Args:
            model_name: LLM model to use for all experts
            config: Configuration instance
            max_concurrent_experts: Maximum number of experts to run concurrently
            enable_diversity_selection: Whether to use diversity-aware expert selection
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.config = config or Config()
        self.max_concurrent_experts = max_concurrent_experts
        self.enable_diversity_selection = enable_diversity_selection

        # Initialize expert panel
        self.experts = self._initialize_expert_panel()
        self.expert_types = list(self.experts.keys())

        # Thread pool for concurrent execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_experts)

        # Initialize caching infrastructure
        self.api_manager = APIManager()

        # Load ensemble config for cache settings
        ensemble_config_path = Path(__file__).parent.parent / "config" / "ensemble_config.json"
        try:
            import json
            with open(ensemble_config_path) as f:
                ensemble_config = json.load(f)
            cache_config = ensemble_config.get("performance_optimization", {}).get("caching", {}).get("expert_panel_cache", {})
        except Exception:
            # Fallback to default cache config
            cache_config = {"enabled": True, "namespace": "expert_panel"}

        self.enable_caching = cache_config.get("enabled", True)
        self.cache_namespace = cache_config.get("namespace", "expert_panel")

        # Cache performance tracking for the entire panel
        self.panel_cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "expert_cache_stats": {}
        }

        # Initialize per-expert cache stats
        for expert_type in self.experts.keys():
            self.panel_cache_stats["expert_cache_stats"][expert_type] = {
                "requests": 0,
                "hits": 0,
                "misses": 0,
                "time_saved": 0.0
            }

        self.logger.info(
            f"✅ ExpertPanelManager initialized with {len(self.experts)} experts, "
            f"max_concurrent={max_concurrent_experts}, diversity_selection={enable_diversity_selection}, "
            f"caching={self.enable_caching}"
        )

    def _initialize_expert_panel(self) -> Dict[str, ClinicalExpertAgent]:
        """Initialize the panel of clinical expert agents."""
        experts = {}

        expert_configs = {
            "clinical_reasoning": {
                "description": "Specializes in detailed clinical rationale and safety considerations",
                "weight": 1.0
            },
            "pattern_recognition": {
                "description": "Expert in identifying complex patterns in clinical data",
                "weight": 0.9
            },
            "comprehensive_analyst": {
                "description": "Provides holistic assessment and risk-benefit analysis",
                "weight": 1.1
            }
        }

        for expert_type, config in expert_configs.items():
            try:
                expert = ClinicalExpertAgent(
                    model_name=self.model_name,
                    expert_type=expert_type,
                    config=self.config
                )
                experts[expert_type] = expert
                self.logger.debug(f"✅ Initialized {expert_type} expert: {config['description']}")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {expert_type} expert: {e}")
                continue

        if not experts:
            raise ValueError("Failed to initialize any experts in the panel")

        return experts

    async def _assess_with_expert_panel_no_cache(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any],
        expert_selection: Optional[List[str]] = None,
        diversity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Assess with expert panel without caching (internal method)."""
        # This is the original logic without caching
        start_time = time.time()

        try:
            # Select experts for this assessment
            selected_experts = self._select_experts(
                expert_selection, diversity_threshold, patient_data, trial_criteria
            )

            if not selected_experts:
                raise ValueError("No experts available for assessment")

            self.logger.info(f"🎯 Selected experts: {selected_experts}")

            # Run expert assessments concurrently
            assessment_tasks = []
            for expert_type in selected_experts:
                if expert_type in self.experts:
                    task = self._run_expert_assessment(
                        expert_type, patient_data, trial_criteria
                    )
                    assessment_tasks.append(task)

            if not assessment_tasks:
                raise ValueError("No valid assessment tasks created")

            # Execute assessments concurrently
            expert_results = await asyncio.gather(*assessment_tasks, return_exceptions=True)

            # Process results and filter out exceptions
            valid_assessments = []
            failed_experts = []

            for i, result in enumerate(expert_results):
                expert_type = selected_experts[i]
                if isinstance(result, Exception):
                    self.logger.error(f"❌ {expert_type} assessment failed: {result}")
                    failed_experts.append(expert_type)
                elif isinstance(result, ExpertPanelAssessment):
                    if result.success:
                        valid_assessments.append(result)
                        self.logger.debug(f"✅ {expert_type} assessment completed in {result.processing_time:.2f}s")
                    else:
                        self.logger.error(f"❌ {expert_type} assessment failed: {result.error}")
                        failed_experts.append(expert_type)
                else:
                    self.logger.error(f"❌ {expert_type} returned invalid result type")
                    failed_experts.append(expert_type)

            if not valid_assessments:
                raise ValueError(f"All expert assessments failed: {failed_experts}")

            # Create ensemble decision
            ensemble_result = self._create_ensemble_decision(
                valid_assessments, patient_data, trial_criteria
            )

            total_time = time.time() - start_time

            self.logger.info(
                f"✅ Expert panel assessment completed in {total_time:.2f}s: "
                f"{len(valid_assessments)} successful, {len(failed_experts)} failed"
            )

            return ensemble_result

        except Exception as e:
            self.logger.error(f"❌ Expert panel assessment failed: {e}")
            total_time = time.time() - start_time
            return self._create_failed_ensemble_result(str(e), total_time)

    async def assess_with_expert_panel(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any],
        expert_selection: Optional[List[str]] = None,
        diversity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Assess patient-trial match using expert panel with caching.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary
            expert_selection: Specific experts to use (if None, uses diversity-aware selection)
            diversity_threshold: Threshold for diversity in expert selection

        Returns:
            Comprehensive assessment with ensemble decision
        """
        self.logger.info(
            f"🔬 Expert panel assessment starting for patient {patient_data.get('id', 'unknown')}"
        )

        start_time = time.time()
        self.panel_cache_stats["total_requests"] += 1

        # Skip caching if disabled
        if not self.enable_caching:
            self.logger.debug("💾 Panel caching disabled - proceeding with direct assessment")
            return await self._assess_with_expert_panel_no_cache(
                patient_data, trial_criteria, expert_selection, diversity_threshold
            )

        # Generate panel cache key
        panel_cache_key = self._generate_panel_cache_key(patient_data, trial_criteria, expert_selection, diversity_threshold)

        try:
            # Check cache first
            cache = await self.api_manager.aget_cache(self.cache_namespace)
            cached_result = await cache.aget_by_key(panel_cache_key)

            if cached_result is not None:
                self.panel_cache_stats["cache_hits"] += 1
                total_time = time.time() - start_time
                self.panel_cache_stats["total_time_saved"] += total_time

                self.logger.info("💾 PANEL CACHE HIT - using cached panel result")

                # Add cache metadata to result
                cached_result["cache_info"] = {
                    "source": "panel_cache",
                    "cache_key_hash": hashlib.md5(json.dumps(panel_cache_key, sort_keys=True).encode()).hexdigest()[:8],
                    "time_saved_seconds": total_time
                }

                return cached_result

            # Cache miss - perform panel assessment
            self.panel_cache_stats["cache_misses"] += 1
            self.logger.debug("💾 PANEL CACHE MISS - performing new panel assessment")

            # Select experts for this assessment
            selected_experts = self._select_experts(
                expert_selection, diversity_threshold, patient_data, trial_criteria
            )

            if not selected_experts:
                raise ValueError("No experts available for assessment")

            self.logger.info(f"🎯 Selected experts: {selected_experts}")

            # Run expert assessments concurrently
            assessment_tasks = []
            for expert_type in selected_experts:
                if expert_type in self.experts:
                    task = self._run_expert_assessment(
                        expert_type, patient_data, trial_criteria
                    )
                    assessment_tasks.append(task)

            if not assessment_tasks:
                raise ValueError("No valid assessment tasks created")

            # Execute assessments concurrently
            expert_results = await asyncio.gather(*assessment_tasks, return_exceptions=True)

            # Process results and filter out exceptions
            valid_assessments = []
            failed_experts = []

            for i, result in enumerate(expert_results):
                expert_type = selected_experts[i]
                if isinstance(result, Exception):
                    self.logger.error(f"❌ {expert_type} assessment failed: {result}")
                    failed_experts.append(expert_type)
                elif isinstance(result, ExpertPanelAssessment):
                    if result.success:
                        valid_assessments.append(result)
                        self.logger.debug(f"✅ {expert_type} assessment completed in {result.processing_time:.2f}s")
                    else:
                        self.logger.error(f"❌ {expert_type} assessment failed: {result.error}")
                        failed_experts.append(expert_type)
                else:
                    self.logger.error(f"❌ {expert_type} returned invalid result type")
                    failed_experts.append(expert_type)

            if not valid_assessments:
                raise ValueError(f"All expert assessments failed: {failed_experts}")

            # Create ensemble decision
            ensemble_result = self._create_ensemble_decision(
                valid_assessments, patient_data, trial_criteria
            )

            total_time = time.time() - start_time

            # Only cache successful assessments
            if ensemble_result.get("success", True):
                await cache.aset_by_key(ensemble_result, panel_cache_key)
                self.logger.debug("💾 PANEL RESULT CACHED")

            self.logger.info(
                f"✅ Expert panel assessment completed in {total_time:.2f}s: "
                f"{len(valid_assessments)} successful, {len(failed_experts)} failed"
            )

            return ensemble_result

        except Exception as e:
            self.logger.error(f"❌ Expert panel assessment failed: {e}")
            total_time = time.time() - start_time
            return self._create_failed_ensemble_result(str(e), total_time)

    def _select_experts(
        self,
        expert_selection: Optional[List[str]],
        diversity_threshold: float,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> List[str]:
        """Select experts for assessment using diversity-aware selection.

        Args:
            expert_selection: Pre-selected experts (if provided)
            diversity_threshold: Threshold for diversity selection
            patient_data: Patient data for context-aware selection
            trial_criteria: Trial criteria for context-aware selection

        Returns:
            List of selected expert types
        """
        if expert_selection:
            # Use pre-selected experts if provided
            return [expert for expert in expert_selection if expert in self.experts]

        if not self.enable_diversity_selection:
            # Use all available experts if diversity selection disabled
            return list(self.experts.keys())

        # Diversity-aware selection based on patient/trial characteristics
        selected_experts = []

        # Always include clinical reasoning for fundamental assessment
        if "clinical_reasoning" in self.experts:
            selected_experts.append("clinical_reasoning")

        # Select additional experts based on case complexity
        complexity_score = self._calculate_case_complexity(patient_data, trial_criteria)

        if complexity_score > diversity_threshold:
            # High complexity case - use comprehensive analyst
            if "comprehensive_analyst" in self.experts:
                selected_experts.append("comprehensive_analyst")
        else:
            # Moderate complexity - use pattern recognition
            if "pattern_recognition" in self.experts:
                selected_experts.append("pattern_recognition")

        # Ensure minimum diversity
        if len(selected_experts) < 2 and len(self.experts) >= 2:
            available_experts = [e for e in self.experts.keys() if e not in selected_experts]
            if available_experts:
                selected_experts.append(random.choice(available_experts))

        return selected_experts

    def _calculate_case_complexity(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> float:
        """Calculate case complexity score for diversity selection.

        Args:
            patient_data: Patient information
            trial_criteria: Trial criteria

        Returns:
            Complexity score between 0.0 and 1.0
        """
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

    async def _run_expert_assessment(
        self,
        expert_type: str,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> ExpertPanelAssessment:
        """Run individual expert assessment.

        Args:
            expert_type: Type of expert to run
            patient_data: Patient data
            trial_criteria: Trial criteria

        Returns:
            ExpertPanelAssessment result
        """
        start_time = time.time()

        try:
            expert = self.experts[expert_type]
            assessment = await expert.assess_match(patient_data, trial_criteria)

            processing_time = time.time() - start_time

            return ExpertPanelAssessment(
                expert_type=expert_type,
                assessment=assessment,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ {expert_type} assessment error: {e}")

            return ExpertPanelAssessment(
                expert_type=expert_type,
                assessment={},
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    def _create_ensemble_decision(
        self,
        assessments: List[ExpertPanelAssessment],
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create ensemble decision from individual expert assessments.

        Args:
            assessments: List of successful expert assessments
            patient_data: Original patient data
            trial_criteria: Original trial criteria

        Returns:
            Ensemble decision with aggregated results
        """
        # Extract individual decisions
        individual_decisions = []
        total_confidence = 0.0
        expert_weights = {
            "clinical_reasoning": 1.0,
            "pattern_recognition": 0.9,
            "comprehensive_analyst": 1.1
        }

        for assessment in assessments:
            expert_type = assessment.expert_type
            weight = expert_weights.get(expert_type, 1.0)

            if assessment.success and assessment.assessment.get("is_match") is not None:
                confidence = assessment.assessment.get("confidence_score", 0.0)
                weighted_confidence = confidence * weight
                total_confidence += weighted_confidence

                individual_decisions.append({
                    "expert_type": expert_type,
                    "is_match": assessment.assessment["is_match"],
                    "confidence": confidence,
                    "weight": weight,
                    "weighted_confidence": weighted_confidence,
                    "assessment": assessment.assessment
                })

        if not individual_decisions:
            return self._create_failed_ensemble_result("No valid decisions from experts")

        # Calculate ensemble decision
        avg_weighted_confidence = total_confidence / len(individual_decisions)

        # Majority vote for final decision
        match_votes = sum(1 for decision in individual_decisions if decision["is_match"])
        no_match_votes = len(individual_decisions) - match_votes

        # Weight votes by confidence
        weighted_match_score = sum(
            decision["weighted_confidence"] for decision in individual_decisions
            if decision["is_match"]
        )
        weighted_no_match_score = sum(
            decision["weighted_confidence"] for decision in individual_decisions
            if not decision["is_match"]
        )

        # Final decision based on weighted majority
        ensemble_match = weighted_match_score > weighted_no_match_score

        # Aggregate reasoning from all experts
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

        # Create ensemble result
        ensemble_result = {
            "is_match": ensemble_match,
            "confidence_score": avg_weighted_confidence,
            "ensemble_method": "weighted_majority_vote",
            "expert_assessments": [assessment.to_dict() for assessment in assessments],
            "individual_decisions": individual_decisions,
            "reasoning": " | ".join(all_reasoning),
            "matched_criteria": unique_matched_criteria,
            "unmatched_criteria": unique_unmatched_criteria,
            "clinical_notes": " | ".join(all_clinical_notes),
            "consensus_level": self._calculate_consensus_level(individual_decisions),
            "assessment_metadata": {
                "total_experts": len(assessments),
                "successful_experts": len([a for a in assessments if a.success]),
                "avg_processing_time": sum(a.processing_time for a in assessments) / len(assessments),
                "diversity_score": self._calculate_diversity_score(assessments)
            },
            "cache_performance": self._get_panel_cache_stats()
        }

        return ensemble_result

    def _calculate_consensus_level(self, decisions: List[Dict[str, Any]]) -> str:
        """Calculate consensus level among expert decisions."""
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
        max_possible_types = len(self.experts)

        return len(expert_types) / max_possible_types if max_possible_types > 0 else 0.0

    def _create_failed_ensemble_result(self, error_message: str, processing_time: float = 0.0) -> Dict[str, Any]:
        """Create failed ensemble result structure."""
        return {
            "is_match": False,
            "confidence_score": 0.0,
            "ensemble_method": "none",
            "expert_assessments": [],
            "individual_decisions": [],
            "reasoning": f"Ensemble assessment failed: {error_message}",
            "matched_criteria": [],
            "unmatched_criteria": [],
            "clinical_notes": "Assessment could not be completed",
            "consensus_level": "none",
            "assessment_metadata": {
                "total_experts": 0,
                "successful_experts": 0,
                "avg_processing_time": processing_time,
                "diversity_score": 0.0,
                "error": error_message
            },
            "success": False,
            "error": error_message
        }

    async def get_expert_panel_status(self) -> Dict[str, Any]:
        """Get status of all experts in the panel.

        Returns:
            Status information for all experts
        """
        status_info = {
            "total_experts": len(self.experts),
            "expert_types": list(self.experts.keys()),
            "max_concurrent_experts": self.max_concurrent_experts,
            "diversity_selection_enabled": self.enable_diversity_selection,
            "model_name": self.model_name,
            "experts_status": {}
        }

        for expert_type, expert in self.experts.items():
            status_info["experts_status"][expert_type] = {
                "initialized": True,
                "model_name": expert.model_name,
                "expert_type": expert.expert_type
            }

        return status_info

    def _update_expert_cache_stats(self, expert_type: str, cache_hit: bool, time_saved: float = 0.0):
        """Update cache statistics for a specific expert.

        Args:
            expert_type: Type of expert
            cache_hit: Whether this was a cache hit
            time_saved: Time saved by cache hit in seconds
        """
        if expert_type in self.panel_cache_stats["expert_cache_stats"]:
            stats = self.panel_cache_stats["expert_cache_stats"][expert_type]
            stats["requests"] += 1
            if cache_hit:
                stats["hits"] += 1
            else:
                stats["misses"] += 1
            stats["time_saved"] += time_saved

    def _generate_panel_cache_key(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any],
        expert_selection: Optional[List[str]],
        diversity_threshold: float
    ) -> Dict[str, Any]:
        """Generate deterministic cache key for panel assessment.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary
            expert_selection: Selected experts for assessment
            diversity_threshold: Threshold for diversity selection

        Returns:
            Cache key dictionary with deterministic components
        """
        # Create patient data hash
        patient_str = json.dumps(patient_data, sort_keys=True)
        patient_hash = hashlib.md5(patient_str.encode()).hexdigest()

        # Create trial criteria hash
        trial_str = json.dumps(trial_criteria, sort_keys=True)
        trial_hash = hashlib.md5(trial_str.encode()).hexdigest()

        # Create expert selection hash
        if expert_selection:
            expert_selection_str = json.dumps(sorted(expert_selection))
        else:
            # For diversity selection, include threshold in key
            expert_selection_str = f"diversity_selection:{diversity_threshold}"

        expert_hash = hashlib.md5(expert_selection_str.encode()).hexdigest()

        # Create panel configuration hash
        panel_config = {
            "max_concurrent_experts": self.max_concurrent_experts,
            "diversity_selection_enabled": self.enable_diversity_selection,
            "expert_types": sorted(self.experts.keys()),
            "model_name": self.model_name
        }
        config_hash = hashlib.md5(json.dumps(panel_config, sort_keys=True).encode()).hexdigest()

        # Combine all components into cache key
        cache_key = {
            "panel_type": "expert_panel_assessment",
            "patient_hash": patient_hash,
            "trial_hash": trial_hash,
            "expert_hash": expert_hash,
            "config_hash": config_hash,
            "namespace": self.cache_namespace
        }

        return cache_key

    def _get_panel_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for the entire panel.

        Returns:
            Dictionary with panel-wide cache statistics
        """
        total_requests = self.panel_cache_stats["total_requests"]
        total_hits = self.panel_cache_stats["cache_hits"]
        total_misses = self.panel_cache_stats["cache_misses"]

        hit_rate = (total_hits / total_requests) if total_requests > 0 else 0.0

        # Calculate per-expert hit rates
        expert_stats = {}
        for expert_type, stats in self.panel_cache_stats["expert_cache_stats"].items():
            expert_total = stats["requests"]
            expert_hit_rate = (stats["hits"] / expert_total) if expert_total > 0 else 0.0
            expert_stats[expert_type] = {
                "requests": expert_total,
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": expert_hit_rate,
                "time_saved_seconds": stats["time_saved"]
            }

        return {
            "panel_total_requests": total_requests,
            "panel_cache_hits": total_hits,
            "panel_cache_misses": total_misses,
            "panel_hit_rate": hit_rate,
            "panel_total_time_saved_seconds": self.panel_cache_stats["total_time_saved"],
            "expert_cache_stats": expert_stats,
            "cache_enabled": self.enable_caching,
            "cache_namespace": self.cache_namespace
        }

    def log_cache_performance(self) -> None:
        """Log comprehensive cache performance statistics."""
        if not self.enable_caching:
            self.logger.info("💾 Caching disabled - no performance data to log")
            return

        stats = self._get_panel_cache_stats()

        self.logger.info("📊 EXPERT PANEL CACHE PERFORMANCE SUMMARY:")
        self.logger.info(f"  Total Requests: {stats['panel_total_requests']}")
        self.logger.info(f"  Cache Hits: {stats['panel_cache_hits']}")
        self.logger.info(f"  Cache Misses: {stats['panel_cache_misses']}")
        self.logger.info(f"  Hit Rate: {stats['panel_hit_rate']:.2%}")
        self.logger.info(f"  Time Saved: {stats['panel_total_time_saved_seconds']:.2f}s")

        # Log per-expert statistics
        self.logger.info("  Per-Expert Statistics:")
        for expert_type, expert_stats in stats['expert_cache_stats'].items():
            self.logger.info(f"    {expert_type}: {expert_stats['hit_rate']:.2%} hit rate ({expert_stats['hits']}/{expert_stats['requests']})")

        # Calculate cost savings estimate (assuming ~5s per API call)
        avg_api_time = 5.0
        estimated_cost_savings = stats['panel_total_time_saved_seconds'] / avg_api_time
        self.logger.info(f"  Estimated API Calls Saved: {estimated_cost_savings:.1f}")

    def get_cache_optimization_recommendations(self) -> List[str]:
        """Get recommendations for cache optimization based on performance data.

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        stats = self._get_panel_cache_stats()

        if not self.enable_caching:
            recommendations.append("Enable caching to improve performance and reduce API costs")
            return recommendations

        # Analyze hit rate
        hit_rate = stats['panel_hit_rate']
        if hit_rate < 0.3:
            recommendations.append("Low cache hit rate detected - consider increasing cache TTL or reviewing cache key strategy")
        elif hit_rate > 0.8:
            recommendations.append("High cache hit rate - consider reducing cache TTL to ensure data freshness")

        # Analyze per-expert performance
        for expert_type, expert_stats in stats['expert_cache_stats'].items():
            if expert_stats['requests'] > 10 and expert_stats['hit_rate'] < 0.2:
                recommendations.append(f"Low hit rate for {expert_type} - may need different cache key strategy")

        # Check for time savings
        if stats['panel_total_time_saved_seconds'] > 300:  # 5 minutes
            recommendations.append("Significant time savings achieved - consider increasing cache TTL for more savings")

        if not recommendations:
            recommendations.append("Cache performance is optimal - no changes recommended")

        return recommendations

    def shutdown(self):
        """Shutdown the expert panel manager and cleanup resources."""
        self.logger.info("🔄 Shutting down ExpertPanelManager")

        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)

        self.logger.info("✅ ExpertPanelManager shutdown complete")
