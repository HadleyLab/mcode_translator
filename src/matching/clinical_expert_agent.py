"""
ClinicalExpertAgent - Specialized LLM expert for clinical trial matching.

Implements specialized prompts for different clinical reasoning styles with
confidence scoring and clinical rationale generation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.llm.service import LLMService
from shared.models import PatientTrialMatchResponse
from utils.api_manager import APIManager
from utils.config import Config
from utils.logging_config import get_logger
from utils.prompt_loader import prompt_loader


class ClinicalExpertAgent:
    """
    Specialized clinical expert agent for patient-trial matching.

    Uses different clinical reasoning styles and provides detailed
    confidence scoring with clinical rationale.
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        expert_type: str = "clinical_reasoning",
        config: Optional[Config] = None
    ):
        """Initialize the clinical expert agent.

        Args:
            model_name: LLM model to use for analysis
            expert_type: Type of clinical expertise ("clinical_reasoning", "pattern_recognition", "comprehensive_analyst")
            config: Configuration instance (creates new one if not provided)
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.expert_type = expert_type
        self.config = config or Config()

        # Initialize LLM service for this expert
        self.llm_service = LLMService(self.config, model_name, f"expert_panel_{expert_type}")

        # Initialize caching infrastructure
        self.api_manager = APIManager()
        # Load ensemble config directly for cache settings
        ensemble_config_path = Path(__file__).parent.parent / "config" / "ensemble_config.json"
        try:
            import json
            with open(ensemble_config_path) as f:
                ensemble_config = json.load(f)
            self.cache_config = ensemble_config.get("performance_optimization", {}).get("caching", {}).get("expert_panel_cache", {})
        except Exception:
            # Fallback to default cache config
            self.cache_config = {"enabled": True, "namespace": "expert_panel"}

        self.enable_caching = self.cache_config.get("enabled", True)
        self.cache_namespace = self.cache_config.get("namespace", "expert_panel")

        # Cache performance tracking
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "total_cache_time_saved": 0.0
        }

        # Map expert types to prompt files
        self.expert_prompts = {
            "clinical_reasoning": "clinical_reasoning_specialist",
            "pattern_recognition": "pattern_recognition_expert",
            "comprehensive_analyst": "comprehensive_analyst"
        }

        self.logger.info(
            f"✅ ClinicalExpertAgent initialized: {expert_type} with model {model_name}, caching={self.enable_caching}"
        )

    def _generate_cache_key(self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deterministic cache key for expert assessment.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            Cache key dictionary with deterministic components
        """
        # Create patient data hash
        patient_str = json.dumps(patient_data, sort_keys=True)
        patient_hash = hashlib.md5(patient_str.encode()).hexdigest()

        # Create trial criteria hash
        trial_str = json.dumps(trial_criteria, sort_keys=True)
        trial_hash = hashlib.md5(trial_str.encode()).hexdigest()

        # Create prompt hash for this expert type
        prompt_name = self.expert_prompts.get(self.expert_type, "clinical_reasoning_specialist")
        prompt_hash = hashlib.md5(f"{self.expert_type}:{prompt_name}".encode()).hexdigest()

        # Combine all components into cache key
        cache_key = {
            "expert_type": self.expert_type,
            "patient_hash": patient_hash,
            "trial_hash": trial_hash,
            "prompt_hash": prompt_hash,
            "model": self.model_name,
            "namespace": self.cache_namespace
        }

        return cache_key

    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics for this expert."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests) if total_requests > 0 else 0.0

        return {
            "expert_type": self.expert_type,
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["hits"],
            "cache_misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "total_time_saved_seconds": self.cache_stats["total_cache_time_saved"]
        }

    async def assess_match(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess patient-trial match using specialized clinical expertise with caching.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            Detailed match assessment with clinical rationale
        """
        self.cache_stats["total_requests"] += 1

        # Skip caching if disabled
        if not self.enable_caching:
            self.logger.debug(f"💾 Caching disabled for {self.expert_type}")
            return await self._assess_match_no_cache(patient_data, trial_criteria)

        self.logger.info(
            f"🔬 {self.expert_type} assessing match for patient {patient_data.get('id', 'unknown')}"
        )

        # Generate cache key
        cache_key = self._generate_cache_key(patient_data, trial_criteria)

        try:
            # Check cache first
            cache = await self.api_manager.aget_cache(self.cache_namespace)
            cached_result = await cache.aget_by_key(cache_key)

            if cached_result is not None:
                self.cache_stats["hits"] += 1
                self.logger.info(f"💾 CACHE HIT: {self.expert_type} - using cached result")

                # Add cache metadata to result
                cached_result["cache_info"] = {
                    "source": "cache",
                    "expert_type": self.expert_type,
                    "cache_key_hash": hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]
                }

                return cached_result

            # Cache miss - perform assessment
            self.cache_stats["misses"] += 1
            self.logger.debug(f"💾 CACHE MISS: {self.expert_type} - performing new assessment")

            # Perform the actual assessment
            assessment = await self._assess_match_no_cache(patient_data, trial_criteria)

            if assessment.get("success", True):  # Only cache successful assessments
                # Store in cache
                await cache.aset_by_key(assessment, cache_key)
                self.logger.debug(f"💾 CACHE STORED: {self.expert_type}")

            return assessment

        except Exception as e:
            self.logger.error(f"❌ {self.expert_type} assessment error: {e}")
            return self._create_failed_assessment(str(e))

    async def _assess_match_no_cache(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform assessment without caching (internal method).

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            Detailed match assessment with clinical rationale
        """
        try:
            # Get specialized prompt for this expert type
            prompt_name = self.expert_prompts.get(self.expert_type, "clinical_reasoning_specialist")
            prompt = self._create_expert_prompt(prompt_name, patient_data, trial_criteria)

            # Use LLM service for assessment
            response = await self.llm_service.match_patient_to_trial(
                patient_data, trial_criteria
            )

            if not response.success:
                self.logger.error(f"❌ {self.expert_type} assessment failed: {response.error_message or 'Unknown error'}")
                return self._create_failed_assessment(response.error_message or "Unknown error")

            # Parse and enhance the response with expert-specific insights
            assessment = self._parse_expert_response(response, patient_data, trial_criteria)

            self.logger.info(
                f"✅ {self.expert_type} assessment completed: match={assessment['is_match']}, confidence={assessment['confidence_score']}"
            )

            return assessment

        except Exception as e:
            self.logger.error(f"❌ {self.expert_type} assessment error: {e}")
            return self._create_failed_assessment(str(e))

    def _create_expert_prompt(
        self,
        prompt_name: str,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> str:
        """Create specialized prompt for expert analysis.

        Args:
            prompt_name: Name of the expert prompt to use
            patient_data: Patient information
            trial_criteria: Trial criteria

        Returns:
            Formatted expert prompt
        """
        try:
            # Load the expert prompt template
            base_prompt = prompt_loader.get_prompt(
                prompt_name,
                patient_data=patient_data,
                trial_criteria=trial_criteria
            )

            # Add expert-specific context
            expert_context = self._get_expert_context()
            enhanced_prompt = f"{expert_context}\n\n{base_prompt}"

            return enhanced_prompt

        except Exception as e:
            self.logger.error(f"Failed to create expert prompt: {e}")
            # Fallback to basic prompt
            return self._create_fallback_prompt(patient_data, trial_criteria)

    def _get_expert_context(self) -> str:
        """Get expert-specific context based on expert type."""
        contexts = {
            "clinical_reasoning": """As a Clinical Reasoning Specialist, you bring deep expertise in:
- Detailed clinical rationale for inclusion/exclusion decisions
- Assessment of clinical appropriateness and safety considerations
- Identification of nuanced clinical factors affecting eligibility
- Evaluation of disease stage, treatment history, and comorbidities""",

            "pattern_recognition": """As a Pattern Recognition Expert, you specialize in:
- Recognition of subtle patterns in patient characteristics and trial requirements
- Identification of eligibility patterns across multiple clinical dimensions
- Detection of edge cases and unusual clinical presentations
- Pattern matching for complex inclusion/exclusion scenarios""",

            "comprehensive_analyst": """As a Comprehensive Clinical Analyst, you provide:
- Comprehensive evaluation of all clinical dimensions simultaneously
- Integration of multiple data sources and clinical parameters
- Holistic risk-benefit analysis for trial participation
- Synthesis of complex clinical information into actionable insights"""
        }

        return contexts.get(self.expert_type, contexts["clinical_reasoning"])

    def _create_fallback_prompt(
        self,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> str:
        """Create fallback prompt if expert prompt loading fails."""
        patient_summary = json.dumps(patient_data, indent=2)
        trial_criteria_text = json.dumps(trial_criteria, indent=2)

        return f"""You are a clinical trial matching expert. Analyze if this patient meets the trial criteria:

PATIENT DATA:
{patient_summary}

TRIAL CRITERIA:
{trial_criteria_text}

Provide detailed clinical reasoning for your match decision in JSON format."""

    def _parse_expert_response(
        self,
        response: PatientTrialMatchResponse,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse and enhance expert response with additional insights.

        Args:
            response: Raw LLM service response
            patient_data: Original patient data
            trial_criteria: Original trial criteria

        Returns:
            Enhanced assessment with expert insights
        """
        # Start with the basic response data
        assessment = {
            "is_match": response.is_match,
            "confidence_score": response.confidence_score,
            "reasoning": response.reasoning,
            "matched_criteria": response.matched_criteria,
            "unmatched_criteria": response.unmatched_criteria,
            "clinical_notes": response.clinical_notes,
            "expert_type": self.expert_type,
            "model_used": self.model_name,
            "processing_metadata": response.processing_metadata
        }

        # Add expert-specific enhancements
        if self.expert_type == "clinical_reasoning":
            assessment.update(self._enhance_clinical_reasoning(response, patient_data))
        elif self.expert_type == "pattern_recognition":
            assessment.update(self._enhance_pattern_recognition(response, patient_data))
        elif self.expert_type == "comprehensive_analyst":
            assessment.update(self._enhance_comprehensive_analysis(response, patient_data))

        return assessment

    def _enhance_clinical_reasoning(
        self,
        response: PatientTrialMatchResponse,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add clinical reasoning specific enhancements."""
        return {
            "clinical_validation": self._validate_clinical_rationale(response),
            "safety_considerations": self._assess_safety_considerations(patient_data),
            "clinical_confidence_factors": self._identify_confidence_factors(response)
        }

    def _enhance_pattern_recognition(
        self,
        response: PatientTrialMatchResponse,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add pattern recognition specific enhancements."""
        return {
            "pattern_analysis": self._analyze_clinical_patterns(patient_data),
            "edge_case_flags": self._identify_edge_cases(response),
            "pattern_confidence": self._calculate_pattern_confidence(response)
        }

    def _enhance_comprehensive_analysis(
        self,
        response: PatientTrialMatchResponse,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add comprehensive analysis specific enhancements."""
        return {
            "holistic_assessment": self._create_holistic_assessment(response, patient_data),
            "risk_benefit_analysis": self._analyze_risk_benefit(response),
            "monitoring_recommendations": self._generate_monitoring_plan(response)
        }

    def _validate_clinical_rationale(self, response: PatientTrialMatchResponse) -> Dict[str, Any]:
        """Validate the clinical rationale provided in the response."""
        reasoning = response.reasoning or ""

        return {
            "rationale_completeness": len(reasoning) > 100,  # Basic heuristic
            "clinical_terms_present": any(term in reasoning.lower() for term in
                                        ["stage", "performance", "toxicity", "comorbidity"]),
            "decision_consistency": self._check_decision_consistency(response)
        }

    def _assess_safety_considerations(self, patient_data: Dict[str, Any]) -> List[str]:
        """Assess safety considerations based on patient data."""
        considerations = []

        # Check for common safety concerns
        if patient_data.get("comorbidities"):
            considerations.append("Review comorbidities for potential interactions")

        if patient_data.get("current_medications"):
            considerations.append("Assess drug interactions with trial therapy")

        if patient_data.get("age", 0) > 70:
            considerations.append("Consider age-related toxicity risks")

        return considerations

    def _identify_confidence_factors(self, response: PatientTrialMatchResponse) -> List[str]:
        """Identify factors affecting confidence in the match decision."""
        factors = []

        if response.confidence_score > 0.9:
            factors.append("Strong match across all criteria")
        elif response.confidence_score > 0.7:
            factors.append("Good match with minor uncertainties")
        else:
            factors.append("Match decision requires careful review")

        if len(response.unmatched_criteria) > 0:
            factors.append(f"{len(response.unmatched_criteria)} criteria need further evaluation")

        return factors

    def _analyze_clinical_patterns(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze clinical patterns in patient data."""
        return {
            "disease_pattern": self._categorize_disease_pattern(patient_data),
            "treatment_pattern": self._analyze_treatment_history(patient_data),
            "toxicity_pattern": self._assess_toxicity_profile(patient_data)
        }

    def _identify_edge_cases(self, response: PatientTrialMatchResponse) -> List[str]:
        """Identify potential edge cases in the matching decision."""
        edge_cases = []

        if len(response.unmatched_criteria) > 0:
            edge_cases.append("Criteria exceptions may apply")

        if response.confidence_score < 0.8:
            edge_cases.append("Low confidence suggests need for specialist review")

        return edge_cases

    def _calculate_pattern_confidence(self, response: PatientTrialMatchResponse) -> float:
        """Calculate pattern-based confidence score."""
        base_confidence = response.confidence_score

        # Adjust based on pattern analysis factors
        if len(response.matched_criteria) > len(response.unmatched_criteria):
            pattern_multiplier = 1.1
        else:
            pattern_multiplier = 0.9

        return min(base_confidence * pattern_multiplier, 1.0)

    def _create_holistic_assessment(
        self,
        response: PatientTrialMatchResponse,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create holistic assessment of patient suitability."""
        return {
            "overall_suitability": "excellent" if response.confidence_score > 0.9 else "good" if response.confidence_score > 0.7 else "fair",
            "clinical_complexity": self._assess_clinical_complexity(patient_data),
            "monitoring_needs": self._determine_monitoring_needs(response)
        }

    def _analyze_risk_benefit(self, response: PatientTrialMatchResponse) -> Dict[str, Any]:
        """Analyze risk-benefit profile for trial participation."""
        return {
            "benefit_potential": "high" if response.confidence_score > 0.8 else "moderate",
            "risk_level": "low" if len(response.unmatched_criteria) == 0 else "moderate",
            "recommendation": "proceed" if response.is_match else "consider_alternatives"
        }

    def _generate_monitoring_plan(self, response: PatientTrialMatchResponse) -> List[str]:
        """Generate monitoring recommendations based on assessment."""
        recommendations = []

        if response.is_match:
            recommendations.append("Standard trial monitoring per protocol")
            if len(response.unmatched_criteria) > 0:
                recommendations.append("Enhanced monitoring for exception criteria")

        return recommendations

    def _categorize_disease_pattern(self, patient_data: Dict[str, Any]) -> str:
        """Categorize the patient's disease pattern."""
        # Simple pattern categorization based on available data
        if patient_data.get("stage") in ["IV", "stage_iv", "metastatic"]:
            return "advanced_metastatic"
        elif patient_data.get("stage") in ["III", "stage_iii", "locally_advanced"]:
            return "locally_advanced"
        else:
            return "early_stage"

    def _analyze_treatment_history(self, patient_data: Dict[str, Any]) -> str:
        """Analyze the patient's treatment history pattern."""
        treatments = patient_data.get("prior_treatments", [])
        if len(treatments) > 3:
            return "heavily_pre_treated"
        elif len(treatments) > 0:
            return "prior_treatment_experienced"
        else:
            return "treatment_naive"

    def _assess_toxicity_profile(self, patient_data: Dict[str, Any]) -> str:
        """Assess the patient's toxicity profile."""
        # Basic toxicity assessment
        if patient_data.get("significant_toxicities"):
            return "complex_toxicity_profile"
        else:
            return "favorable_toxicity_profile"

    def _assess_clinical_complexity(self, patient_data: Dict[str, Any]) -> str:
        """Assess overall clinical complexity."""
        complexity_factors = 0

        if patient_data.get("comorbidities"):
            complexity_factors += 1
        if len(patient_data.get("current_medications", [])) > 3:
            complexity_factors += 1
        if patient_data.get("age", 0) > 70:
            complexity_factors += 1

        if complexity_factors > 2:
            return "high"
        elif complexity_factors > 0:
            return "moderate"
        else:
            return "low"

    def _determine_monitoring_needs(self, response: PatientTrialMatchResponse) -> str:
        """Determine monitoring needs based on response."""
        if response.confidence_score < 0.7:
            return "enhanced_monitoring"
        elif len(response.unmatched_criteria) > 0:
            return "standard_plus_monitoring"
        else:
            return "standard_monitoring"

    def _check_decision_consistency(self, response: PatientTrialMatchResponse) -> bool:
        """Check if the match decision is consistent with reasoning."""
        # Basic consistency check
        if response.is_match and response.confidence_score < 0.5:
            return False
        if not response.is_match and response.confidence_score > 0.8:
            return False
        return True

    def _create_failed_assessment(self, error_message: str) -> Dict[str, Any]:
        """Create assessment structure for failed analysis."""
        return {
            "is_match": False,
            "confidence_score": 0.0,
            "reasoning": f"Assessment failed: {error_message}",
            "matched_criteria": [],
            "unmatched_criteria": [],
            "clinical_notes": "Unable to complete expert assessment",
            "expert_type": self.expert_type,
            "model_used": self.model_name,
            "error": error_message,
            "success": False
        }
