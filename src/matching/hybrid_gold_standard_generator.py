"""
HybridGoldStandardGenerator - Orchestrates ensemble approach for gold standard generation.

Combines LLM expert assessments with rule-based gold standard logic to generate
improved gold standard matches with better accuracy and detailed reasoning.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.matching.ensemble_decision_engine import EnsembleDecisionEngine, EnsembleResult, ConsensusMethod, ConfidenceCalibration
from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.shared.models import McodeElement


@dataclass
class HybridMatchResult:
    """Result from hybrid gold standard generation."""
    patient_id: str
    trial_id: str
    ensemble_score: float
    rule_based_score: Optional[float]
    hybrid_score: Optional[float]
    is_match: bool
    confidence_score: float
    reasoning: str
    matched_criteria: List[str]
    unmatched_criteria: List[str]
    clinical_notes: str
    mcode_criteria_met: List[str]
    processing_metadata: Dict[str, Any]
    expert_consensus_level: str
    diversity_score: float


@dataclass
class PatientProfile:
    """Structured patient profile for hybrid curation."""
    patient_id: str
    age: int
    gender: str
    cancer_type: str
    cancer_subtype: str
    stage: str
    biomarkers: Dict[str, Any]
    prior_treatments: List[str]
    comorbidities: List[str]
    performance_status: str
    mcode_elements: List[McodeElement]


@dataclass
class TrialProfile:
    """Structured trial profile for hybrid curation."""
    trial_id: str
    title: str
    cancer_types: List[str]
    phases: List[str]
    interventions: List[str]
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    biomarkers_required: Dict[str, Any]
    age_range: Tuple[int, int]
    prior_treatments_allowed: List[str]


class HybridGoldStandardGenerator:
    """
    Orchestrates the combination of LLM experts and rule-based approach
    to generate improved gold standard matches with better accuracy.

    Provides detailed reasoning for ensemble decisions and maintains
    compatibility with existing evaluation framework.
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        config: Optional[Config] = None,
        consensus_method: ConsensusMethod = ConsensusMethod.DYNAMIC_WEIGHTING,
        confidence_calibration: ConfidenceCalibration = ConfidenceCalibration.ISOTONIC_REGRESSION,
        enable_rule_based_integration: bool = True,
        min_confidence_threshold: float = 0.6,
        batch_size: int = 100
    ):
        """
        Initialize the hybrid gold standard generator.

        Args:
            model_name: LLM model to use for expert assessments
            config: Configuration instance
            consensus_method: Method for combining expert opinions
            confidence_calibration: Method for calibrating confidence scores
            enable_rule_based_integration: Whether to integrate rule-based scoring
            min_confidence_threshold: Minimum confidence for accepting matches
            batch_size: Batch size for processing patient-trial pairs
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.config = config or Config()
        self.consensus_method = consensus_method
        self.confidence_calibration = confidence_calibration
        self.enable_rule_based_integration = enable_rule_based_integration
        self.min_confidence_threshold = min_confidence_threshold
        self.batch_size = batch_size

        # Initialize ensemble decision engine
        self.ensemble_engine = EnsembleDecisionEngine(
            model_name=model_name,
            config=self.config,
            consensus_method=consensus_method,
            confidence_calibration=confidence_calibration,
            enable_rule_based_integration=enable_rule_based_integration,
            enable_dynamic_weighting=True,
            min_experts=2,
            max_experts=3
        )

        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "total_processed": 0,
            "ensemble_matches": 0,
            "high_confidence_matches": 0,
            "avg_confidence_score": 0.0,
            "avg_processing_time": 0.0
        }

        self.logger.info(
            f"✅ HybridGoldStandardGenerator initialized: model={model_name}, "
            f"consensus={consensus_method.value}, threshold={min_confidence_threshold}"
        )

    def load_patients(self, filepath: str) -> List[PatientProfile]:
        """Load and parse patient data from FHIR bundles."""
        patients = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    patient_data = json.loads(line.strip())
                    profile = self._parse_patient_data(patient_data)
                    patients.append(profile)
        except Exception as e:
            self.logger.error(f"❌ Failed to load patients from {filepath}: {e}")

        self.logger.info(f"✅ Loaded {len(patients)} patients from {filepath}")
        return patients

    def load_trials(self, filepath: str) -> List[TrialProfile]:
        """Load and parse trial data from ClinicalTrials.gov format."""
        trials = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    trial_data = json.loads(line.strip())
                    profile = self._parse_trial_data(trial_data)
                    trials.append(profile)
        except Exception as e:
            self.logger.error(f"❌ Failed to load trials from {filepath}: {e}")

        self.logger.info(f"✅ Loaded {len(trials)} trials from {filepath}")
        return trials

    def _parse_patient_data(self, data: Dict[str, Any]) -> PatientProfile:
        """Parse FHIR patient bundle into structured profile."""
        patient_resource = None
        conditions = []
        observations = []

        for entry in data.get('entry', []):
            resource = entry['resource']
            if resource['resourceType'] == 'Patient':
                patient_resource = resource
            elif resource['resourceType'] == 'Condition':
                conditions.append(resource)
            elif resource['resourceType'] == 'Observation':
                observations.append(resource)

        # Extract basic demographics
        if not patient_resource:
            raise ValueError("Patient resource not found in FHIR bundle")

        patient_id = patient_resource['id']
        age = self._calculate_age(patient_resource.get('birthDate', '1980-01-01'))
        gender = patient_resource.get('gender', 'female')

        # Extract cancer information
        cancer_type = "Breast Cancer"
        cancer_subtype = "Unknown"
        stage = "Unknown"
        biomarkers = {}

        for obs in observations:
            code = obs.get('code', {}).get('text', '')
            value = obs.get('valueString', '')

            if 'histology' in code.lower() or 'histologic' in code.lower():
                cancer_subtype = value
            elif 'stage' in code.lower():
                stage = value
            elif 'estrogen' in code.lower() or 'er' in code.lower():
                biomarkers['ER'] = 'positive' if 'positive' in value.lower() else 'negative'
            elif 'progesterone' in code.lower() or 'pr' in code.lower():
                biomarkers['PR'] = 'positive' if 'positive' in value.lower() else 'negative'
            elif 'her2' in code.lower():
                biomarkers['HER2'] = 'positive' if 'positive' in value.lower() else 'negative'

        # Default biomarkers if not found
        biomarkers.setdefault('ER', 'unknown')
        biomarkers.setdefault('PR', 'unknown')
        biomarkers.setdefault('HER2', 'unknown')

        # Determine subtype based on biomarkers
        if biomarkers['ER'] == 'negative' and biomarkers['PR'] == 'negative' and biomarkers['HER2'] == 'negative':
            cancer_subtype = "Triple Negative Breast Cancer"
        elif biomarkers['HER2'] == 'positive':
            cancer_subtype = "HER2 Positive Breast Cancer"
        elif biomarkers['ER'] == 'positive' or biomarkers['PR'] == 'positive':
            cancer_subtype = "Hormone Receptor Positive Breast Cancer"

        prior_treatments = ["Chemotherapy", "Surgery"]  # Simplified
        comorbidities: List[str] = []  # Simplified
        performance_status = "ECOG 0-1"  # Simplified

        # Create mCODE elements (simplified)
        mcode_elements = [
            McodeElement(
                element_type="CancerCondition",
                code="254837009",
                system="http://snomed.info/sct",
                display=cancer_subtype,
                confidence=0.95
            )
        ]

        return PatientProfile(
            patient_id=patient_id,
            age=age,
            gender=gender,
            cancer_type=cancer_type,
            cancer_subtype=cancer_subtype,
            stage=stage,
            biomarkers=biomarkers,
            prior_treatments=prior_treatments,
            comorbidities=comorbidities,
            performance_status=performance_status,
            mcode_elements=mcode_elements
        )

    def _parse_trial_data(self, data: Dict[str, Any]) -> TrialProfile:
        """Parse ClinicalTrials.gov trial data into structured profile."""
        protocol = data.get('protocolSection', {})

        trial_id = protocol.get('identificationModule', {}).get('nctId', '')
        title = protocol.get('identificationModule', {}).get('briefTitle', '')

        # Extract conditions
        conditions_module = protocol.get('conditionsModule', {})
        cancer_types = conditions_module.get('conditions', [])

        # Extract phases
        design = protocol.get('designModule', {})
        phases = design.get('phases', [])

        # Extract interventions
        arms = protocol.get('armsInterventionsModule', {}).get('interventions', [])
        interventions = [arm.get('name', '') for arm in arms]

        # Extract eligibility criteria
        eligibility = protocol.get('eligibilityModule', {})
        criteria_text = eligibility.get('eligibilityCriteria', '')

        # Parse inclusion/exclusion (simplified)
        inclusion_criteria = self._parse_eligibility_criteria(criteria_text)
        exclusion_criteria: Dict[str, Any] = {}  # Simplified

        # Extract biomarker requirements
        biomarkers_required = {}
        if 'triple negative' in criteria_text.lower():
            biomarkers_required['subtype'] = 'triple_negative'
        elif 'her2' in criteria_text.lower():
            biomarkers_required['HER2'] = 'positive'

        # Age range
        min_age = eligibility.get('minimumAge', '18 Years')
        max_age = eligibility.get('maximumAge', '120 Years')
        age_range = (int(min_age.split()[0]), int(max_age.split()[0]))

        # Prior treatments
        prior_treatments_allowed = []
        if 'prior chemotherapy' in criteria_text.lower():
            prior_treatments_allowed.append('chemotherapy')

        return TrialProfile(
            trial_id=trial_id,
            title=title,
            cancer_types=cancer_types,
            phases=phases,
            interventions=interventions,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            biomarkers_required=biomarkers_required,
            age_range=age_range,
            prior_treatments_allowed=prior_treatments_allowed
        )

    def _calculate_age(self, birth_date: str) -> int:
        """Calculate age from birth date."""
        try:
            year = int(birth_date.split('-')[0])
            return 2024 - year
        except:
            return 50  # Default age

    def _parse_eligibility_criteria(self, criteria_text: str) -> Dict[str, Any]:
        """Parse eligibility criteria text into structured format."""
        criteria = {}

        # Simple parsing for common criteria
        if 'ecog' in criteria_text.lower():
            criteria['performance_status'] = 'ECOG 0-2'
        if 'karnofsky' in criteria_text.lower():
            criteria['performance_status'] = 'Karnofsky >= 70'

        return criteria

    async def generate_hybrid_gold_standard(
        self,
        patients: List[PatientProfile],
        trials: List[TrialProfile],
        output_filepath: str,
        use_ensemble_approach: bool = True,
        use_rule_based_fallback: bool = True
    ) -> List[HybridMatchResult]:
        """
        Generate hybrid gold standard matches using ensemble approach.

        Args:
            patients: List of patient profiles
            trials: List of trial profiles
            output_filepath: Path to save results
            use_ensemble_approach: Whether to use ensemble decision engine
            use_rule_based_fallback: Whether to fall back to rule-based scoring

        Returns:
            List of hybrid match results
        """
        self.logger.info(
            f"🔬 Starting hybrid gold standard generation: {len(patients)} patients × {len(trials)} trials"
        )

        start_time = time.time()
        hybrid_results = []

        # Process patient-trial pairs in batches
        total_pairs = len(patients) * len(trials)

        for i in range(0, total_pairs, self.batch_size):
            batch_start_time = time.time()

            # Get batch of patient-trial pairs
            batch_pairs = []
            for patient in patients:
                for trial in trials:
                    pair_index = (patients.index(patient) * len(trials)) + trials.index(trial)
                    if i <= pair_index < i + self.batch_size:
                        batch_pairs.append((patient, trial))

            self.logger.info(f"📊 Processing batch {i//self.batch_size + 1}: {len(batch_pairs)} pairs")

            # Process batch
            for patient, trial in batch_pairs:
                try:
                    if use_ensemble_approach:
                        result = await self._generate_ensemble_match(patient, trial)
                    else:
                        result = await self._generate_rule_based_match(patient, trial)

                    if result:
                        hybrid_results.append(result)
                        self._update_performance_metrics(result)

                except Exception as e:
                    self.logger.error(f"❌ Failed to process {patient.patient_id} vs {trial.trial_id}: {e}")

            batch_time = time.time() - batch_start_time
            self.logger.info(f"✅ Batch completed in {batch_time:.2f}s")

        # Save results
        await self._save_hybrid_results(hybrid_results, output_filepath)

        total_time = time.time() - start_time
        self.logger.info(
            f"✅ Hybrid gold standard generation completed in {total_time:.2f}s: "
            f"{len(hybrid_results)} results saved to {output_filepath}"
        )

        return hybrid_results

    async def _generate_ensemble_match(
        self,
        patient: PatientProfile,
        trial: TrialProfile
    ) -> Optional[HybridMatchResult]:
        """Generate match result using ensemble decision engine."""
        try:
            # Convert profiles to dictionaries for ensemble engine
            patient_data = self._patient_profile_to_dict(patient)
            trial_criteria = self._trial_profile_to_dict(trial)

            # Get ensemble assessment
            ensemble_result: EnsembleResult = await self.ensemble_engine._perform_ensemble_assessment(
                patient_data, trial_criteria
            )

            # Convert to hybrid result format
            hybrid_result = HybridMatchResult(
                patient_id=patient.patient_id,
                trial_id=trial.trial_id,
                ensemble_score=ensemble_result.confidence_score,
                rule_based_score=ensemble_result.rule_based_score,
                hybrid_score=ensemble_result.hybrid_confidence,
                is_match=ensemble_result.is_match,
                confidence_score=ensemble_result.confidence_score,
                reasoning=ensemble_result.reasoning,
                matched_criteria=ensemble_result.matched_criteria,
                unmatched_criteria=ensemble_result.unmatched_criteria,
                clinical_notes=ensemble_result.clinical_notes,
                mcode_criteria_met=self._extract_mcode_criteria(patient, trial, ensemble_result),
                processing_metadata=ensemble_result.processing_metadata,
                expert_consensus_level=ensemble_result.consensus_level,
                diversity_score=ensemble_result.diversity_score
            )

            return hybrid_result

        except Exception as e:
            self.logger.error(f"❌ Ensemble matching failed for {patient.patient_id} vs {trial.trial_id}: {e}")
            return None

    async def _generate_rule_based_match(
        self,
        patient: PatientProfile,
        trial: TrialProfile
    ) -> Optional[HybridMatchResult]:
        """Generate match result using rule-based approach."""
        try:
            # Use the existing rule-based curation logic from data_curation_script.py
            score, justification, criteria_met = self._curate_match_with_rules(patient, trial)

            # Convert to hybrid result format
            hybrid_result = HybridMatchResult(
                patient_id=patient.patient_id,
                trial_id=trial.trial_id,
                ensemble_score=float(score) / 5.0,  # Convert 0-5 scale to 0-1
                rule_based_score=float(score) / 5.0,
                hybrid_score=float(score) / 5.0,
                is_match=score >= 3,  # Threshold for match
                confidence_score=min(float(score) / 5.0, 1.0),
                reasoning=justification,
                matched_criteria=criteria_met,
                unmatched_criteria=[],
                clinical_notes=f"Rule-based curation with score {score}/5",
                mcode_criteria_met=criteria_met,
                processing_metadata={
                    "method": "rule_based",
                    "score_range": "0-5",
                    "match_threshold": 3
                },
                expert_consensus_level="n/a",
                diversity_score=0.0
            )

            return hybrid_result

        except Exception as e:
            self.logger.error(f"❌ Rule-based matching failed for {patient.patient_id} vs {trial.trial_id}: {e}")
            return None

    def _patient_profile_to_dict(self, patient: PatientProfile) -> Dict[str, Any]:
        """Convert PatientProfile to dictionary format."""
        return {
            "id": patient.patient_id,
            "age": patient.age,
            "gender": patient.gender,
            "cancer_type": patient.cancer_type,
            "cancer_subtype": patient.cancer_subtype,
            "stage": patient.stage,
            "biomarkers": patient.biomarkers,
            "prior_treatments": patient.prior_treatments,
            "comorbidities": patient.comorbidities,
            "performance_status": patient.performance_status,
            "mcode_elements": [
                {
                    "element_type": elem.element_type,
                    "code": elem.code,
                    "system": elem.system,
                    "display": elem.display,
                    "confidence_score": elem.confidence_score
                }
                for elem in patient.mcode_elements
            ]
        }

    def _trial_profile_to_dict(self, trial: TrialProfile) -> Dict[str, Any]:
        """Convert TrialProfile to dictionary format."""
        return {
            "trial_id": trial.trial_id,
            "title": trial.title,
            "conditions": trial.cancer_types,
            "phases": trial.phases,
            "interventions": trial.interventions,
            "eligibilityCriteria": json.dumps(trial.inclusion_criteria),
            "biomarkers_required": trial.biomarkers_required,
            "minimumAge": str(trial.age_range[0]) + " Years",
            "maximumAge": str(trial.age_range[1]) + " Years",
            "prior_treatments_allowed": trial.prior_treatments_allowed
        }

    def _extract_mcode_criteria(
        self,
        patient: PatientProfile,
        trial: TrialProfile,
        ensemble_result: EnsembleResult
    ) -> List[str]:
        """Extract mCODE criteria met based on ensemble result."""
        criteria_met = []

        # Check cancer condition compatibility
        if any('breast' in ct.lower() for ct in trial.cancer_types):
            criteria_met.append("CancerCondition")

        # Check biomarker criteria
        if trial.biomarkers_required:
            for biomarker, required_value in trial.biomarkers_required.items():
                if biomarker in patient.biomarkers:
                    patient_value = patient.biomarkers[biomarker]
                    if patient_value == required_value:
                        criteria_met.append("TumorMarkerTest")

        # Check performance status
        if 'ecog' in patient.performance_status.lower():
            criteria_met.append("ECOGPerformanceStatus")

        return criteria_met

    def _curate_match_with_rules(
        self,
        patient: PatientProfile,
        trial: TrialProfile
    ) -> Tuple[int, str, List[str]]:
        """
        Perform rule-based curation of patient-trial match.

        Returns:
            Tuple of (relevance_score, justification, mcode_criteria_met)
        """
        score = 0
        justifications = []
        criteria_met = []

        # Cancer type compatibility
        if any('breast' in ct.lower() for ct in trial.cancer_types):
            score += 2
            justifications.append("Cancer type matches: Breast cancer trial suitable for breast cancer patient")
            criteria_met.append("CancerCondition")
        else:
            justifications.append("Cancer type mismatch: Trial not designed for breast cancer")
            return 0, "; ".join(justifications), criteria_met

        # Biomarker compatibility
        biomarker_score = self._evaluate_biomarker_match(patient, trial)
        score += biomarker_score
        if biomarker_score >= 1.5:
            justifications.append("Biomarker status compatible with trial requirements")
            criteria_met.append("TumorMarkerTest")
        elif biomarker_score >= 1:
            justifications.append("Biomarker status partially compatible")
        else:
            justifications.append("Biomarker mismatch may limit eligibility")

        # Stage compatibility
        stage_score = self._evaluate_stage_match(patient, trial)
        score += stage_score
        if stage_score >= 1:
            justifications.append("Disease stage appropriate for trial phase and design")

        # Age compatibility
        if patient.age >= trial.age_range[0] and patient.age <= trial.age_range[1]:
            score += 1
            justifications.append("Patient age within trial requirements")
        else:
            justifications.append("Patient age outside trial age range")

        # Performance status
        if 'ecog' in patient.performance_status.lower():
            score += 1
            justifications.append("Performance status likely compatible")
            criteria_met.append("ECOGPerformanceStatus")

        # Prior treatment compatibility
        treatment_score = self._evaluate_treatment_history(patient, trial)
        score += treatment_score
        if treatment_score >= 1:
            justifications.append("Prior treatment history compatible with trial requirements")

        # Comorbidity safety
        if not patient.comorbidities:
            score += 1
            justifications.append("No significant comorbidities that would contraindicate participation")

        # Intervention appropriateness
        intervention_score = self._evaluate_intervention_appropriateness(patient, trial)
        score += intervention_score

        # Ensure score is within 0-5 range and convert to int
        final_score = int(max(0, min(5, score)))

        # Add therapeutic rationale
        if final_score >= 4:
            justifications.append("Excellent therapeutic match with strong clinical rationale")
        elif final_score >= 3:
            justifications.append("Good therapeutic match with reasonable clinical rationale")
        elif final_score >= 2:
            justifications.append("Moderate therapeutic match; may benefit from participation")
        elif final_score >= 1:
            justifications.append("Limited therapeutic match; potential benefit uncertain")
        else:
            justifications.append("Poor therapeutic match; unlikely to benefit")

        return final_score, "; ".join(justifications), criteria_met

    def _evaluate_biomarker_match(self, patient: PatientProfile, trial: TrialProfile) -> float:
        """Evaluate biomarker compatibility."""
        score = 0

        required_subtype = trial.biomarkers_required.get('subtype')
        if required_subtype == 'triple_negative' and 'triple negative' in patient.cancer_subtype.lower():
            score += 1.5
        elif required_subtype == 'triple_negative' and 'triple negative' not in patient.cancer_subtype.lower():
            score -= 0.5

        required_her2 = trial.biomarkers_required.get('HER2')
        if required_her2 and patient.biomarkers.get('HER2') == required_her2:
            score += 1
        elif required_her2 and patient.biomarkers.get('HER2') != required_her2:
            score -= 0.5

        return score

    def _evaluate_stage_match(self, patient: PatientProfile, trial: TrialProfile) -> float:
        """Evaluate stage compatibility."""
        score = 0

        # Phase I trials: often advanced disease
        if 'PHASE1' in trial.phases and ('IV' in patient.stage or 'stage iv' in patient.stage.lower()):
            score += 1

        # Phase II/III trials: various stages
        if 'PHASE2' in trial.phases or 'PHASE3' in trial.phases:
            score += 0.5

        return score

    def _evaluate_treatment_history(self, patient: PatientProfile, trial: TrialProfile) -> float:
        """Evaluate prior treatment compatibility."""
        score = 0

        if trial.prior_treatments_allowed:
            for allowed in trial.prior_treatments_allowed:
                if any(allowed.lower() in pt.lower() for pt in patient.prior_treatments):
                    score += 0.5

        # Assume some compatibility if no specific restrictions
        if not trial.prior_treatments_allowed:
            score += 0.5

        return score

    def _evaluate_intervention_appropriateness(self, patient: PatientProfile, trial: TrialProfile) -> float:
        """Evaluate intervention appropriateness for patient."""
        score = 0

        # Target therapies for specific subtypes
        if 'her2' in ' '.join(trial.interventions).lower() and patient.biomarkers.get('HER2') == 'positive':
            score += 1
        elif 'hormone' in ' '.join(trial.interventions).lower() and patient.biomarkers.get('ER') == 'positive':
            score += 1
        elif 'chemotherapy' in ' '.join(trial.interventions).lower():
            score += 0.5  # Chemotherapy often appropriate for various subtypes

        return score

    def _update_performance_metrics(self, result: HybridMatchResult):
        """Update performance tracking metrics."""
        self.performance_metrics["total_processed"] += 1

        if result.is_match:
            self.performance_metrics["ensemble_matches"] += 1

        if result.confidence_score >= self.min_confidence_threshold:
            self.performance_metrics["high_confidence_matches"] += 1

        # Update running averages
        total = self.performance_metrics["total_processed"]
        self.performance_metrics["avg_confidence_score"] = (
            (self.performance_metrics["avg_confidence_score"] * (total - 1)) +
            result.confidence_score
        ) / total

    async def _save_hybrid_results(
        self,
        results: List[HybridMatchResult],
        output_filepath: str
    ):
        """Save hybrid results to file."""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert results to dictionaries
            results_data = []
            for result in results:
                result_dict = {
                    'patient_id': result.patient_id,
                    'trial_id': result.trial_id,
                    'ensemble_score': result.ensemble_score,
                    'rule_based_score': result.rule_based_score,
                    'hybrid_score': result.hybrid_score,
                    'is_match': result.is_match,
                    'confidence_score': result.confidence_score,
                    'reasoning': result.reasoning,
                    'matched_criteria': result.matched_criteria,
                    'unmatched_criteria': result.unmatched_criteria,
                    'clinical_notes': result.clinical_notes,
                    'mcode_criteria_met': result.mcode_criteria_met,
                    'processing_metadata': result.processing_metadata,
                    'expert_consensus_level': result.expert_consensus_level,
                    'diversity_score': result.diversity_score
                }
                results_data.append(result_dict)

            # Save to file
            with open(output_filepath, 'w') as f:
                for result in results_data:
                    f.write(json.dumps(result) + '\n')

            self.logger.info(f"💾 Saved {len(results)} hybrid results to {output_filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save hybrid results: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the hybrid generation process."""
        return {
            "total_processed": self.performance_metrics["total_processed"],
            "ensemble_matches": self.performance_metrics["ensemble_matches"],
            "high_confidence_matches": self.performance_metrics["high_confidence_matches"],
            "avg_confidence_score": self.performance_metrics["avg_confidence_score"],
            "match_rate": (
                self.performance_metrics["ensemble_matches"] / self.performance_metrics["total_processed"]
                if self.performance_metrics["total_processed"] > 0 else 0.0
            ),
            "high_confidence_rate": (
                self.performance_metrics["high_confidence_matches"] / self.performance_metrics["total_processed"]
                if self.performance_metrics["total_processed"] > 0 else 0.0
            ),
            "ensemble_engine_status": self.ensemble_engine.get_ensemble_status(),
            "configuration": {
                "model_name": self.model_name,
                "consensus_method": self.consensus_method.value,
                "confidence_calibration": self.confidence_calibration.value,
                "min_confidence_threshold": self.min_confidence_threshold,
                "batch_size": self.batch_size
            }
        }

    def shutdown(self):
        """Shutdown the hybrid gold standard generator."""
        self.logger.info("🔄 Shutting down HybridGoldStandardGenerator")

        if self.ensemble_engine:
            self.ensemble_engine.shutdown()

        self.logger.info("✅ HybridGoldStandardGenerator shutdown complete")