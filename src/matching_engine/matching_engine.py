import logging
from typing import Dict, List, Any, Tuple
from src.pipeline.breast_cancer_profile import BreastCancerProfile
from src.utils.metrics import MatchingMetrics

class MatchingEngine:
    """
    Engine for matching breast cancer patient profiles to clinical trials
    based on extracted mCODE features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profile_validator = BreastCancerProfile()
        self.metrics = MatchingMetrics()
    
    def match_trials(self, patient_profile: Dict, trials: List[Dict]) -> List[Dict]:
        """
        Match patient profile to clinical trials based on breast cancer features
        
        Args:
            patient_profile: Dictionary containing patient's mCODE features
            trials: List of clinical trials with extracted mCODE features
            
        Returns:
            List of matching trials with match scores and reasons
        """
        try:
            # Validate patient profile
            validated_profile = self.profile_validator.process(patient_profile.copy())
            
            # Calculate matches
            matched_trials = []
            for trial in trials:
                if 'mcode_data' not in trial:
                    self.logger.debug("Skipping trial without mcode_data")
                    continue
                    
                trial_features = trial['mcode_data']['features']
                nct_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'Unknown')
                self.logger.debug(f"Calculating match for trial {nct_id}")
                match_score, match_reasons = self._calculate_match(validated_profile, trial_features)
                self.logger.info(f"Match score for trial {nct_id}: {match_score}")
                
                if match_score > 0:
                    matched_trial = trial.copy()
                    matched_trial['match_score'] = match_score
                    matched_trial['match_reasons'] = match_reasons
                    matched_trials.append(matched_trial)
                    self.logger.debug(f"Trial {nct_id} added to matched trials with score {match_score}")
                    
                    # Record metrics
                    self.metrics.record_match(match_reasons, validated_profile.get('genomic_variants', []))
            
            # Sort by match score descending
            matched_trials.sort(key=lambda t: t['match_score'], reverse=True)
            
            # Update metrics
            self.metrics.total_patients += 1
            self.metrics.total_trials += len(trials)
            
            return matched_trials
            
        except Exception as e:
            self.logger.error(f"Matching failed: {str(e)}")
            return []
            
    def log_metrics(self):
        """Log matching metrics"""
        self.metrics.log_summary()
    
    def _calculate_match(self, patient: Dict, trial: Dict) -> Tuple[float, List[str]]:
        """
        Calculate match score between patient profile and trial criteria
        
        Returns:
            Tuple of (match_score, match_reasons)
        """
        score = 0.0
        reasons = []
        
        # Biomarker matching
        patient_biomarkers = {b['name']: b for b in patient.get('biomarkers', [])}
        self.logger.debug(f"Patient biomarkers: {list(patient_biomarkers.keys())}")
        self.logger.debug(f"Trial biomarkers: {[b['name'] for b in trial.get('biomarkers', [])]}")
        for trial_biomarker in trial.get('biomarkers', []):
            name = trial_biomarker['name']
            if name in patient_biomarkers:
                p_status = patient_biomarkers[name]['status'].lower()
                t_status = trial_biomarker['status'].lower()
                
                if p_status == t_status:
                    score += 1.0
                    reasons.append(f"Biomarker match: {name} ({t_status})")
                    self.logger.debug(f"Biomarker match: {name} ({t_status}) (+1.0 points)")
                elif t_status == 'any':
                    score += 0.5
                    reasons.append(f"Biomarker acceptable: {name} ({p_status})")
                    self.logger.debug(f"Biomarker acceptable: {name} ({p_status}) (+0.5 points)")
        
        # Genomic variant matching
        patient_variants = {(v['gene'], v['variant']) for v in patient.get('genomic_variants', [])}
        self.logger.debug(f"Patient variants: {patient_variants}")
        self.logger.debug(f"Trial variants: {[(v['gene'], v['variant']) for v in trial.get('genomic_variants', [])]}")
        for trial_variant in trial.get('genomic_variants', []):
            key = (trial_variant['gene'], trial_variant['variant'])
            if key in patient_variants:
                score += 1.0
                reasons.append(f"Variant match: {trial_variant['gene']} {trial_variant['variant']}")
                self.logger.debug(f"Variant match: {trial_variant['gene']} {trial_variant['variant']} (+1.0 points)")
        
        # Cancer stage matching
        if patient.get('cancer_characteristics', {}).get('stage') and \
           trial.get('cancer_characteristics', {}).get('stage'):
            p_stage = patient['cancer_characteristics']['stage']
            t_stage = trial['cancer_characteristics']['stage']
            self.logger.debug(f"Patient stage: {p_stage}, Trial stage: {t_stage}")
            
            if p_stage == t_stage:
                score += 1.0
                reasons.append(f"Stage match: {t_stage}")
                self.logger.debug(f"Stage match: {t_stage} (+1.0 points)")
            elif self._stage_compatible(p_stage, t_stage):
                score += 0.7
                reasons.append(f"Stage compatible: {p_stage} vs {t_stage}")
                self.logger.debug(f"Stage compatible: {p_stage} vs {t_stage} (+0.7 points)")
        
        # Treatment history matching
        common_treatments = set(patient.get('treatment_history', {}).get('chemotherapy', [])) & \
                           set(trial.get('treatment_history', {}).get('chemotherapy', []))
        self.logger.debug(f"Patient treatments: {patient.get('treatment_history', {}).get('chemotherapy', [])}")
        self.logger.debug(f"Trial treatments: {trial.get('treatment_history', {}).get('chemotherapy', [])}")
        if common_treatments:
            points = min(len(common_treatments) * 0.3, 1.0)
            score += points
            reasons.append(f"Shared treatments: {', '.join(common_treatments)}")
            self.logger.debug(f"Shared treatments: {', '.join(common_treatments)} (+{points} points)")
        
        # Age matching
        p_age = patient.get('demographics', {}).get('age', {})
        t_age = trial.get('demographics', {}).get('age', {})
        self.logger.debug(f"Patient age: {p_age}, Trial age range: {t_age}")
        if p_age and t_age:
            p_value = p_age.get('value') or (p_age.get('min', 0) + p_age.get('max', 100)) / 2
            t_min = t_age.get('min', 0)
            t_max = t_age.get('max', 100)
            
            if t_min <= p_value <= t_max:
                score += 0.5
                reasons.append(f"Age match: {p_value} in [{t_min}, {t_max}]")
                self.logger.debug(f"Age match: {p_value} in [{t_min}, {t_max}] (+0.5 points)")
        
        return score, reasons
    
    def _stage_compatible(self, patient_stage: str, trial_stage: str) -> bool:
        """
        Check if patient stage is compatible with trial stage requirements
        """
        # Simplified compatibility check - could be enhanced with TNM parsing
        if patient_stage == trial_stage:
            return True
            
        # Check for metastatic compatibility
        if 'M1' in trial_stage and 'M1' in patient_stage:
            return True
            
        # Check for stage grouping compatibility
        stage_groups = {
            'early': ['0', 'I', 'II'],
            'advanced': ['III', 'IV']
        }
        
        for group, stages in stage_groups.items():
            if any(s in patient_stage for s in stages) and any(s in trial_stage for s in stages):
                return True
                
        return False