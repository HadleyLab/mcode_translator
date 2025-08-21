"""
Patient-Trial Matching Algorithm for mCODE Translator
Calculates match scores between patient profiles and clinical trials
"""
import logging

class PatientMatcher:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.biomarker_weights = {
            'ER': 0.8,
            'PR': 0.7,
            'HER2': 0.9,
            'PD-L1': 0.6,
            'Ki-67': 0.5
        }
        self.variant_weights = {
            'PIK3CA': 0.9,
            'TP53': 0.8,
            'BRCA1': 0.85,
            'BRCA2': 0.85,
            'ESR1': 0.7
        }
    
    def calculate_match_score(self, patient_profile, trial_features, return_details=False):
        """
        Calculate match score between patient profile and trial features
        
        Args:
            patient_profile: Dict with patient characteristics
            trial_features: Dict with trial genomic features
            return_details: Bool to return match details
            
        Returns:
            Float match score (0-100) or tuple (score, details)
        """
        score = 0
        max_score = 0
        match_details = {
            'cancer_type': False,
            'biomarkers': False,
            'genomic_variants': False,
            'stage_grade': False
        }
        
        # Log input data for debugging
        self.logger.debug(f"Calculating match score - Patient cancer type: {patient_profile.get('cancer_type')}, Trial cancer type: {trial_features.get('cancer_type')}")
        
        # Cancer type match (30% weight)
        cancer_match = patient_profile['cancer_type'] == trial_features.get('cancer_type', '')
        if cancer_match:
            score += 30
            match_details['cancer_type'] = True
            self.logger.debug("Cancer type match found (+30 points)")
            
            # Boost weights for breast cancer biomarkers
            if patient_profile['cancer_type'] == 'breast cancer':
                self.biomarker_weights = {
                    'ER': 1.0,
                    'PR': 0.9,
                    'HER2': 1.0,
                    'PD-L1': 0.7,
                    'Ki-67': 0.6
                }
        
        # Biomarker matches (40% weight)
        patient_biomarkers = patient_profile.get('biomarkers', {})
        trial_biomarkers = trial_features.get('biomarkers', {})
        
        self.logger.debug(f"Patient biomarkers: {patient_biomarkers}")
        self.logger.debug(f"Trial biomarkers: {trial_biomarkers}")
        for biomarker, status in patient_biomarkers.items():
            weight = self.biomarker_weights.get(biomarker, 0.5)
            if biomarker in trial_biomarkers:
                if status == trial_biomarkers[biomarker]:
                    points = 8 * weight
                    score += points  # 8% per matching biomarker
                    match_details['biomarkers'] = True
                    self.logger.debug(f"Biomarker match found: {biomarker} ({status}) (+{points} points)")
                max_score += 8
        
        # Genomic variant matches (20% weight)
        patient_variants = patient_profile.get('genomic_variants', [])
        trial_variants = trial_features.get('genomic_variants', [])
        
        self.logger.debug(f"Patient variants: {patient_variants}")
        self.logger.debug(f"Trial variants: {trial_variants}")
        for variant in patient_variants:
            weight = self.variant_weights.get(variant, 0.5)
            if variant in trial_variants:
                points = 4 * weight
                score += points  # 4% per matching variant
                match_details['genomic_variants'] = True
                self.logger.debug(f"Variant match found: {variant} (+{points} points)")
            max_score += 4
        
        # Stage/Grade compatibility (10% weight)
        stage_mapping = {'I':1, 'II':2, 'III':3, 'IV':4}
        grade_mapping = {'low':1, 'intermediate':2, 'high':3}
        
        # Get patient stage/grade
        patient_char = patient_profile.get('cancer_characteristics', {})
        patient_stage = stage_mapping.get(patient_char.get('stage', ''), 0)
        patient_grade = grade_mapping.get(patient_char.get('grade', ''), 0)
        
        # Get trial requirements
        trial_min_stage = stage_mapping.get(trial_features.get('min_stage', 'I'), 1)
        trial_min_grade = grade_mapping.get(trial_features.get('min_grade', 'low'), 1)
        
        self.logger.debug(f"Patient stage: {patient_stage}, Trial min stage: {trial_min_stage}")
        self.logger.debug(f"Patient grade: {patient_grade}, Trial min grade: {trial_min_grade}")
        if patient_stage >= trial_min_stage and patient_grade >= trial_min_grade:
            score += 10
            match_details['stage_grade'] = True
            self.logger.debug("Stage/Grade match found (+10 points)")
        
        final_score = min(100, int(score * 100 / max_score)) if max_score > 0 else 0
        self.logger.debug(f"Final score calculation - Score: {score}, Max score: {max_score}, Final score: {final_score}")
        
        # Remove duplicate stage/grade assignment
        
        if return_details:
            self.logger.info(f"Match calculation completed - Final score: {final_score}")
            return final_score, match_details
        self.logger.info(f"Match calculation completed - Final score: {final_score}")
        return final_score

    def get_match_description(self, score):
        """Get human-readable match description"""
        if score >= 90:
            return "Excellent match"
        elif score >= 75:
            return "Strong match"
        elif score >= 60:
            return "Good match"
        elif score >= 40:
            return "Partial match"
        else:
            return "Low match"