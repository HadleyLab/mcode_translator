"""
Patient-Trial Matching Algorithm for mCODE Translator
Calculates match scores between patient profiles and clinical trials
"""

class PatientMatcher:
    def __init__(self):
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
        
        # Cancer type match (30% weight)
        cancer_match = patient_profile['cancer_type'] == trial_features.get('cancer_type', '')
        if cancer_match:
            score += 30
            match_details['cancer_type'] = True
            
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
        
        for biomarker, status in patient_biomarkers.items():
            weight = self.biomarker_weights.get(biomarker, 0.5)
            if biomarker in trial_biomarkers:
                if status == trial_biomarkers[biomarker]:
                    score += 8 * weight  # 8% per matching biomarker
                    match_details['biomarkers'] = True
                max_score += 8
        
        # Genomic variant matches (20% weight)
        patient_variants = patient_profile.get('genomic_variants', [])
        trial_variants = trial_features.get('genomic_variants', [])
        
        for variant in patient_variants:
            weight = self.variant_weights.get(variant, 0.5)
            if variant in trial_variants:
                score += 4 * weight  # 4% per matching variant
                match_details['genomic_variants'] = True
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
        
        if patient_stage >= trial_min_stage and patient_grade >= trial_min_grade:
            score += 10
            match_details['stage_grade'] = True
        
        final_score = min(100, int(score * 100 / max_score)) if max_score > 0 else 0
        
        # Remove duplicate stage/grade assignment
        
        if return_details:
            return final_score, match_details
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