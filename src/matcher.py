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
    
    def calculate_match_score(self, patient_profile, trial_features):
        """
        Calculate match score between patient profile and trial features
        
        Args:
            patient_profile: Dict with patient characteristics
            trial_features: Dict with trial genomic features
            
        Returns:
            Float match score (0-100)
        """
        score = 0
        max_score = 0
        
        # Cancer type match (30% weight)
        cancer_match = patient_profile['cancer_type'] == trial_features.get('cancer_type', '')
        if cancer_match:
            score += 30
            
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
                max_score += 8
        
        # Genomic variant matches (20% weight)
        patient_variants = patient_profile.get('genomic_variants', [])
        trial_variants = trial_features.get('genomic_variants', [])
        
        for variant in patient_variants:
            weight = self.variant_weights.get(variant, 0.5)
            if variant in trial_variants:
                score += 4 * weight  # 4% per matching variant
            max_score += 4
        
        # Stage compatibility (10% weight)
        stage_mapping = {'I':1, 'II':2, 'III':3, 'IV':4}
        # Get stage from cancer_characteristics
        patient_stage = stage_mapping.get(
            patient_profile.get('cancer_characteristics', {}).get('stage', ''),
            0
        )
        trial_min_stage = stage_mapping.get(trial_features.get('min_stage', 'I'), 1)
        
        if patient_stage >= trial_min_stage:
            score += 10
        
        return min(100, int(score * 100 / max_score)) if max_score > 0 else 0

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