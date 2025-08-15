import logging
from typing import Dict, Any, List

class BreastCancerProfile:
    """
    Specialized mCODE profile for breast cancer patients
    Handles mapping and validation of breast cancer-specific features
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_biomarkers = ['ER', 'PR', 'HER2']
        self.common_genes = ['BRCA1', 'BRCA2', 'PIK3CA', 'TP53', 'HER2']
        
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate breast cancer features against mCODE requirements
        Returns validated features and any issues found
        """
        issues = []
        validated = features.copy()
        
        # Validate biomarkers
        biomarkers = features.get('biomarkers', [])
        validated['biomarkers'] = self._validate_biomarkers(biomarkers, issues)
        
        # Validate genomic variants
        variants = features.get('genomic_variants', [])
        validated['genomic_variants'] = self._validate_variants(variants, issues)
        
        # Validate cancer characteristics
        cancer_chars = features.get('cancer_characteristics', {})
        validated['cancer_characteristics'] = self._validate_cancer_chars(cancer_chars, issues)
        
        return {
            'validated_features': validated,
            'issues': issues,
            'is_complete': len(issues) == 0
        }
    
    def _validate_biomarkers(self, biomarkers: List[Dict], issues: List[str]) -> List[Dict]:
        """Validate breast cancer biomarkers"""
        valid_biomarkers = []
        found_biomarkers = set()
        
        for biomarker in biomarkers:
            name = biomarker.get('name', '').upper()
            status = biomarker.get('status', '').lower()
            
            # Skip invalid biomarkers
            if not name or not status:
                continue
                
            # Track found biomarkers
            if name in self.required_biomarkers:
                found_biomarkers.add(name)
                
            valid_biomarkers.append({
                'name': name,
                'status': status,
                'value': biomarker.get('value', '')
            })
        
        # Check for missing required biomarkers
        missing = set(self.required_biomarkers) - found_biomarkers
        if missing:
            issues.append(f"Missing required biomarkers: {', '.join(missing)}")
            
        return valid_biomarkers
    
    def _validate_variants(self, variants: List[Dict], issues: List[str]) -> List[Dict]:
        """Validate genomic variants with breast cancer focus"""
        valid_variants = []
        
        for variant in variants:
            gene = variant.get('gene', '')
            change = variant.get('variant', '')
            
            if not gene or not change:
                continue
                
            # Check if gene is breast cancer relevant
            if gene.upper() not in self.common_genes:
                self.logger.info(f"Non-breast cancer gene detected: {gene}")
                
            valid_variants.append({
                'gene': gene.upper(),
                'variant': change,
                'significance': variant.get('significance', 'unknown')
            })
            
        return valid_variants
    
    def _validate_cancer_chars(self, cancer_chars: Dict, issues: List[str]) -> Dict:
        """Validate breast cancer characteristics"""
        validated = cancer_chars.copy()
        
        # Validate TNM staging format
        stage = validated.get('stage', '')
        if stage and not any(x in stage for x in ['T', 'N', 'M']):
            issues.append(f"Invalid TNM stage format: {stage}")
            
        # Validate metastasis sites
        mets = validated.get('metastasis_sites', [])
        if mets and not isinstance(mets, list):
            validated['metastasis_sites'] = []
            issues.append("Metastasis sites must be a list")
            
        return validated
    
    def to_fhir(self, features: Dict[str, Any]) -> Dict:
        """Convert breast cancer features to FHIR resources"""
        # Implementation would go here
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }