"""Utility functions for standardizing NLP feature extraction results.

Provides common functionality for processing and standardizing:
- Biomarker results
- Genomic variants
- Demographic information
- Cancer characteristics
"""

from typing import Dict, List, Union

def standardize_features(features: Dict) -> Dict:
    """Ensure features have consistent structure across NLP engines.
    
    Parameters
    ----------
    features : Dict
        Raw extracted features from NLP processing
        
    Returns
    -------
    Dict
        Standardized features with consistent structure:
        {
            'demographics': Dict,
            'cancer_characteristics': Dict,
            'biomarkers': List[Dict],
            'genomic_variants': List[Dict],
            'treatment_history': Dict,
            'performance_status': Dict
        }
    """
    return {
        'demographics': features.get('demographics', {}),
        'cancer_characteristics': features.get('cancer_characteristics', {}),
        'biomarkers': standardize_biomarkers(features.get('biomarkers', [])),
        'genomic_variants': standardize_variants(features.get('genomic_variants', [])),
        'treatment_history': features.get('treatment_history', {}),
        'performance_status': features.get('performance_status', {})
    }

def standardize_biomarkers(biomarkers: Union[List, Dict]) -> List[Dict]:
    """Convert biomarkers to standardized list format.
    
    Parameters
    ----------
    biomarkers : Union[List, Dict]
        Biomarkers in either list or dict format
        
    Returns
    -------
    List[Dict]
        Standardized biomarker list with structure:
        [{
            'name': str,
            'status': str,
            'value': str (optional)
        }]
    """
    if isinstance(biomarkers, dict):
        return [{'name': k, 'status': v} for k,v in biomarkers.items()]
    return biomarkers

def standardize_variants(variants: List[Dict]) -> List[Dict]:
    """Ensure variants have required fields.
    
    Parameters
    ----------
    variants : List[Dict]
        Raw extracted variants
        
    Returns
    -------
    List[Dict]
        Standardized variants with structure:
        [{
            'gene': str,
            'variant': str,
            'significance': str
        }]
    """
    return [{
        'gene': v.get('gene', ''),
        'variant': v.get('variant', ''),
        'significance': v.get('significance', '')
    } for v in variants]