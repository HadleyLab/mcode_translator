import re
from typing import Dict, List, Any
import logging
from .nlp_engine import NLPEngine, ProcessingResult
from src.utils.pattern_config import (
    BIOMARKER_PATTERNS,
    GENE_PATTERN,
    VARIANT_PATTERN,
    COMPLEX_VARIANT_PATTERN,
    STAGE_PATTERN,
    CANCER_TYPE_PATTERN,
    ECOG_PATTERN,
    GENDER_PATTERN,
    AGE_PATTERN,
    CONDITION_PATTERN
)

class RegexNLPEngine(NLPEngine):
    """Regex-based NLP engine for clinical text processing.
    
    Specialized for extracting mCODE features using regular expression patterns.
    Handles common clinical trial eligibility criteria patterns.
    
    Inherits from:
        NLPEngine (base class with common functionality)
    
    Attributes
    ----------
    biomarker_patterns : Dict[str, Pattern]
        Centralized regex patterns for biomarkers (imported from pattern_config)
    gene_pattern : Pattern
        Centralized regex for gene mentions (imported from pattern_config)
    variant_pattern : Pattern
        Centralized regex for variant descriptions (imported from pattern_config)
    complex_variant_pattern : Pattern
        Centralized regex for protein-level variants (imported from pattern_config)
    stage_pattern : Pattern
        Centralized regex for cancer staging (imported from pattern_config)
    cancer_type_pattern : Pattern
        Centralized regex for cancer types (imported from pattern_config)
    ecog_pattern : Pattern
        Centralized regex for ECOG scores (imported from pattern_config)
    gender_pattern : Pattern
        Centralized regex for gender references (imported from pattern_config)
    age_pattern : Pattern
        Centralized regex for age ranges (imported from pattern_config)
    logger : logging.Logger
        Configured logger instance from base class
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize patterns from centralized config
        self.biomarker_patterns = BIOMARKER_PATTERNS
        self.gene_pattern = GENE_PATTERN
        self.variant_pattern = VARIANT_PATTERN
        self.complex_variant_pattern = COMPLEX_VARIANT_PATTERN
        self.stage_pattern = STAGE_PATTERN
        self.cancer_type_pattern = CANCER_TYPE_PATTERN
        self.ecog_pattern = ECOG_PATTERN
        self.gender_pattern = GENDER_PATTERN
        self.age_pattern = AGE_PATTERN
        self.condition_pattern = CONDITION_PATTERN

    def process_text(self, text: str) -> ProcessingResult:
        """Process clinical text and extract mCODE features using regex patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process. Must be non-empty.
            
        Returns
        -------
        ProcessingResult
            Contains standardized extraction results with fields:
            - features: Extracted mCODE features (standardized format)
            - mcode_mappings: Placeholder for future FHIR mappings
            - metadata: Processing metadata including counts
            - entities: Raw extracted entities with source text
            - error: None if successful, error message if failed
            
        Raises
        ------
        ValueError
            If input text is empty or invalid type
            
        Notes
        -----
        Uses the following internal methods:
        - _extract_demographics(): For age/gender extraction
        - _extract_cancer_condition(): For cancer type/stage
        - _extract_biomarkers(): For ER/PR/HER2 status
        - _extract_genomic_variants(): For gene mutations
        - _extract_performance_status(): For ECOG scores
            
        Examples
        --------
        >>> engine = RegexNLPEngine()
        >>> result = engine.process_text("ER+ HER2- breast cancer")
        >>> "ER" in [b['name'] for b in result.features['biomarkers']]
        True
        """
        if not text or not isinstance(text, str):
            error_msg = "Input text must be a non-empty string"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
            
        try:
            import time
            start_time = time.time()
            
            # Extract additional conditions
            additional_conditions = self._extract_additional_conditions(text)
            
            features = {
                'demographics': self._extract_demographics(text),
                'cancer_characteristics': self._extract_cancer_condition(text),
                'biomarkers': self._extract_biomarkers(text),
                'genomic_variants': self._extract_genomic_variants(text),
                'performance_status': self._extract_performance_status(text),
                'treatment_history': {},  # Placeholder for future implementation
                'additional_conditions': additional_conditions
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                features=features,
                mcode_mappings={},  # Placeholder for future FHIR mappings
                metadata={
                    'processing_time': processing_time,
                    'engine': 'regex',
                    'biomarkers_count': len(features['biomarkers']),
                    'genomic_variants_count': len(features['genomic_variants'])
                },
                entities=self._collect_entities(features),
                error=None
            )
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg)

    def _extract_demographics(self, text: str) -> Dict[str, Any]:
        """Extract demographic information from text using regex patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted demographics:
            - gender: str or None (from gender_pattern)
            - age_range: str or None (from age_pattern)
            
        See Also
        --------
        pattern_config.GENDER_PATTERN : Centralized regex for gender extraction
        pattern_config.AGE_PATTERN : Centralized regex for age range extraction
        """
        demographics = {}
        
        # Extract gender
        gender_match = self.gender_pattern.search(text)
        if gender_match:
            demographics['gender'] = gender_match.group(1).capitalize()
            
        # Extract age range
        age_match = self.age_pattern.search(text)
        if age_match:
            demographics['age_range'] = f"{age_match.group(1)}-{age_match.group(2)}"
            
        return demographics

    def _extract_cancer_condition(self, text: str) -> Dict[str, Any]:
        """Extract cancer type and stage information using regex patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted cancer characteristics:
            - cancer_type: str or None (from cancer_type_pattern)
            - stage: str or None (from stage_pattern)
            
        See Also
        --------
        pattern_config.CANCER_TYPE_PATTERN : Centralized regex for cancer types
        pattern_config.STAGE_PATTERN : Centralized regex for cancer stages
        
        Notes
        -----
        Currently supports detection of:
        - Breast cancer
        - Lung cancer
        - Colorectal cancer
        Stages I-IV (including substages)
        """
        condition = {}
        
        # Extract all cancer types
        cancer_matches = list(self.cancer_type_pattern.finditer(text))
        if cancer_matches:
            # Take the first match for backward compatibility
            first_match = cancer_matches[0]
            condition['cancer_type'] = first_match.group(1).capitalize() + " cancer"
            
            # Store all matches for entity extraction
            condition['all_cancer_types'] = [match.group(1).capitalize() + " cancer" for match in cancer_matches]
        else:
            condition['all_cancer_types'] = []
            
        # Extract stage
        stage_match = self.stage_pattern.search(text)
        if stage_match:
            condition['stage'] = stage_match.group(1).upper()
            
        return condition

    def _extract_additional_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract additional conditions beyond cancer types using regex patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        List[Dict[str, Any]]
            List of standardized condition results with:
            - name: str (condition name from condition_pattern)
            - source_text: str (original matched text)
        """
        conditions = []
        for match in self.condition_pattern.finditer(text):
            condition_name = match.group(1)
            # Only add conditions that are not already captured by cancer type extraction
            if condition_name.lower() not in ['breast', 'lung', 'colorectal']:
                conditions.append({
                    'name': condition_name,
                    'source_text': match.group()
                })
                
        # Deduplicate conditions
        unique_conditions = []
        seen = set()
        for cond in conditions:
            if cond['name'] not in seen:
                seen.add(cond['name'])
                unique_conditions.append(cond)
                
        return unique_conditions

    def _extract_biomarkers(self, text: str) -> List[Dict[str, Any]]:
        """Extract biomarker status information using predefined patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        List[Dict[str, Any]]
            List of standardized biomarker results with:
            - name: str (biomarker name from biomarker_patterns keys)
            - status: str (standardized positive/negative/expression)
            - source_text: str (original matched text)
            
        See Also
        --------
        pattern_config.BIOMARKER_PATTERNS : Centralized dictionary of regex patterns
        
        Notes
        -----
        Handles status standardization for:
        - Positive/Negative (multiple representations)
        - Percentage expressions (e.g., '30%')
        - IHC scores (e.g., '3+')
        - TMB (mutations/megabase)
        """
        biomarkers = []
        for name, pattern in self.biomarker_patterns.items():
            for match in pattern.finditer(text):
                value = match.group(1)
                
                # Standardize status values
                if value.lower() in ['positive', 'pos', '+']:
                    status = 'Positive'
                elif value.lower() in ['negative', 'neg', '-']:
                    status = 'Negative'
                elif '%' in value:
                    status = f"{value.strip()} expression"
                elif '+' in value:
                    status = f"{value.strip()} IHC"
                elif 'mut/Mb' in value:
                    status = f"{value.strip()} TMB"
                else:
                    status = value.capitalize()
                
                biomarkers.append({
                    'name': name,
                    'status': status,
                    'source_text': match.group()
                })
                
        # Deduplicate biomarkers
        unique_biomarkers = []
        seen = set()
        for bio in biomarkers:
            key = (bio['name'], bio['status'])
            if key not in seen:
                seen.add(key)
                unique_biomarkers.append(bio)
                
        return unique_biomarkers

    def _extract_genomic_variants(self, text: str) -> List[Dict[str, Any]]:
        """Extract genomic variants using multiple regex patterns.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        List[Dict[str, Any]]
            List of standardized variant results with:
            - gene: str (detected gene symbol)
            - variant_type: str (mutation/amplification/fusion/etc.)
            - variant: str (specific variant if available)
            - source_text: str (original matched text)
            
        See Also
        --------
        pattern_config.GENE_PATTERN : Centralized regex for gene mentions
        pattern_config.VARIANT_PATTERN : Centralized regex for variant descriptions
        pattern_config.COMPLEX_VARIANT_PATTERN : Centralized regex for protein changes
        
        Notes
        -----
        Handles three types of variant patterns:
        1. Simple gene mentions (e.g., "BRCA1 mutation")
        2. Variant descriptions (e.g., "EGFR amplification")
        3. Protein-level changes (e.g., "BRAF p.Val600Glu")
        """
        variants = []
        
        # Extract simple gene mentions
        for match in self.gene_pattern.finditer(text):
            variants.append({
                'gene': match.group(1).upper(),
                'variant_type': 'gene_mention',
                'variant': '',
                'source_text': match.group()
            })
            
        # Extract variant descriptions
        for match in self.variant_pattern.finditer(text):
            variants.append({
                'gene': match.group(1).upper(),
                'variant_type': match.group(1).lower() + '_variant',
                'variant': '',
                'source_text': match.group()
            })
            
        # Extract protein-level variants
        for match in self.complex_variant_pattern.finditer(text):
            variants.append({
                'gene': match.group(1).upper(),
                'variant_type': 'protein_change',
                'variant': match.group(2),
                'source_text': match.group()
            })
            
        # Deduplicate variants
        unique_variants = []
        seen = set()
        for var in variants:
            key = (var['gene'], var['variant_type'], var['variant'])
            if key not in seen:
                seen.add(key)
                unique_variants.append(var)
                
        return unique_variants

    def _extract_performance_status(self, text: str) -> Dict[str, Any]:
        """Extract performance status (ECOG score) from text.
        
        Parameters
        ----------
        text : str
            Input clinical text to process
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - ecog_score: str or None (0-4 from ecog_pattern)
            
        See Also
        --------
        pattern_config.ECOG_PATTERN : Centralized regex for ECOG scores
        
        Notes
        -----
        Only extracts the first ECOG score found in text.
        Scores are returned as strings (e.g., "1").
        """
        status = {}
        
        # Extract ECOG score
        ecog_match = self.ecog_pattern.search(text)
        if ecog_match:
            status['ecog_score'] = ecog_match.group(1)
            
        return status

    def _collect_entities(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert extracted features into standardized entity format.
        
        Parameters
        ----------
        features : Dict[str, Any]
            Dictionary containing extracted features from:
            - _extract_biomarkers()
            - _extract_genomic_variants()
            
        Returns
        -------
        List[Dict[str, Any]]
            List of standardized entities with:
            - type: str ('biomarker' or 'genomic_variant')
            - text: str (original matched text)
            - label: str (biomarker name or gene symbol)
            - value: str (status or variant description)
            
        Notes
        -----
        Used to prepare extracted features for:
        - Downstream processing
        - Result visualization
        - FHIR resource generation
        """
        entities = []
        
        # Add conditions/cancer types
        cancer_characteristics = features.get('cancer_characteristics', {})
        all_cancer_types = cancer_characteristics.get('all_cancer_types', [])
        if all_cancer_types:
            for cancer_type in all_cancer_types:
                entities.append({
                    'type': 'CONDITION',
                    'text': cancer_type.lower(),
                    'label': 'cancer_type',
                    'value': cancer_type
                })
        elif cancer_characteristics.get('cancer_type'):
            cancer_type = cancer_characteristics['cancer_type']
            entities.append({
                'type': 'CONDITION',
                'text': cancer_type.lower(),
                'label': 'cancer_type',
                'value': cancer_type
            })
        
        # Add additional conditions
        additional_conditions = features.get('additional_conditions', [])
        for condition in additional_conditions:
            entities.append({
                'type': 'CONDITION',
                'text': condition['name'].lower(),
                'label': 'additional_condition',
                'value': condition['name']
            })
        
        # Add biomarkers
        for bio in features.get('biomarkers', []):
            entities.append({
                'type': 'biomarker',
                'text': bio['source_text'],
                'label': bio['name'],
                'value': bio['status']
            })
            
        # Add genomic variants
        for var in features.get('genomic_variants', []):
            entities.append({
                'type': 'genomic_variant',
                'text': var['source_text'],
                'label': var['gene'],
                'value': var['variant'] or var['variant_type']
            })
            
        return entities