import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeExtractionModule:
    """
    Code Extraction Module for mCODE Translator
    Identifies and validates medical codes (ICD-10-CM, CPT, LOINC, RxNorm) from clinical trial eligibility criteria
    """
    
    def __init__(self):
        """
        Initialize the Code Extraction Module
        """
        logger.info("Code Extraction Module initialized")
        
        # Define code format patterns
        self.code_patterns = {
            'ICD10CM': r'\b[A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?\b',
            'CPT': r'(?<![-\d])\b\d{5}\b(?![\d-])',
            'LOINC': r'\b[A-Z0-9]+-\d+\b',
            'RxNorm': r'\b\d+\b'
        }
        
        # Define extended patterns for code references with prefixes
        self.extended_patterns = {
            'ICD10CM': r'\bICD[-\s]?10(?:[-\s]?CM)?[-\s]?([A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?)\b',
            'LOINC': r'\bLOINC[-\s]?(\d+-\d+)\b',
            'RxNorm': r'\bR[x]?[-\s]?(\d+)\b'
        }
        
        # Sample valid codes for prototype validation
        self.valid_codes = {
            'ICD10CM': ['C50.911', 'C34.90', 'C18.9', 'E11.9', 'I25.9'],
            'CPT': ['12345', '67890', '54321', '99213', '99214', '19303'],
            'LOINC': ['12345-6', '78901-2', '34567-8', '8867-4', '2345-7', 'LP417347-6', 'LP417348-4', 'LP417351-8'],
            'RxNorm': ['123456', '789012', '345678', '987654', '456789']
        }
        
        # Sample mCODE required codes
        self.mcode_required_codes = {
            'ICD10CM': ['C50.911', 'C34.90'],
            'CPT': ['12345', '67890']
        }
        
        # Sample cross-walk mappings
        self.cross_walks = {
            ('C50.911', 'ICD10CM', 'SNOMEDCT'): '254837009',
            ('C50.911', 'ICD10CM', 'LOINC'): 'LP12345-6',
            ('C34.90', 'ICD10CM', 'SNOMEDCT'): '254838004'
        }
        
        # Code hierarchy information
        self.hierarchies = {
            'ICD10CM': {
                'C50.911': {
                    'parent': 'C50',
                    'children': ['C50.9111', 'C50.9112']
                }
            }
        }
    
    def identify_icd10cm_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify ICD-10-CM codes in text
        
        Args:
            text: Input text to search for codes
            
        Returns:
            List of identified ICD-10-CM codes with metadata
        """
        codes = []
        
        # Direct pattern matching
        direct_matches = re.finditer(self.code_patterns['ICD10CM'], text, re.IGNORECASE)
        for match in direct_matches:
            code = match.group()
            # Validate format
            if self.validate_code_format(code, 'ICD10CM'):
                codes.append({
                    'text': match.group(),
                    'code': code,
                    'system': 'ICD-10-CM',
                    'start': match.start(),
                    'end': match.end(),
                    'direct_reference': True
                })
        
        # Extended pattern matching (with prefixes)
        extended_matches = re.finditer(self.extended_patterns['ICD10CM'], text, re.IGNORECASE)
        for match in extended_matches:
            code = match.group(1) if match.groups() else match.group()
            if self.validate_code_format(code, 'ICD10CM'):
                codes.append({
                    'text': match.group(),
                    'code': code,
                    'system': 'ICD-10-CM',
                    'start': match.start(),
                    'end': match.end(),
                    'direct_reference': False
                })
        
        return codes
    
    def identify_cpt_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify CPT codes in text
        
        Args:
            text: Input text to search for codes
            
        Returns:
            List of identified CPT codes with metadata
        """
        codes = []
        
        # Pattern matching for 5-digit codes
        matches = re.finditer(self.code_patterns['CPT'], text)
        for match in matches:
            code = match.group()
            # Validate that it's a valid CPT code
            if self.validate_code_existence(code, 'CPT'):
                codes.append({
                    'text': match.group(),
                    'code': code,
                    'system': 'CPT',
                    'start': match.start(),
                    'end': match.end(),
                    'direct_reference': True
                })
        
        return codes
    
    def identify_loinc_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify LOINC codes in text
        
        Args:
            text: Input text to search for codes
            
        Returns:
            List of identified LOINC codes with metadata
        """
        codes = []
        
        # Direct pattern matching
        direct_matches = re.finditer(self.code_patterns['LOINC'], text)
        for match in direct_matches:
            code = match.group()
            codes.append({
                'text': match.group(),
                'code': code,
                'system': 'LOINC',
                'start': match.start(),
                'end': match.end(),
                'direct_reference': True
            })
        
        # Extended pattern matching (with prefixes)
        extended_matches = re.finditer(self.extended_patterns['LOINC'], text, re.IGNORECASE)
        for match in extended_matches:
            code = match.group(1) if match.groups() else match.group()
            codes.append({
                'text': match.group(),
                'code': code,
                'system': 'LOINC',
                'start': match.start(),
                'end': match.end(),
                'direct_reference': False
            })
        
        # Validate all codes
        validated_codes = []
        for code_info in codes:
            if self.validate_code_existence(code_info['code'], 'LOINC'):
                validated_codes.append(code_info)
            else:
                # For prototype, we'll still include unvalidated codes but mark them
                code_info['validated'] = False
                validated_codes.append(code_info)
        
        return validated_codes
    
    def identify_rxnorm_codes(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify RxNorm codes in text
        
        Args:
            text: Input text to search for codes
            
        Returns:
            List of identified RxNorm codes with metadata
        """
        codes = []
        
        # Direct pattern matching (numeric codes)
        direct_matches = re.finditer(self.code_patterns['RxNorm'], text)
        for match in direct_matches:
            code = match.group()
            # Additional validation for RxNorm codes
            if len(code) >= 4:  # RxNorm codes are typically longer
                codes.append({
                    'text': match.group(),
                    'code': code,
                    'system': 'RxNorm',
                    'start': match.start(),
                    'end': match.end(),
                    'direct_reference': True
                })
        
        # Extended pattern matching (with prefixes)
        extended_matches = re.finditer(self.extended_patterns['RxNorm'], text, re.IGNORECASE)
        for match in extended_matches:
            code = match.group(1) if match.groups() else match.group()
            codes.append({
                'text': match.group(),
                'code': code,
                'system': 'RxNorm',
                'start': match.start(),
                'end': match.end(),
                'direct_reference': False
            })
        
        # Validate all codes
        validated_codes = []
        for code_info in codes:
            if self.validate_code_existence(code_info['code'], 'RxNorm'):
                validated_codes.append(code_info)
            else:
                # For prototype, we'll still include unvalidated codes but mark them
                code_info['validated'] = False
                validated_codes.append(code_info)
        
        return validated_codes
    
    def identify_all_codes(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify all types of codes in text
        
        Args:
            text: Input text to search for codes
            
        Returns:
            Dictionary with all identified codes by system
        """
        all_codes = {
            'ICD10CM': self.identify_icd10cm_codes(text),
            'CPT': self.identify_cpt_codes(text),
            'LOINC': self.identify_loinc_codes(text),
            'RxNorm': self.identify_rxnorm_codes(text)
        }
        
        return all_codes
    
    def validate_code_format(self, code: str, system: str) -> bool:
        """
        Validate code format for a specific system
        
        Args:
            code: Code to validate
            system: Coding system (ICD10CM, CPT, LOINC, RxNorm)
            
        Returns:
            True if format is valid, False otherwise
        """
        format_rules = {
            'ICD10CM': r'^[A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?$',
            'CPT': r'^\d{5}$',
            'LOINC': r'^[A-Z0-9]+-\d+$',
            'RxNorm': r'^\d+$'
        }
        
        if system in format_rules:
            pattern = format_rules[system]
            return bool(re.match(pattern, code))
        
        return False
    
    def validate_code_existence(self, code: str, system: str) -> bool:
        """
        Validate that a code exists in the system
        
        Args:
            code: Code to validate
            system: Coding system
            
        Returns:
            True if code exists, False otherwise
        """
        # For prototype, use sample valid codes
        if system in self.valid_codes:
            return code in self.valid_codes[system]
        
        return True  # Default to valid if not in sample set
    
    def validate_mcode_compliance(self, code: str, system: str) -> bool:
        """
        Validate that a code is mCODE compliant
        
        Args:
            code: Code to validate
            system: Coding system
            
        Returns:
            True if code is mCODE compliant, False otherwise
        """
        # Check if code is part of mCODE required or recommended codes
        if system in self.mcode_required_codes:
            return code in self.mcode_required_codes[system]
        
        return True  # Default to compliant if not specifically required
    
    def map_between_systems(self, source_code: str, source_system: str, target_system: str) -> Optional[str]:
        """
        Map a code from one system to another
        
        Args:
            source_code: Code to map
            source_system: Source coding system
            target_system: Target coding system
            
        Returns:
            Mapped code or None if no mapping found
        """
        key = (source_code, source_system, target_system)
        if key in self.cross_walks:
            return self.cross_walks[key]
        
        return None  # No mapping found
    
    def get_code_hierarchy(self, code: str, system: str) -> Dict[str, Any]:
        """
        Get parent/child relationships for a code
        
        Args:
            code: Code to get hierarchy for
            system: Coding system
            
        Returns:
            Dictionary with hierarchy information
        """
        if system in self.hierarchies and code in self.hierarchies[system]:
            return self.hierarchies[system][code]
        
        return {}
    
    def calculate_code_confidence(self, code_info: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a code
        
        Args:
            code_info: Code information dictionary
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Increase for direct code references
        if code_info.get('direct_reference'):
            confidence += 0.3
        
        # Increase for validated codes
        if code_info.get('validated', True):  # Default to validated
            confidence += 0.2
        
        # Increase for mCODE compliance
        if code_info.get('validation', {}).get('mcode_compliant', False):
            confidence += 0.1
        
        # Decrease for potentially ambiguous terms
        if code_info.get('ambiguous'):
            confidence -= 0.3
        
        # Handle invalid or retired codes
        if code_info.get('status') == 'retired':
            confidence = 0  # Don't use retired codes
        
        # Adjust based on code length (longer codes are often more specific)
        code = code_info.get('code', '')
        if len(code) > 5:
            confidence += 0.1
        elif len(code) < 3:
            confidence -= 0.1
        
        # Adjust based on context (if available)
        text = code_info.get('text', '')
        if 'confirmed' in text.lower() or 'diagnosed' in text.lower():
            confidence += 0.1
        elif 'suspected' in text.lower() or 'possible' in text.lower():
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    def extract_codes_from_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract codes from recognized medical entities including biomarkers and genomic variants
        """
        """
        Extract codes from recognized medical entities
        
        Args:
            entities: List of recognized medical entities
            
        Returns:
            List of extracted codes
        """
        # Enhanced term to code mapping with medical terms and biomarkers
        term_to_code_map = {
            # Biomarkers
            'ER positive': {'LOINC': '16112-5'},
            'PR positive': {'LOINC': '16113-3'},
            'HER2 positive': {'LOINC': '48676-1'},
            'BRCA mutation': {'HGNC': '1100'},
            'PD-L1 positive': {'LOINC': '82397-3'},
            'Ki-67 high': {'LOINC': '85337-4'},
            
            # Genomic variants
            'PIK3CA mutation': {'HGNC': '8985'},
            'TP53 mutation': {'HGNC': '11998'},
            'ESR1 mutation': {'HGNC': '3467'},
            # Cancers
            'breast cancer': {
                'ICD10CM': 'C50.911',
                'SNOMEDCT': '254837009'
            },
            'lung cancer': {
                'ICD10CM': 'C34.90',
                'SNOMEDCT': '254838004'
            },
            'colorectal cancer': {
                'ICD10CM': 'C18.9',
                'SNOMEDCT': '363346000'
            },
            'prostate cancer': {
                'ICD10CM': 'C61',
                'SNOMEDCT': '399068003'
            },
            'melanoma': {
                'ICD10CM': 'C43.9',
                'SNOMEDCT': '372130007'
            },
            'leukemia': {
                'ICD10CM': 'C91.9',
                'SNOMEDCT': '93143009'
            },
            'lymphoma': {
                'ICD10CM': 'C80.9',
                'SNOMEDCT': '93143009'
            },
            
            # Treatments
            'chemotherapy': {
                'CPT': '12345',
                'SNOMEDCT': '367336001'
            },
            'radiation therapy': {
                'CPT': '67890',
                'SNOMEDCT': '128934006'
            },
            'surgery': {
                'CPT': '10021',
                'SNOMEDCT': '387713003'
            },
            'immunotherapy': {
                'CPT': '96405',
                'SNOMEDCT': '61439004'
            },
            
            # Medications
            'paclitaxel': {
                'RxNorm': '123456',
                'SNOMEDCT': '386906001'
            },
            'doxorubicin': {
                'RxNorm': '789012',
                'SNOMEDCT': '386907005'
            },
            'trastuzumab': {
                'RxNorm': '224905',
                'SNOMEDCT': '386908000'
            },
            'tamoxifen': {
                'RxNorm': '10324',
                'SNOMEDCT': '386909008'
            },
            
            # Conditions
            'diabetes': {
                'ICD10CM': 'E11.9',
                'SNOMEDCT': '73211009'
            },
            'hypertension': {
                'ICD10CM': 'I10',
                'SNOMEDCT': '38341003'
            },
            'heart disease': {
                'ICD10CM': 'I25.9',
                'SNOMEDCT': '56265001'
            },
            'stroke': {
                'ICD10CM': 'I63.9',
                'SNOMEDCT': '116288000'
            },
            
            # Procedures
            'mri': {
                'CPT': '70551',
                'LOINC': '39658-9'
            },
            'ct scan': {
                'CPT': '71250',
                'LOINC': '39659-7'
            },
            'blood test': {
                'CPT': '80053',
                'LOINC': '24357-6'
            },
            'biopsy': {
                'CPT': '10004',
                'SNOMEDCT': '73761001'
            }
        }
        
        mapped_codes = []
        for entity in entities:
            term = entity['text'].lower()
            # Try exact match first
            if term in term_to_code_map:
                codes = term_to_code_map[term]
                mapped_codes.append({
                    'entity': entity,
                    'codes': codes,
                    'confidence': entity.get('confidence', 0.8)
                })
            else:
                # Try partial match for more flexible matching
                for key, codes in term_to_code_map.items():
                    if key in term or term in key:
                        mapped_codes.append({
                            'entity': entity,
                            'codes': codes,
                            'confidence': entity.get('confidence', 0.6)  # Lower confidence for partial matches
                        })
                        break
        
        return mapped_codes
    
    def process_criteria_for_codes(self, criteria_text, entities: List[Dict[str, Any]] = None, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Process eligibility criteria text and extract codes with enhanced error handling
        
        Args:
            criteria_text: Text containing eligibility criteria
            entities: Optional pre-identified entities
            
        Returns:
            Dictionary containing extracted codes and metadata, or error information
        """
        result = {
            'extracted_codes': {},
            'mapped_entities': [],
            'metadata': {
                'errors': False
            }
        }
        
        try:
            # Handle list inputs
            if isinstance(criteria_text, list):
                criteria_text = ' '.join(str(item) for item in criteria_text)
            elif not isinstance(criteria_text, str):
                criteria_text = str(criteria_text)
            
            # Identify all codes in text with error context
            # Try to get from cache if key provided
            cached_result = None
            if cache_key and hasattr(self, 'cache_manager'):
                cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                identified_codes = cached_result
            else:
                try:
                    identified_codes = self.identify_all_codes(criteria_text)
                    if cache_key and hasattr(self, 'cache_manager'):
                        self.cache_manager.set(cache_key, identified_codes)
                except Exception as e:
                    self.logger.error(f"Code identification failed: {str(e)}")
                    identified_codes = {}
        
            # Validate codes with error handling
            validated_codes = {}
            for system, codes in identified_codes.items():
                validated_codes[system] = []
                for code in codes:
                    try:
                        # Add validation information
                        code['validation'] = {
                            'format_valid': self.validate_code_format(code['code'], system.replace('-', '')),
                            'exists': self.validate_code_existence(code['code'], system.replace('-', '')),
                            'mcode_compliant': self.validate_mcode_compliance(code['code'], system.replace('-', ''))
                        }
                    except Exception as e:
                        self.logger.error(f"Validation failed for code {code['code']}: {str(e)}")
                        code['validation'] = {
                            'error': str(e),
                            'valid': False
                        }
                    
                    # Add mapped codes
                    code['mapped_codes'] = {}
                    for target_system in ['SNOMEDCT', 'LOINC']:
                        mapped_code = self.map_between_systems(
                            code['code'],
                            system.replace('-', ''),
                            target_system
                        )
                        if mapped_code:
                            code['mapped_codes'][target_system] = mapped_code
                    
                    # Add confidence score
                    code['confidence'] = self.calculate_code_confidence(code)
                    
                    validated_codes[system].append(code)
            
            # Extract codes from entities if provided
            mapped_entities = []
            if entities:
                try:
                    mapped_entities = self.extract_codes_from_entities(entities)
                except Exception as e:
                    self.logger.error(f"Entity code extraction failed: {str(e)}")
                    mapped_entities = []
            
            # Create successful result structure
            result = {
                'extracted_codes': validated_codes,
                'mapped_entities': mapped_entities,
                'metadata': {
                    'total_codes': sum(len(codes) for codes in validated_codes.values()),
                    'systems_found': list(validated_codes.keys()),
                    'errors': False
                }
            }

        except Exception as e:
            self.logger.error(f"Criteria processing failed: {str(e)}")
            result = {
                'error': str(e),
                'metadata': {
                    'errors': True,
                    'error_details': str(e)
                }
            }
            
        return result
        
        return result


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    code_extractor = CodeExtractionModule()
    
    # Sample criteria text with codes
    sample_text = """
    INCLUSION CRITERIA:
    - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
    - Must have received prior chemotherapy treatment (CPT: 12345)
    - Laboratory values within normal limits (LOINC: 12345-6)
    - Currently taking medication (RxNorm: 123456)
    
    EXCLUSION CRITERIA:
    - History of other malignancies within the past 5 years (ICD-10-CM: C18.9)
    """
    
    # Process the sample text
    result = code_extractor.process_criteria_for_codes(sample_text)
    
    print("Code extraction complete. Found:")
    print(f"- {result['metadata']['total_codes']} total codes")
    print(f"- Systems: {', '.join(result['metadata']['systems_found'])}")
    
    # Print identified codes
    for system, codes in result['extracted_codes'].items():
        if codes:
            print(f"\n{system} codes:")
            for code in codes:
                print(f"  - {code['code']} (confidence: {code['confidence']:.2f})")