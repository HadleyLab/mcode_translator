import re
from typing import Dict, List, Any, Optional
from src.utils.logging_config import Loggable

class CriteriaParser(Loggable):
    """
    Component for parsing and structuring eligibility criteria from clinical trial records
    """
    
    def __init__(self):
        """
        Initialize the Criteria Parser
        """
        super().__init__()
        self.logger.info("Criteria Parser initialized")
        
        # Define patterns for criteria section identification
        self.section_patterns = {
            'inclusion': [
                r'inclusion criteria',
                r'eligible subjects',
                r'selection criteria',
                r'patient selection',
                r'entry criteria'
            ],
            'exclusion': [
                r'exclusion criteria',
                r'ineligible subjects',
                r'non[-\s]inclusion criteria',
                r'exclusionary criteria',
                r'contraindications'
            ]
        }
        
        # Define patterns for structured elements
        self.structured_patterns = {
            'age': [
                r'(?:age|aged?)\s*(?:of\s*)?(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?',
                r'(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?\s*(?:of\s+)?age',
                r'(?:between|from)\s+(\d+)\s*(?:and|to)\s*(\d+)'
            ],
            'gender': [
                r'\b(male|men)\b',
                r'\b(female|women)\b',
                r'\b(pregnant|nursing|breast[-\s]?feeding)\b'
            ],
            'performance_status': [
                r'ECOG\s*(?:performance\s+status\s+)?(\d)',
                r'WHO\s*[-\s]?\s*(\d)',
                r'Karnofsky\s*(?:performance\s+status\s+)?[>]?(\d{2})'
            ],
            'lab_values': [
                r'(?:WBC|Hemoglobin|Platelets|ANC|Creatinine|ALT|AST|Bilirubin|LDH|Albumin|INR|PTT|aPTT|CrCl|eGFR|TSH|T3|T4|PSA|CA-125|CEA|AFP|HCG|HbA1c|Glucose|Potassium|Sodium|Calcium|Magnesium|Phosphorus|BUN|pH|pCO2|pO2|HCO3|O2 saturation|O2 sat|SpO2|ECMO|CPAP|BiPAP|NIPPV|NIV|IMV|MV|Ventilator|ECMO|CPB|CABG|PCI|PTCA)\b'
            ],
            'biomarkers': [
                r'(ER|PR|HER2|HR)[\s\-\+]*(?:positive|negative|\+|\-)',
                r'(estrogen receptor|progesterone receptor|human epidermal growth factor receptor 2|hormone receptor)[\s\-\+]*(?:positive|negative|\+|\-)'
            ]
        }
        
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse eligibility criteria text and extract structured information
        
        Args:
            text: Eligibility criteria text to parse
            
        Returns:
            Dictionary containing parsed information
        """
        result = {
            'inclusion': [],
            'exclusion': [],
            'entities': [],
            'demographics': {},
            'performance_status': {},
            'lab_values': []
        }
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Extract inclusion criteria
        inclusion_section = self._extract_section(text_lower, 'inclusion')
        if inclusion_section:
            result['inclusion'] = self._parse_criteria_list(inclusion_section)
            
        # Extract exclusion criteria
        exclusion_section = self._extract_section(text_lower, 'exclusion')
        if exclusion_section:
            result['exclusion'] = self._parse_criteria_list(exclusion_section)
            
        # Extract entities and structured data
        result['entities'] = self._extract_entities(text)
        result['demographics'] = self._extract_demographics(text)
        result['performance_status'] = self._extract_performance_status(text)
        result['lab_values'] = self._extract_lab_values(text)
        result['biomarkers'] = self._extract_biomarkers(text)
        
        return result
        
    def _extract_section(self, text: str, section_type: str) -> Optional[str]:
        """
        Extract a specific section from the criteria text
        
        Args:
            text: Full criteria text
            section_type: Type of section to extract ('inclusion' or 'exclusion')
            
        Returns:
            Extracted section text or None if not found
        """
        # Look for section headers
        for pattern in self.section_patterns[section_type]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find the start of the section
                start_pos = match.end()
                
                # Find the end of the section (next section header or end of text)
                next_section_patterns = '|'.join([
                    pattern for patterns in self.section_patterns.values() 
                    for pattern in patterns
                ])
                end_match = re.search(next_section_patterns, text[start_pos:], re.IGNORECASE)
                
                if end_match:
                    end_pos = start_pos + end_match.start()
                    return text[start_pos:end_pos]
                else:
                    return text[start_pos:]
                    
        return None
        
    def _parse_criteria_list(self, section_text: str) -> List[str]:
        """
        Parse a section of criteria into a list of individual criteria
        
        Args:
            section_text: Text containing criteria
            
        Returns:
            List of individual criteria
        """
        criteria = []
        
        # Split by common list indicators
        lines = re.split(r'[-•\d+\.)\n]+', section_text)
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                criteria.append(line)
                
        return criteria
        
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # This is a simplified implementation
        # In a real implementation, this would use NLP models
        conditions = re.findall(r'(\w+\s+cancer|\w+\s+tumor|\w+\s+carcinoma)', text, re.IGNORECASE)
        for condition in conditions:
            entities.append({
                'text': condition,
                'type': 'CONDITION',
                'confidence': 0.9
            })
            
        medications = re.findall(r'(\w+[ae]zole|\w+cin|\w+vir|\w+nib|\w+umab)', text, re.IGNORECASE)
        for medication in medications:
            entities.append({
                'text': medication,
                'type': 'MEDICATION',
                'confidence': 0.8
            })
            
        return entities
        
    def _extract_demographics(self, text: str) -> Dict[str, Any]:
        """
        Extract demographic information from text
        
        Args:
            text: Text to extract demographics from
            
        Returns:
            Dictionary containing demographic information
        """
        demographics = {}
        
        # Extract gender
        if re.search(r'\b(male|men)\b', text, re.IGNORECASE):
            demographics['gender'] = 'male'
        elif re.search(r'\b(female|women)\b', text, re.IGNORECASE):
            demographics['gender'] = 'female'
            
        # Extract age range
        # Handle "aged 18-75 years" pattern
        age_range_match = re.search(r'(?:aged?)\s+(\d+)[-–](\d+)\s+years?', text, re.IGNORECASE)
        if age_range_match:
            demographics['age'] = {
                'min': int(age_range_match.group(1)),
                'max': int(age_range_match.group(2))
            }
        else:
            # Handle "between 18 and 65 years old" pattern
            between_match = re.search(r'(?:between|from)\s+(\d+)\s*(?:and|to)\s*(\d+)', text, re.IGNORECASE)
            if between_match:
                demographics['age'] = {
                    'min': int(between_match.group(1)),
                    'max': int(between_match.group(2))
                }
            else:
                # Handle "18 years of age or older" pattern
                older_match = re.search(r'(\d+)\s+years?\s+(?:of\s+)?age\s+or\s+(?:older|greater)', text, re.IGNORECASE)
                if older_match:
                    demographics['age'] = {
                        'min': int(older_match.group(1)),
                        'max': None
                    }
                else:
                    # Handle general age pattern
                    age_match = re.search(r'(?:age|aged?)\s*(?:of\s*)?(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?', text, re.IGNORECASE)
                    if age_match:
                        if age_match.group(2) in ['older', 'greater']:
                            demographics['age'] = {
                                'min': int(age_match.group(1)),
                                'max': None
                            }
                        elif age_match.group(2) in ['younger', 'less']:
                            demographics['age'] = {
                                'min': None,
                                'max': int(age_match.group(1))
                            }
                        else:
                            # Just a specific age
                            demographics['age'] = {
                                'min': int(age_match.group(1)),
                                'max': int(age_match.group(1))
                            }
            
        return demographics
        
    def _extract_performance_status(self, text: str) -> Dict[str, Any]:
        """
        Extract performance status information from text
        
        Args:
            text: Text to extract performance status from
            
        Returns:
            Dictionary containing performance status information
        """
        performance_status = {}
        
        # Extract ECOG score range (e.g., "ECOG performance status 0-1" or "performance status 0-2")
        ecog_range_match = re.search(r'(?:ECOG\s+)?performance\s+status\s+(\d)[-–](\d)', text, re.IGNORECASE)
        if ecog_range_match:
            performance_status['scale'] = 'ECOG'
            performance_status['values'] = list(range(int(ecog_range_match.group(1)), int(ecog_range_match.group(2)) + 1))
        else:
            # Extract single ECOG score
            ecog_match = re.search(r'ECOG\s*(?:performance\s+status\s+)?(\d)', text, re.IGNORECASE)
            if ecog_match:
                performance_status['scale'] = 'ECOG'
                performance_status['values'] = [int(ecog_match.group(1))]
            
        # Extract WHO score
        who_match = re.search(r'WHO\s*[-\s]?\s*(\d)', text, re.IGNORECASE)
        if who_match:
            performance_status['scale'] = 'WHO'
            performance_status['values'] = [int(who_match.group(1))]
            
        # Extract Karnofsky score (e.g., "Karnofsky performance status >70")
        karnofsky_match = re.search(r'Karnofsky\s*(?:performance\s+status\s+)?[>]?(\d{2})', text, re.IGNORECASE)
        if karnofsky_match:
            performance_status['scale'] = 'Karnofsky'
            # For >70, we'll just use 70 as the value
            performance_status['values'] = [int(karnofsky_match.group(1))]
            
        return performance_status
        
    def _extract_lab_values(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract lab value information from text
        
        Args:
            text: Text to extract lab values from
            
        Returns:
            List of extracted lab values
        """
        lab_values = []
        
        # This is a simplified implementation
        # In a real implementation, this would be more comprehensive
        lab_matches = re.findall(r'(WBC|Hemoglobin|Platelets|Creatinine|ALT|AST|Bilirubin|LDH|Albumin|INR|PTT|aPTT|CrCl|eGFR|TSH|T3|T4|PSA|CA-125|CEA|AFP|HCG|HbA1c|Glucose|Potassium|Sodium|Calcium|Magnesium|Phosphorus|BUN|pH|pCO2|pO2|HCO3|O2 saturation|O2 sat|SpO2)', text, re.IGNORECASE)
        
        for match in lab_matches:
            lab_values.append({
                'test': match,
                'confidence': 0.8
            })
            
        return lab_values
        
    def _extract_biomarkers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomarker information from text
        
        Args:
            text: Text to extract biomarkers from
            
        Returns:
            List of extracted biomarkers
        """
        biomarkers = []
        
        # Extract biomarker patterns
        biomarker_matches = re.findall(r'(ER|PR|HER2|HR)[\s\-\+]*(positive|negative|\+|\-)', text, re.IGNORECASE)
        for match in biomarker_matches:
            name = match[0].upper()
            status = match[1].lower()
            # Normalize status
            if status in ['+', 'positive']:
                status = 'positive'
            elif status in ['-', 'negative']:
                status = 'negative'
                
            biomarkers.append({
                'name': name,
                'status': status,
                'confidence': 0.9
            })
            
        # Extract full name biomarker patterns
        full_name_matches = re.findall(r'(estrogen receptor|progesterone receptor|human epidermal growth factor receptor 2|hormone receptor)[\s\-\+]*(positive|negative|\+|\-)', text, re.IGNORECASE)
        for match in full_name_matches:
            name_map = {
                'estrogen receptor': 'ER',
                'progesterone receptor': 'PR',
                'human epidermal growth factor receptor 2': 'HER2',
                'hormone receptor': 'HR'
            }
            name = name_map.get(match[0].lower(), match[0].upper())
            status = match[1].lower()
            # Normalize status
            if status in ['+', 'positive']:
                status = 'positive'
            elif status in ['-', 'negative']:
                status = 'negative'
                
            biomarkers.append({
                'name': name,
                'status': status,
                'confidence': 0.9
            })
            
        return biomarkers