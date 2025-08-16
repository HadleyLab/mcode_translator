from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
from utils.feature_utils import (
    standardize_features,
    standardize_biomarkers,
    standardize_variants
)

@dataclass
class ProcessingResult:
    """Standardized result container for NLP processing.
    
    Attributes:
        features (Dict[str, Any]): Extracted mCODE features in standardized format
        mcode_mappings (Dict[str, Any]): FHIR mappings for extracted features
        metadata (Dict[str, Any]): Processing metadata and statistics
        entities (List[Dict[str, Any]]): Raw extracted entities
        error (Optional[str]): Error message if processing failed, None if successful
    """
    features: Dict[str, Any]
    mcode_mappings: Dict[str, Any]
    metadata: Dict[str, Any]
    entities: List[Dict[str, Any]]
    error: Optional[str] = None

class NLPEngine(ABC):
    """Abstract base class for all NLP engine implementations.
    
    Provides common functionality including:
    - Standardized logging configuration
    - Error handling patterns
    - Result formatting to ProcessingResult
    - Type hints for all public methods
    - Common utility methods for feature standardization
    
    Subclasses must implement:
    - process_text(): Core text processing method
    
    Attributes:
        logger (logging.Logger): Configured logger instance
    """
    
    def __init__(self):
        """Initialize the NLP engine with standard configuration.
        
        Sets up:
        - Logging configuration
        - Standard error handling
        - Common utilities
        """
        self._init_logging()
        
    def _init_logging(self):
        """Configure standardized logging for all NLP engines.
        
        Creates logger with:
        - Class name as logger name
        - INFO level by default
        - Console handler with standard format
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
    @abstractmethod
    def process_text(self, text: str) -> ProcessingResult:
        """Process clinical text and extract mCODE features.
        
        Args:
            text: Input clinical text to process. Can be a single string or list of strings.
            
        Returns:
            ProcessingResult containing:
            - features: Extracted mCODE features (Dict)
            - mcode_mappings: FHIR mappings (Dict)
            - metadata: Processing metadata (Dict)
            - entities: Raw extracted entities (List[Dict])
            - error: Optional error message (str)
            
        Raises:
            ValueError: If input text is empty or invalid
        """
        pass
        
    def _create_error_result(self, error_msg: str) -> ProcessingResult:
        """Create standardized error result when processing fails.
        
        Args:
            error_msg (str): Description of the error that occurred
            
        Returns:
            ProcessingResult: Configured with empty features and the error message
            
        Example:
            >>> engine._create_error_result("Invalid input")
            ProcessingResult(features={}, mcode_mappings={},
                           metadata={'error': 'Invalid input'},
                           entities=[], error='Invalid input')
        """
        return ProcessingResult(
            features={},
            mcode_mappings={},
            metadata={'error': error_msg},
            entities=[],
            error=error_msg
        )
        
    def process_criteria(self, criteria: Union[str, List[str]]) -> ProcessingResult:
        """Process eligibility criteria text with standardized output.
        
        Args:
            criteria: Clinical trial eligibility criteria text
            
        Returns:
            Standardized output dictionary with structure:
            {
                'features': {
                    'demographics': Dict,
                    'cancer_characteristics': Dict,
                    'biomarkers': List[Dict],
                    'genomic_variants': List[Dict],
                    'treatment_history': Dict,
                    'performance_status': Dict
                },
                'mcode_mappings': Dict,
                'metadata': {
                    'processing_time': float,
                    'biomarkers_count': int,
                    'genomic_variants_count': int
                },
                'entities': List[Dict]
            }
        """
        try:
            result = self.process_text(criteria)
            
            # Standardize output format
            # Convert dict result to ProcessingResult if needed
            if isinstance(result, dict):
                result = ProcessingResult(
                    features=self._standardize_features(result.get('features', {})),
                    mcode_mappings=result.get('mcode_mappings', {}),
                    metadata={
                        'processing_time': result.get('metadata', {}).get('processing_time', 0),
                        'biomarkers_count': len(result.get('features', {}).get('biomarkers', [])),
                        'genomic_variants_count': len(result.get('features', {}).get('genomic_variants', []))
                    },
                    entities=result.get('entities', [])
                )
            
            return result
        except ValueError as e:
            self.logger.error(f"Validation error: {str(e)}")
            return self._create_error_result(f"Validation error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error processing criteria: {str(e)}", exc_info=True)
            return self._create_error_result(f"Unexpected error: {str(e)}")
    
    def _standardize_features(self, features: Dict) -> Dict:
        """Ensure features have consistent structure across engines.
        
        Parameters
        ----------
        features : Dict
            Raw extracted features from NLP processing
            
        Returns
        -------
        Dict
            Standardized features using shared utility functions
        """
        return standardize_features(features)