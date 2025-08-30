from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from src.utils.logging_config import get_logger

@dataclass
class ProcessingResult:
    """Result from NLP processing containing extracted features and metadata"""
    features: Dict[str, Any]
    mcode_mappings: Dict[str, Any]
    metadata: Dict[str, Any]
    entities: List[Dict[str, Any]]
    error: Optional[str] = None

class NLPEngine(ABC):
    """Abstract base class for NLP engines"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @abstractmethod
    def process_text(self, text: str) -> ProcessingResult:
        """Process clinical text and return extracted features
        
        Args:
            text: Clinical text to process
            
        Returns:
            ProcessingResult containing extracted features
        """
        pass