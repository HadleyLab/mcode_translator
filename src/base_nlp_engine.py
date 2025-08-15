from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseNLPEngine(ABC):
    """
    Abstract base class for NLP engines defining the common interface
    """
    
    @abstractmethod
    def extract_features(self, criteria_text: str) -> Dict[str, Any]:
        """
        Extract structured features from clinical trial eligibility criteria
        Args:
            criteria_text: Input text containing eligibility criteria
        Returns:
            Dictionary of extracted features
        """
        pass

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text
        Args:
            text: Input text to clean
        Returns:
            Cleaned text
        """
        pass

    @abstractmethod
    def identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify inclusion/exclusion sections in criteria text
        Args:
            text: Input criteria text
        Returns:
            Dictionary with section names and content
        """
        pass