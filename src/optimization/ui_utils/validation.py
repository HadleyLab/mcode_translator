"""
Input Validation Utilities for Modern Optimization UI
"""

from typing import Any, List, Dict
import re

class InputValidator:
    """Validate user inputs for the UI"""
    
    @staticmethod
    def validate_prompt_key(prompt_key: str) -> bool:
        """Validate prompt key format"""
        if not prompt_key or not isinstance(prompt_key, str):
            return False
        # Add specific validation rules as needed
        return len(prompt_key) > 0 and len(prompt_key) <= 100
    
    @staticmethod
    def validate_model_key(model_key: str) -> bool:
        """Validate model key format"""
        if not model_key or not isinstance(model_key, str):
            return False
        # Add specific validation rules as needed
        return len(model_key) > 0 and len(model_key) <= 100
    
    @staticmethod
    def validate_test_case_id(test_case_id: str) -> bool:
        """Validate test case ID format"""
        if not test_case_id or not isinstance(test_case_id, str):
            return False
        # Add specific validation rules as needed
        return len(test_case_id) > 0 and len(test_case_id) <= 100
    
    @staticmethod
    def validate_metric_name(metric_name: str) -> bool:
        """Validate metric name"""
        valid_metrics = ['f1_score', 'precision', 'recall', 'compliance_score']
        return metric_name in valid_metrics
    
    @staticmethod
    def validate_concurrency_level(concurrency: int) -> bool:
        """Validate concurrency level"""
        return isinstance(concurrency, int) and 1 <= concurrency <= 10
    
    @staticmethod
    def validate_timeout(timeout: int) -> bool:
        """Validate timeout value"""
        return isinstance(timeout, int) and 30 <= timeout <= 3600