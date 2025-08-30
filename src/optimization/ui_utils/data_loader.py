"""
Data Loading Utilities for Modern Optimization UI
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class DataLoader:
    """Load and process data for the UI"""
    
    @staticmethod
    def load_test_cases(file_path: str) -> Dict[str, Any]:
        """Load test cases from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_benchmark_results(results_dir: str) -> pd.DataFrame:
        """Load benchmark results from directory"""
        results_dir = Path(results_dir)
        all_results = []
        
        for result_file in results_dir.glob("benchmark_*.json"):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                all_results.append(result_data)
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    @staticmethod
    def load_prompt_library(config_path: str) -> Dict[str, Any]:
        """Load prompt library configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_model_library(config_path: str) -> Dict[str, Any]:
        """Load model library configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)