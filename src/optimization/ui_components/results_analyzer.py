"""
Results Analysis Components for Modern Optimization UI
"""

import pandas as pd
import json
from typing import Dict, Any, List
from datetime import datetime

class ResultsAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, framework):
        self.framework = framework
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for benchmark results"""
        df = self.framework.get_results_dataframe()
        
        if df.empty:
            return {}
        
        return {
            'total_runs': len(df),
            'success_rate': df['success'].mean(),
            'avg_duration_ms': df['duration_ms'].mean(),
            'avg_entities_extracted': df['entities_extracted'].mean(),
            'avg_compliance_score': df['compliance_score'].mean(),
            'avg_f1_score': df['f1_score'].mean(),
            'models_tested': df['model'].nunique(),
            'prompts_tested': df['prompt_name'].nunique()
        }
    
    def get_best_performers(self, metric: str = 'f1_score', top_n: int = 5) -> pd.DataFrame:
        """Get best performing prompt-model combinations"""
        return self.framework.get_best_combinations(metric, top_n)
    
    def generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualizations"""
        return self.framework.get_visualization_data()
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """Export results to CSV"""
        return self.framework.export_results_to_csv(filename)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        return self.framework.generate_performance_report()