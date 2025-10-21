"""
Evaluation and scoring utilities for matching engines.
"""

from collections import defaultdict
from typing import Any, Dict, List

from src.matching.base import MatchingResult
from src.utils.logging_config import get_logger


class MatchingEvaluator:
    """
    Evaluates and scores matching results from different engines.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_match_score(self, result: MatchingResult) -> float:
        """
        Calculate an overall match score for a single result.

        Args:
            result: MatchingResult to score

        Returns:
            Float score between 0 and 1
        """
        if result.error or not result.elements:
            return 0.0

        # Weight different element types
        weights = {
            'AgeMatch': 0.2,
            'StageMatch': 0.3,
            'CancerTypeMatch': 0.3,
            'BiomarkerMatch': 0.2,
            'PrimaryCancerCondition': 0.25,
            'CancerStage': 0.25,
            'TumorMarker': 0.15,
            'ECOGPerformanceStatus': 0.15,
            'PatientTrialRelationship': 0.2
        }

        total_weight = 0
        weighted_score = 0

        for element in result.elements:
            weight = weights.get(element.element_type, 0.1)
            weighted_score += element.confidence_score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def compare_engines(
        self,
        results_a: List[MatchingResult],
        results_b: List[MatchingResult],
        engine_a_name: str = "Engine A",
        engine_b_name: str = "Engine B"
    ) -> Dict[str, Any]:
        """
        Compare results from two different engines.

        Args:
            results_a: Results from first engine
            results_b: Results from second engine
            engine_a_name: Name of first engine
            engine_b_name: Name of second engine

        Returns:
            Dictionary with comparison statistics
        """
        if len(results_a) != len(results_b):
            raise ValueError("Result lists must have the same length for comparison")

        scores_a = [self.calculate_match_score(r) for r in results_a]
        scores_b = [self.calculate_match_score(r) for r in results_b]

        # Calculate basic statistics
        avg_score_a = sum(scores_a) / len(scores_a) if scores_a else 0
        avg_score_b = sum(scores_b) / len(scores_b) if scores_b else 0

        # Count matches above threshold
        threshold = 0.5
        matches_a = sum(1 for s in scores_a if s >= threshold)
        matches_b = sum(1 for s in scores_b if s >= threshold)

        # Calculate agreement
        agreements = 0
        disagreements = 0
        for sa, sb in zip(scores_a, scores_b):
            match_a = sa >= threshold
            match_b = sb >= threshold
            if match_a == match_b:
                agreements += 1
            else:
                disagreements += 1

        agreement_rate = agreements / len(scores_a) if scores_a else 0

        return {
            f'{engine_a_name}_avg_score': avg_score_a,
            f'{engine_b_name}_avg_score': avg_score_b,
            f'{engine_a_name}_matches': matches_a,
            f'{engine_b_name}_matches': matches_b,
            'total_pairs': len(scores_a),
            'agreement_rate': agreement_rate,
            'disagreement_rate': disagreements / len(scores_a) if scores_a else 0
        }

    def analyze_element_types(self, results: List[MatchingResult]) -> Dict[str, Any]:
        """
        Analyze which element types are most commonly matched.

        Args:
            results: List of MatchingResult objects

        Returns:
            Dictionary with element type statistics
        """
        element_counts = defaultdict(int)
        element_scores = defaultdict(list)

        for result in results:
            if not result.error:
                for element in result.elements:
                    element_counts[element.element_type] += 1
                    element_scores[element.element_type].append(element.confidence_score)

        # Calculate averages
        element_stats = {}
        for elem_type, scores in element_scores.items():
            element_stats[elem_type] = {
                'count': element_counts[elem_type],
                'avg_confidence': sum(scores) / len(scores) if scores else 0,
                'max_confidence': max(scores) if scores else 0,
                'min_confidence': min(scores) if scores else 0
            }

        return dict(sorted(element_stats.items(), key=lambda x: x[1]['count'], reverse=True))

    def generate_report(
        self,
        results: List[MatchingResult],
        engine_name: str,
        comparison_results: Dict[str, Any] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            results: Matching results to analyze
            engine_name: Name of the engine
            comparison_results: Optional comparison with another engine

        Returns:
            Formatted report string
        """
        scores = [self.calculate_match_score(r) for r in results]
        element_stats = self.analyze_element_types(results)

        report = f"""
# Matching Engine Evaluation Report: {engine_name}

## Summary Statistics
- Total pairs processed: {len(results)}
- Successful matches: {sum(1 for r in results if not r.error)}
- Failed matches: {sum(1 for r in results if r.error)}
- Average match score: {sum(scores) / len(scores) if scores else 0:.3f}
- High-confidence matches (≥0.7): {sum(1 for s in scores if s >= 0.7)}

## Element Type Analysis
"""

        for elem_type, stats in element_stats.items():
            report += f"- {elem_type}: {stats['count']} matches, avg confidence {stats['avg_confidence']:.3f}\n"

        if comparison_results:
            report += """
## Engine Comparison
"""
            for key, value in comparison_results.items():
                if isinstance(value, float):
                    report += f"- {key}: {value:.3f}\n"
                else:
                    report += f"- {key}: {value}\n"

        return report
