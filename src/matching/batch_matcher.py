"""
Batch matching functionality for patient-trial matching engines.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.matching.base import MatchingEngineBase, MatchingResult
from src.matching.data_loader import load_patients, load_trials, create_patient_trial_pairs
from src.utils.logging_config import get_logger


class BatchMatcher:
    """
    Handles batch processing of patient-trial matching using any matching engine.
    """

    def __init__(self, engine: MatchingEngineBase, max_concurrent: int = 10):
        """
        Initialize batch matcher with a specific engine.

        Args:
            engine: The matching engine to use
            max_concurrent: Maximum number of concurrent matching operations
        """
        self.engine = engine
        self.max_concurrent = max_concurrent
        self.logger = get_logger(__name__)

    async def match_batch(self, pairs: List[Dict[str, Any]]) -> List[MatchingResult]:
        """
        Match a batch of patient-trial pairs.

        Args:
            pairs: List of dictionaries with 'patient' and 'trial' keys

        Returns:
            List of MatchingResult objects
        """
        self.logger.info(f"Starting batch matching of {len(pairs)} pairs with {self.max_concurrent} concurrent operations")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def match_with_semaphore(pair: Dict[str, Any]) -> MatchingResult:
            async with semaphore:
                return await self.engine.match_with_recovery(pair['patient'], pair['trial'])

        # Process all pairs concurrently with semaphore
        tasks = [match_with_semaphore(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, MatchingResult):
                valid_results.append(result)
            else:
                self.logger.error(f"Matching failed for pair {i}: {result}")

        self.logger.info(f"Batch matching completed: {len(valid_results)} successful, {len(results) - len(valid_results)} failed")
        return valid_results

    async def match_files(self, patients_file: str, trials_file: str) -> List[MatchingResult]:
        """
        Match all patients against all trials from files.

        Args:
            patients_file: Path to patients NDJSON file
            trials_file: Path to trials NDJSON file

        Returns:
            List of MatchingResult objects for all patient-trial pairs
        """
        self.logger.info(f"Loading data from {patients_file} and {trials_file}")

        patients = load_patients(patients_file)
        trials = load_trials(trials_file)

        self.logger.info(f"Loaded {len(patients)} patients and {len(trials)} trials")

        pairs = create_patient_trial_pairs(patients, trials)
        self.logger.info(f"Created {len(pairs)} patient-trial pairs for matching")

        return await self.match_batch(pairs)

    def calculate_matching_scores(self, results: List[MatchingResult]) -> Dict[str, Any]:
        """
        Calculate aggregate matching scores and statistics.

        Args:
            results: List of MatchingResult objects

        Returns:
            Dictionary with matching statistics
        """
        total_matches = len(results)
        successful_matches = sum(1 for r in results if not r.error)
        error_matches = total_matches - successful_matches

        # Calculate average confidence scores for successful matches
        confidence_scores = []
        total_elements = 0

        for result in results:
            if not result.error and result.elements:
                total_elements += len(result.elements)
                for element in result.elements:
                    confidence_scores.append(element.confidence_score)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_elements_per_match = total_elements / successful_matches if successful_matches > 0 else 0

        return {
            'total_pairs': total_matches,
            'successful_matches': successful_matches,
            'error_matches': error_matches,
            'success_rate': successful_matches / total_matches if total_matches > 0 else 0,
            'average_confidence_score': avg_confidence,
            'average_elements_per_match': avg_elements_per_match,
            'total_matched_elements': total_elements
        }

    def save_results(self, results: List[MatchingResult], output_file: str) -> None:
        """
        Save matching results to NDJSON file.

        Args:
            output_file: Path to output file
        """
        self.logger.info(f"Saving {len(results)} results to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json_line = json.dumps(result.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

        self.logger.info(f"Results saved successfully")

    async def run_complete_matching(
        self,
        patients_file: str,
        trials_file: str,
        output_file: str,
        save_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete matching pipeline: load data, match, save results, return stats.

        Args:
            patients_file: Path to patients NDJSON file
            trials_file: Path to trials NDJSON file
            output_file: Path to save results
            save_stats: Whether to save statistics alongside results

        Returns:
            Dictionary with matching statistics
        """
        self.logger.info("Starting complete matching pipeline")

        # Run matching
        results = await self.match_files(patients_file, trials_file)

        # Calculate statistics
        stats = self.calculate_matching_scores(results)

        # Save results
        self.save_results(results, output_file)

        # Save statistics if requested
        if save_stats:
            stats_file = output_file.replace('.ndjson', '_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Statistics saved to {stats_file}")

        self.logger.info("Complete matching pipeline finished")
        return stats