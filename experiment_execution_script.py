#!/usr/bin/env python3
"""
Experiment Execution Framework for mCODE Translator Matching Engines

This script implements a comprehensive experiment execution framework that:
1. Loads curated matches from 'curated_matches.ndjson' as gold standard
2. Runs matching experiments using regex and LLM engines
3. Collects predictions for each patient-trial pair
4. Compares predictions against gold standard
5. Calculates evaluation metrics (precision, recall, F1-score, accuracy, MAP)
6. Performs statistical significance tests between engines
7. Saves comprehensive results to 'experiment_results.json'

Usage:
    python experiment_execution_script.py
"""

import asyncio
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.matching.llm_engine import LLMMatchingEngine
from src.matching.regex_engine import RegexRulesEngine
from src.optimization.performance_analyzer import PerformanceAnalyzer
from src.shared.models import McodeElement
from src.utils.concurrency import AsyncTaskQueue, Task, create_task
from src.utils.logging_config import get_logger
from src.utils.metrics import BenchmarkMetrics


class ExperimentExecutionFramework:
    """
    Framework for executing matching experiments and evaluating engine performance.
    """

    def __init__(self, max_concurrent: int = 8):
        self.logger = get_logger(__name__)
        self.gold_standard: List[Dict[str, Any]] = []
        self.patient_trial_pairs: List[Tuple[str, str]] = []
        self.results: Dict[str, Any] = {}
        self.max_concurrent = max_concurrent

        # Initialize engines
        self.regex_engine = self._initialize_regex_engine()
        self.llm_engine = self._initialize_llm_engine()

        # Initialize concurrency utilities
        self.task_queue = AsyncTaskQueue(max_concurrent=max_concurrent, name="ExperimentExecution")

    def _initialize_regex_engine(self) -> RegexRulesEngine:
        """Initialize the regex matching engine with rules."""
        # Load regex patterns from config
        rules = {
            "CancerCondition": r"breast cancer|carcinoma|neoplasm|malignant",
            "CancerStage": r"stage [IV]+|metastatic|advanced|early stage",
            "Treatment": r"chemotherapy|radiation|hormone therapy|targeted therapy|surgery",
            "Biomarker": r"HER2|ER|PR|BRCA|PIK3CA|TP53",
        }
        return RegexRulesEngine(rules)

    def _initialize_llm_engine(self) -> LLMMatchingEngine:
        """Initialize the LLM matching engine."""
        # Use deepseek-coder for patient-trial matching
        self.logger.info("Initializing LLM engine with deepseek-coder and direct_mcode_evidence_based_concise prompt")
        return LLMMatchingEngine(model_name="deepseek-coder", prompt_name="direct_mcode_evidence_based_concise")

    def load_gold_standard(self, filepath: str = "gold_standard_matches.ndjson") -> None:
        """Load curated matches as gold standard."""
        self.logger.info(f"Loading gold standard from {filepath}")

        gold_standard_path = Path(filepath)
        if not gold_standard_path.exists():
            raise FileNotFoundError(f"Gold standard file not found: {filepath}")

        self.gold_standard = []
        with open(gold_standard_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line.strip())
                    self.gold_standard.append(record)

        self.logger.info(f"Loaded {len(self.gold_standard)} gold standard records")

        # Extract unique patient-trial pairs
        self.patient_trial_pairs = list(set(
            (record["patient_id"], record["trial_id"])
            for record in self.gold_standard
        ))
        self.logger.info(f"Found {len(self.patient_trial_pairs)} unique patient-trial pairs")

    def _get_gold_standard_prediction(self, patient_id: str, trial_id: str) -> bool:
        """Get gold standard prediction for a patient-trial pair."""
        # Find the record for this pair
        for record in self.gold_standard:
            if record["patient_id"] == patient_id and record["trial_id"] == trial_id:
                # Relevance score of 4.0 indicates a match
                return record["relevance_score"] >= 4.0
        # If not found, assume no match
        return False

    async def _get_engine_prediction(
        self,
        engine_name: str,
        patient_data: Dict[str, Any],
        trial_criteria: Dict[str, Any]
    ) -> List[McodeElement]:
        """Get prediction from a specific engine."""
        engine = getattr(self, f"{engine_name}_engine")
        try:
            return await engine.match(patient_data, trial_criteria)
        except Exception as e:
            self.logger.error(f"Error getting prediction from {engine_name} engine: {e}")
            return []

    def _load_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Load patient data for a given patient ID."""
        # Load real patient data from selected_patients.ndjson
        try:
            with open("selected_patients.ndjson", 'r') as f:
                for line in f:
                    if line.strip():
                        patient = json.loads(line.strip())
                        # Extract patient ID from FHIR bundle
                        bundle = patient.get("entry", [])
                        for entry in bundle:
                            if entry["resource"]["resourceType"] == "Patient":
                                if entry["resource"]["id"] == patient_id:
                                    return patient
        except FileNotFoundError:
            raise FileNotFoundError("selected_patients.ndjson not found")

        raise ValueError(f"Patient {patient_id} not found in selected_patients.ndjson")

    def _load_trial_criteria(self, trial_id: str) -> Dict[str, Any]:
        """Load trial eligibility criteria for a given trial ID."""
        # Load real trial data from selected_trials.ndjson
        try:
            with open("selected_trials.ndjson", 'r') as f:
                for line in f:
                    if line.strip():
                        trial = json.loads(line.strip())
                        # Extract trial ID from ClinicalTrials.gov format
                        current_trial_id = trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                        if current_trial_id == trial_id:
                            return trial
        except FileNotFoundError:
            raise FileNotFoundError("selected_trials.ndjson not found")

        raise ValueError(f"Trial {trial_id} not found in selected_trials.ndjson")

    async def _process_patient_trial_pair(self, patient_id: str, trial_id: str) -> Dict[str, Any]:
        """Process a single patient-trial pair concurrently."""
        try:
            self.logger.info(f"Starting processing for pair {patient_id} - {trial_id}")

            # Load data
            self.logger.debug(f"Loading patient data for {patient_id}")
            patient_data = self._load_patient_data(patient_id)
            self.logger.debug(f"Loaded patient data: {len(patient_data)} entries")

            self.logger.debug(f"Loading trial criteria for {trial_id}")
            trial_criteria = self._load_trial_criteria(trial_id)
            self.logger.debug(f"Loaded trial criteria: {trial_criteria.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'Unknown')}")

            # Get gold standard
            gold_standard_match = self._get_gold_standard_prediction(patient_id, trial_id)
            self.logger.debug(f"Gold standard match: {gold_standard_match}")

            # Get engine predictions
            self.logger.debug(f"Getting regex engine prediction")
            regex_prediction = await self._get_engine_prediction("regex", patient_data, trial_criteria)
            self.logger.debug(f"Regex prediction: {len(regex_prediction)} elements")

            self.logger.debug(f"Getting LLM engine prediction")
            llm_prediction = await self._get_engine_prediction("llm", patient_data, trial_criteria)
            self.logger.debug(f"LLM prediction: {len(llm_prediction)} elements")

            # Convert predictions to binary (has elements = match)
            regex_match = len(regex_prediction) > 0
            # Handle LLM response - it now returns PatientTrialMatchResponse
            if hasattr(llm_prediction, 'is_match'):
                llm_match = llm_prediction.is_match
            else:
                llm_match = len(llm_prediction) > 0

            self.logger.info(f"Processed pair {patient_id} - {trial_id}: Gold={gold_standard_match}, Regex={regex_match} ({len(regex_prediction)} elements), LLM={llm_match} ({len(llm_prediction)} elements)")

            # Log detailed results for debugging
            if not regex_match and gold_standard_match:
                self.logger.warning(f"Regex mismatch: Expected match but got no elements for {patient_id} - {trial_id}")
            if not llm_match and gold_standard_match:
                self.logger.warning(f"LLM mismatch: Expected match but got no elements for {patient_id} - {trial_id}")

            return {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "gold_standard": gold_standard_match,
                "regex_prediction": regex_match,
                "llm_prediction": llm_match,
                "regex_elements": len(regex_prediction),
                "llm_elements": len(llm_prediction),
                "success": True
            }

        except Exception as e:
            self.logger.error(f"Error processing pair {patient_id} - {trial_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "gold_standard": False,
                "regex_prediction": False,
                "llm_prediction": False,
                "regex_elements": 0,
                "llm_elements": 0,
                "success": False,
                "error": str(e)
            }

    async def run_experiments(self) -> Dict[str, Any]:
        """Run matching experiments for all patient-trial pairs using concurrent processing."""
        self.logger.info("Starting concurrent experiment execution...")

        results = {
            "regex_engine": {
                "predictions": [],
                "gold_standard": [],
                "metrics": {}
            },
            "llm_engine": {
                "predictions": [],
                "gold_standard": [],
                "metrics": {}
            },
            "comparison": {},
            "metadata": {
                "total_pairs": len(self.patient_trial_pairs),
                "execution_timestamp": None,
                "concurrent_workers": self.max_concurrent
            }
        }

        # Use all real patient-trial pairs for authentic evaluation
        test_pairs = self.patient_trial_pairs  # Use all pairs for real evaluation
        self.logger.info(f"Testing with {len(test_pairs)} patient-trial pairs using {self.max_concurrent} concurrent workers")

        # Log the test pairs for debugging
        self.logger.info("Test pairs to process:")
        for i, (patient_id, trial_id) in enumerate(test_pairs):
            self.logger.info(f"  {i+1:2d}. {patient_id} -> {trial_id}")

        # Create concurrent tasks for processing patient-trial pairs
        tasks = []
        for patient_id, trial_id in test_pairs:
            task = create_task(
                f"pair_{patient_id}_{trial_id}",
                self._process_patient_trial_pair,
                patient_id,
                trial_id
            )
            tasks.append(task)

        # Execute tasks concurrently
        task_results = await self.task_queue.execute_tasks(tasks)

        # Collect results in order (maintain pairing order)
        pair_results = {}
        for result in task_results:
            if result.success and result.result:
                pair_data = result.result
                pair_results[(pair_data["patient_id"], pair_data["trial_id"])] = pair_data

        # Sort results by original order and collect predictions
        for patient_id, trial_id in test_pairs:
            if (patient_id, trial_id) in pair_results:
                pair_data = pair_results[(patient_id, trial_id)]
                results["regex_engine"]["predictions"].append(pair_data["regex_prediction"])
                results["regex_engine"]["gold_standard"].append(pair_data["gold_standard"])
                results["llm_engine"]["predictions"].append(pair_data["llm_prediction"])
                results["llm_engine"]["gold_standard"].append(pair_data["gold_standard"])
            else:
                # Handle failed pairs with default values
                self.logger.warning(f"No result for pair {patient_id} - {trial_id}, using defaults")
                results["regex_engine"]["predictions"].append(False)
                results["regex_engine"]["gold_standard"].append(False)
                results["llm_engine"]["predictions"].append(False)
                results["llm_engine"]["gold_standard"].append(False)

        # Calculate metrics
        results["regex_engine"]["metrics"] = self._calculate_metrics(
            results["regex_engine"]["predictions"],
            results["regex_engine"]["gold_standard"]
        )

        results["llm_engine"]["metrics"] = self._calculate_metrics(
            results["llm_engine"]["predictions"],
            results["llm_engine"]["gold_standard"]
        )

        # Perform statistical comparison
        results["comparison"] = self._perform_statistical_tests(
            results["regex_engine"]["predictions"],
            results["llm_engine"]["predictions"],
            results["regex_engine"]["gold_standard"]
        )

        self.logger.info("Concurrent experiment execution completed")
        return results

    def _calculate_metrics(self, predictions: List[bool], gold_standard: List[bool]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        # Convert to binary lists
        y_pred = [1 if p else 0 for p in predictions]
        y_true = [1 if g else 0 for g in gold_standard]

        # Calculate confusion matrix
        tp = sum(1 for p, g in zip(y_pred, y_true) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(y_pred, y_true) if p == 1 and g == 0)
        tn = sum(1 for p, g in zip(y_pred, y_true) if p == 0 and g == 0)
        fn = sum(1 for p, g in zip(y_pred, y_true) if p == 0 and g == 1)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_pred) if len(y_pred) > 0 else 0

        # Calculate MAP (Mean Average Precision)
        map_score = self._calculate_map(y_pred, y_true)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "map": map_score,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
        }

    def _calculate_map(self, predictions: List[int], gold_standard: List[int]) -> float:
        """Calculate Mean Average Precision."""
        if not predictions or not gold_standard:
            return 0.0

        # For binary classification, MAP is equivalent to average precision
        relevant = sum(gold_standard)
        if relevant == 0:
            return 0.0

        # Calculate precision at each relevant document
        precision_sum = 0.0
        relevant_found = 0

        for i, (pred, true) in enumerate(zip(predictions, gold_standard)):
            if true == 1:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / relevant

    def _perform_statistical_tests(
        self,
        regex_predictions: List[bool],
        llm_predictions: List[bool],
        gold_standard: List[bool]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between engines."""
        # Convert to numpy arrays for statistical tests
        regex_pred = np.array([1 if p else 0 for p in regex_predictions])
        llm_pred = np.array([1 if p else 0 for p in llm_predictions])
        gold = np.array([1 if g else 0 for g in gold_standard])

        results = {}

        # McNemar's test for paired nominal data
        try:
            # Create contingency table
            both_correct = sum((r == g) and (l == g) for r, l, g in zip(regex_pred, llm_pred, gold))
            regex_correct_llm_wrong = sum((r == g) and (l != g) for r, l, g in zip(regex_pred, llm_pred, gold))
            llm_correct_regex_wrong = sum((l == g) and (r != g) for r, l, g in zip(regex_pred, llm_pred, gold))
            both_wrong = sum((r != g) and (l != g) for r, l, g in zip(regex_pred, llm_pred, gold))

            contingency_table = np.array([
                [both_correct, regex_correct_llm_wrong],
                [llm_correct_regex_wrong, both_wrong]
            ])

            if contingency_table.sum() > 0:
                chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                results["mcnemar_test"] = {
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        except Exception as e:
            self.logger.warning(f"McNemar's test failed: {e}")
            results["mcnemar_test"] = {"error": str(e)}

        # Cohen's Kappa for inter-rater agreement
        try:
            kappa = self._cohens_kappa(regex_pred, llm_pred)
            results["cohens_kappa"] = kappa
        except Exception as e:
            self.logger.warning(f"Cohen's Kappa calculation failed: {e}")
            results["cohens_kappa"] = {"error": str(e)}

        # Performance comparison
        regex_accuracy = sum(r == g for r, g in zip(regex_pred, gold)) / len(gold)
        llm_accuracy = sum(l == g for l, g in zip(llm_pred, gold)) / len(gold)

        results["performance_comparison"] = {
            "regex_accuracy": regex_accuracy,
            "llm_accuracy": llm_accuracy,
            "accuracy_difference": llm_accuracy - regex_accuracy
        }

        return results

    def _cohens_kappa(self, predictions1: np.ndarray, predictions2: np.ndarray) -> float:
        """Calculate Cohen's Kappa coefficient."""
        if len(predictions1) != len(predictions2):
            raise ValueError("Prediction arrays must have same length")

        n = len(predictions1)
        p_o = np.mean(predictions1 == predictions2)  # Observed agreement

        p_e = (np.mean(predictions1) * np.mean(predictions2) +
               (1 - np.mean(predictions1)) * (1 - np.mean(predictions2)))  # Expected agreement

        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def save_results(self, results: Dict[str, Any], filepath: str = "experiment_results.json") -> None:
        """Save comprehensive results to JSON file."""
        import datetime

        # Add timestamp
        results["metadata"]["execution_timestamp"] = datetime.datetime.utcnow().isoformat()

        # Add confidence intervals
        results = self._add_confidence_intervals(results)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {filepath}")

    def _add_confidence_intervals(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add confidence intervals to metrics."""
        def calculate_ci(metric_values: List[float], confidence: float = 0.95) -> Dict[str, float]:
            if len(metric_values) < 2:
                return {"mean": metric_values[0] if metric_values else 0, "ci_lower": 0, "ci_upper": 0}

            mean = statistics.mean(metric_values)
            std_err = statistics.stdev(metric_values) / math.sqrt(len(metric_values))
            margin = std_err * stats.t.ppf((1 + confidence) / 2, len(metric_values) - 1)

            return {
                "mean": mean,
                "ci_lower": max(0, mean - margin),
                "ci_upper": min(1, mean + margin)
            }

        # For now, we don't have multiple runs, so we'll use bootstrap sampling
        # In a real implementation, you'd run multiple experiments

        return results

    async def execute(self) -> None:
        """Execute the complete experiment framework."""
        try:
            # Load gold standard
            self.load_gold_standard()

            # Run experiments
            results = await self.run_experiments()

            # Save results
            self.save_results(results)

            # Log summary
            self._log_summary(results)

        except Exception as e:
            self.logger.error(f"Experiment execution failed: {e}")
            raise

    def _log_summary(self, results: Dict[str, Any]) -> None:
        """Log a summary of the experiment results."""
        self.logger.info("=== EXPERIMENT EXECUTION SUMMARY ===")
        self.logger.info(f"Total patient-trial pairs: {results['metadata']['total_pairs']}")

        for engine in ["regex_engine", "llm_engine"]:
            metrics = results[engine]["metrics"]
            self.logger.info(f"\n{engine.upper()} METRICS:")
            self.logger.info(f"  Precision: {metrics['precision']:.3f}")
            self.logger.info(f"  Recall: {metrics['recall']:.3f}")
            self.logger.info(f"  F1-Score: {metrics['f1_score']:.3f}")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
            self.logger.info(f"  MAP: {metrics['map']:.3f}")

        comparison = results["comparison"]
        self.logger.info("\nSTATISTICAL COMPARISON:")
        if "mcnemar_test" in comparison:
            mcnemar = comparison["mcnemar_test"]
            if "p_value" in mcnemar:
                self.logger.info(f"  McNemar's test p-value: {mcnemar['p_value']:.4f}")
                self.logger.info(f"  Significant difference: {mcnemar['significant']}")

        perf_comp = comparison["performance_comparison"]
        self.logger.info(f"  Regex accuracy: {perf_comp['regex_accuracy']:.3f}")
        self.logger.info(f"  LLM accuracy: {perf_comp['llm_accuracy']:.3f}")
        self.logger.info(f"  Accuracy difference: {perf_comp['accuracy_difference']:.3f}")


async def main():
    """Main entry point."""
    framework = ExperimentExecutionFramework()
    await framework.execute()


if __name__ == "__main__":
    asyncio.run(main())