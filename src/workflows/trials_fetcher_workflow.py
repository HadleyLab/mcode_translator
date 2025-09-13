"""
Trials Fetcher Workflow - Fetch clinical trials from external APIs.

This workflow handles fetching raw clinical trial data from ClinicalTrials.gov
without any processing or core memory storage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.fetcher import (ClinicalTrialsAPIError, get_full_study,
                                  search_trials)
from src.utils.logging_config import get_logger

from .base_workflow import FetcherWorkflow, WorkflowResult


class TrialsFetcherWorkflow(FetcherWorkflow):
    """
    Workflow for fetching clinical trials from ClinicalTrials.gov.

    Fetches raw trial data without processing or storage to core memory.
    """

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the trials fetching workflow.

        Args:
            **kwargs: Workflow parameters including:
                - condition: Medical condition to search for
                - nct_id: Specific NCT ID to fetch
                - nct_ids: List of NCT IDs to fetch
                - limit: Maximum number of results
                - output_path: Where to save results

        Returns:
            WorkflowResult: Fetch results
        """
        try:
            # Extract parameters
            condition = kwargs.get("condition")
            nct_id = kwargs.get("nct_id")
            nct_ids = kwargs.get("nct_ids")
            limit = kwargs.get("limit", 10)
            output_path = kwargs.get("output_path")

            # Validate inputs
            if not self._validate_fetch_params(condition, nct_id, nct_ids):
                return self._create_result(
                    success=False,
                    error_message="Invalid fetch parameters. Must provide condition, nct_id, or nct_ids.",
                )

            # Execute fetch
            if condition:
                results = self._fetch_by_condition(condition, limit)
            elif nct_id:
                results = self._fetch_single_trial(nct_id)
            elif nct_ids:
                results = self._fetch_multiple_trials(nct_ids)
            else:
                return self._create_result(
                    success=False, error_message="No valid fetch parameters provided."
                )

            # Save results if output path provided
            if output_path and results["success"]:
                self._save_results(results["data"], output_path)

            return self._create_result(
                success=results["success"],
                data=results["data"],
                error_message=results.get("error"),
                metadata={
                    "fetch_type": results.get("type", "unknown"),
                    "total_fetched": len(results.get("data", [])),
                    "output_path": str(output_path) if output_path else None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials fetching")

    def _validate_fetch_params(
        self,
        condition: Optional[str],
        nct_id: Optional[str],
        nct_ids: Optional[List[str]],
    ) -> bool:
        """Validate fetch parameters."""
        params = [condition, nct_id, nct_ids]
        provided = [p for p in params if p is not None]
        return len(provided) == 1  # Exactly one parameter should be provided

    def _fetch_by_condition(self, condition: str, limit: int) -> Dict[str, Any]:
        """Fetch trials by medical condition."""
        try:
            self.logger.info(f"🔍 Searching for trials: '{condition}' (limit: {limit})")

            search_result = search_trials(condition, max_results=limit)
            trials = search_result.get("studies", [])

            if not trials:
                return {
                    "success": False,
                    "error": f"No trials found for condition: {condition}",
                    "data": [],
                }

            self.logger.info(f"📋 Found {len(trials)} trials")

            return {
                "success": True,
                "type": "condition_search",
                "data": trials,
                "metadata": {
                    "condition": condition,
                    "total_found": len(trials),
                    "limit": limit,
                },
            }

        except ClinicalTrialsAPIError as e:
            self.logger.error(f"API Error: {e}")
            return {
                "success": False,
                "error": f"ClinicalTrials.gov API error: {e}",
                "data": [],
            }
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return {"success": False, "error": f"Unexpected error: {e}", "data": []}

    def _fetch_single_trial(self, nct_id: str) -> Dict[str, Any]:
        """Fetch a single trial by NCT ID."""
        try:
            self.logger.info(f"📥 Fetching trial: {nct_id}")

            trial = get_full_study(nct_id)

            if not trial:
                return {
                    "success": False,
                    "error": f"Trial not found: {nct_id}",
                    "data": [],
                }

            self.logger.info(f"✅ Successfully fetched trial: {nct_id}")

            return {
                "success": True,
                "type": "single_trial",
                "data": [trial],
                "metadata": {"nct_id": nct_id},
            }

        except ClinicalTrialsAPIError as e:
            self.logger.error(f"API Error fetching {nct_id}: {e}")
            return {
                "success": False,
                "error": f"ClinicalTrials.gov API error: {e}",
                "data": [],
            }
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {nct_id}: {e}")
            return {"success": False, "error": f"Unexpected error: {e}", "data": []}

    def _fetch_multiple_trials(self, nct_ids: List[str]) -> Dict[str, Any]:
        """Fetch multiple trials by NCT IDs."""
        try:
            self.logger.info(f"📥 Fetching {len(nct_ids)} trials")

            successful_trials = []
            failed_trials = []

            for nct_id in nct_ids:
                try:
                    trial = get_full_study(nct_id.strip())
                    if trial:
                        successful_trials.append(trial)
                        self.logger.info(f"✅ Fetched: {nct_id}")
                    else:
                        failed_trials.append(nct_id)
                        self.logger.warning(f"❌ Not found: {nct_id}")
                except Exception as e:
                    failed_trials.append(nct_id)
                    self.logger.error(f"❌ Failed to fetch {nct_id}: {e}")

            success_rate = len(successful_trials) / len(nct_ids) if nct_ids else 0

            self.logger.info(
                f"📊 Fetch complete: {len(successful_trials)}/{len(nct_ids)} successful"
            )

            return {
                "success": len(successful_trials) > 0,
                "type": "multiple_trials",
                "data": successful_trials,
                "metadata": {
                    "total_requested": len(nct_ids),
                    "successful": len(successful_trials),
                    "failed": len(failed_trials),
                    "success_rate": success_rate,
                    "failed_trials": failed_trials,
                },
            }

        except Exception as e:
            self.logger.error(f"Unexpected error in batch fetch: {e}")
            return {"success": False, "error": f"Batch fetch error: {e}", "data": []}

    def _save_results(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save fetch results to file."""
        try:
            output_file = Path(output_path)

            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Results saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results to {output_path}: {e}")
            raise
