"""
Trials Fetcher Workflow - Fetch clinical trials from external APIs.

This workflow handles fetching raw clinical trial data from ClinicalTrials.gov
without any processing or core memory storage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.fetcher import (ClinicalTrialsAPIError, get_full_study,
                               search_trials, get_full_studies_batch,
                               search_trials_parallel)
from typing import Tuple
from src.utils.logging_config import get_logger
from src.utils.concurrency import TaskQueue, create_task, get_fetcher_pool

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
                - cli_args: CLI arguments for concurrency configuration

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
            cli_args = kwargs.get("cli_args")  # CLI arguments for concurrency

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
        """Fetch trials by medical condition with full study data."""
        try:
            self.logger.info(f"🔍 Searching for trials: '{condition}' (limit: {limit})")

            # Step 1: Search for trials (basic info) with parallel pagination if needed
            if limit > 100:
                # Use parallel search for large result sets
                self.logger.info(f"🔄 Using parallel search for {limit} results")
                search_result = search_trials_parallel(
                    condition,
                    max_results=limit,
                    page_size=100,  # API limit per page
                    max_workers=4
                )
                basic_trials = search_result.get("studies", [])
            else:
                # Use regular search for smaller result sets
                search_result = search_trials(condition, max_results=limit)
                basic_trials = search_result.get("studies", [])

            if not basic_trials:
                return {
                    "success": False,
                    "error": f"No trials found for condition: {condition}",
                    "data": [],
                }

            self.logger.info(f"📋 Found {len(basic_trials)} trials in search")

            # Step 2: Extract NCT IDs and fetch full study data
            nct_ids = []
            for trial in basic_trials:
                try:
                    # Extract NCT ID from search result
                    protocol_section = trial.get("protocolSection", {})
                    identification = protocol_section.get("identificationModule", {})
                    nct_id = identification.get("nctId")
                    if nct_id:
                        nct_ids.append(nct_id)
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"Could not extract NCT ID from trial: {e}")
                    continue

            if not nct_ids:
                self.logger.warning("No valid NCT IDs found in search results")
                return {
                    "success": False,
                    "error": "No valid NCT IDs found in search results",
                    "data": [],
                }

            self.logger.info(f"📥 Fetching full study data for {len(nct_ids)} NCT IDs")

            # Step 3: Fetch full study data for each NCT ID concurrently
            full_trials = []
            successful_fetches = 0

            # Create tasks for concurrent fetching
            tasks = []
            for i, nct_id in enumerate(nct_ids):
                task = create_task(
                    task_id=f"full_fetch_{i}",
                    func=self._fetch_single_trial_data,
                    nct_id=nct_id
                )
                tasks.append(task)

            # Execute tasks concurrently
            fetcher_pool = get_fetcher_pool()
            task_queue = TaskQueue(max_workers=fetcher_pool.max_workers, name="FullDataFetcherQueue")

            def progress_callback(completed, total, result):
                nct_id = nct_ids[int(result.task_id.split('_')[2])]
                if result.success and result.result:
                    self.logger.debug(f"✅ Fetched full data for {nct_id}")
                else:
                    self.logger.warning(f"❌ No data returned for {nct_id}")

            task_results = task_queue.execute_tasks(tasks, progress_callback=progress_callback)

            # Process results
            for task_result in task_results:
                nct_id = nct_ids[int(task_result.task_id.split('_')[2])]

                if task_result.success and task_result.result:
                    full_trials.append(task_result.result)
                    successful_fetches += 1
                else:
                    # Include basic trial data if full fetch fails
                    basic_trial = next((t for t in basic_trials if
                                       t.get("protocolSection", {}).get("identificationModule", {}).get("nctId") == nct_id), None)
                    if basic_trial:
                        full_trials.append(basic_trial)
                    if task_result.error:
                        self.logger.error(f"❌ Failed to fetch full data for {nct_id}: {task_result.error}")

            self.logger.info(f"📊 Concurrent full data fetch complete: {successful_fetches}/{len(nct_ids)} successful")

            # Step 4: Validate data completeness
            data_quality = self._assess_data_quality(full_trials)

            return {
                "success": True,
                "type": "condition_search_with_full_data",
                "data": full_trials,
                "metadata": {
                    "condition": condition,
                    "total_found": len(basic_trials),
                    "limit": limit,
                    "nct_ids_extracted": len(nct_ids),
                    "full_data_fetched": successful_fetches,
                    "data_quality": data_quality,
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
        """Fetch multiple trials by NCT IDs using optimized batch processing."""
        try:
            self.logger.info(f"📥 Fetching {len(nct_ids)} trials using batch processing")

            # Get concurrency configuration from CLI args
            cli_args = self._get_cli_args()
            max_workers = 8  # Default batch size
            if cli_args:
                # Extract concurrency settings from CLI args if available
                max_workers = getattr(cli_args, 'max_workers', 8)

            self.logger.info(f"🚀 Using batch processing with {max_workers} concurrent workers")

            # Use the optimized batch processing function
            batch_results = get_full_studies_batch(nct_ids, max_workers=max_workers)

            # Process results
            successful_trials = []
            failed_trials = []

            for nct_id, result in batch_results.items():
                if isinstance(result, dict) and "error" not in result:
                    successful_trials.append(result)
                    self.logger.debug(f"✅ Fetched: {nct_id}")
                else:
                    failed_trials.append(nct_id)
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                    self.logger.warning(f"❌ Failed to fetch {nct_id}: {error_msg}")

            success_rate = len(successful_trials) / len(nct_ids) if nct_ids else 0

            self.logger.info(
                f"📊 Batch fetch complete: {len(successful_trials)}/{len(nct_ids)} successful"
            )

            return {
                "success": len(successful_trials) > 0,
                "type": "multiple_trials_batch",
                "data": successful_trials,
                "metadata": {
                    "total_requested": len(nct_ids),
                    "successful": len(successful_trials),
                    "failed": len(failed_trials),
                    "success_rate": success_rate,
                    "failed_trials": failed_trials,
                    "batch_workers": max_workers,
                    "processing_method": "batch_api_calls"
                },
            }

        except Exception as e:
            self.logger.error(f"Unexpected error in batch fetch: {e}")
            return {"success": False, "error": f"Batch fetch error: {e}", "data": []}

    def _fetch_single_trial_data(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single trial's data (used by concurrent fetcher)."""
        try:
            trial = get_full_study(nct_id)
            return trial
        except Exception as e:
            self.logger.debug(f"Failed to fetch {nct_id}: {e}")
            return None

    def _save_results(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save fetch results to file in NDJSON format."""
        try:
            output_file = Path(output_path)

            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save as NDJSON (one JSON object per line)
            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

            self.logger.info(f"💾 Results saved to: {output_file} (NDJSON format)")

        except Exception as e:
            self.logger.error(f"Failed to save results to {output_path}: {e}")
            raise

    def _assess_data_quality(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the completeness and quality of trial data."""
        if not trials:
            return {"status": "no_data", "completeness_score": 0.0}

        total_trials = len(trials)
        complete_trials = 0
        quality_scores = []

        for trial in trials:
            score = self._calculate_trial_completeness(trial)
            quality_scores.append(score)
            if score >= 0.8:  # Consider trial complete if 80%+ of expected fields present
                complete_trials += 1

        avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        quality_assessment = {
            "status": "good" if avg_score >= 0.8 else "partial" if avg_score >= 0.5 else "incomplete",
            "completeness_score": round(avg_score, 2),
            "complete_trials": complete_trials,
            "total_trials": total_trials,
            "completeness_percentage": round((complete_trials / total_trials) * 100, 1) if total_trials > 0 else 0.0
        }

        if quality_assessment["status"] != "good":
            self.logger.warning(
                f"⚠️  Trial data quality assessment: {quality_assessment['status']} "
                f"({quality_assessment['completeness_percentage']}% complete, "
                f"avg score: {quality_assessment['completeness_score']})"
            )
            self.logger.info(
                "💡 For better summarization quality, consider using specific NCT IDs "
                "instead of condition search to get complete clinical trial data"
            )

        return quality_assessment

    def _calculate_trial_completeness(self, trial: Dict[str, Any]) -> float:
        """Calculate completeness score for a single trial (0.0 to 1.0)."""
        if not isinstance(trial, dict):
            return 0.0

        protocol_section = trial.get("protocolSection", {})
        if not isinstance(protocol_section, dict):
            return 0.0

        # Define expected fields for a complete trial
        expected_fields = {
            "identificationModule": ["nctId", "briefTitle", "officialTitle"],
            "statusModule": ["overallStatus", "startDateStruct", "completionDateStruct"],
            "descriptionModule": ["briefSummary", "detailedDescription"],
            "eligibilityModule": ["eligibilityCriteria", "minimumAge", "maximumAge", "sex"],
            "conditionsModule": ["conditions"],
            "armsInterventionsModule": ["interventions"],
            "designModule": ["studyType", "phases", "primaryPurpose"],
            "sponsorCollaboratorsModule": ["leadSponsor"]
        }

        total_fields = 0
        present_fields = 0

        for module_name, fields in expected_fields.items():
            module = protocol_section.get(module_name, {})
            if isinstance(module, dict):
                for field in fields:
                    total_fields += 1
                    if field in module and module[field] is not None:
                        present_fields += 1
            else:
                total_fields += len(fields)

        return present_fields / total_fields if total_fields > 0 else 0.0

    def _get_cli_args(self):
        """Get CLI arguments from the current execution context."""
        # This will be set by the CLI script when calling execute()
        return getattr(self, '_cli_args', None)

    def _set_cli_args(self, args):
        """Set CLI arguments for concurrency configuration."""
        self._cli_args = args
