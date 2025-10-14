import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.concurrency import AsyncTaskQueue, create_task
from src.utils.fetcher import (
    get_full_studies_batch,
    get_full_study,
    search_trials,
    search_trials_parallel,
)

from .base_workflow import FetcherWorkflow, WorkflowResult


class TrialsFetcherWorkflow(FetcherWorkflow):
    """
    Workflow for fetching clinical trials from ClinicalTrials.gov.

    Fetches raw trial data without processing or storage to core memory.
    """

    async def execute_async(self, **kwargs: Any) -> WorkflowResult:
        condition = kwargs.get("condition")
        nct_id = kwargs.get("nct_id")
        nct_ids = kwargs.get("nct_ids")
        limit = kwargs.get("limit", 50)
        output_path = kwargs.get("output_path")

        if not self._validate_fetch_params(condition, nct_id, nct_ids):
            raise ValueError(
                "Invalid fetch parameters. Must provide condition, nct_id, or nct_ids."
            )

        if condition:
            results = await self._fetch_by_condition_async(condition, limit)
        elif nct_id:
            results = await self._fetch_single_trial_async(nct_id)
        elif nct_ids:
            results = await self._fetch_multiple_trials_async(nct_ids)
        else:
            raise ValueError("No valid fetch parameters provided.")

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

    def execute(self, **kwargs: Any) -> WorkflowResult:
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
        import asyncio

        return asyncio.run(self.execute_async(**kwargs))

    def _validate_fetch_params(
        self,
        condition: str,
        nct_id: str,
        nct_ids: List[str],
    ) -> bool:
        params = [condition, nct_id, nct_ids]
        provided = [p for p in params if p is not None]
        return len(provided) == 1

    async def _fetch_by_condition_async(self, condition: str, limit: int) -> Dict[str, Any]:
        if limit > 100:
            search_result = search_trials_parallel(
                condition,
                max_results=limit,
                page_size=100,
                max_workers=4,
            )
            basic_trials = search_result.get("studies", [])
        else:
            search_result = search_trials(condition, max_results=limit)
            basic_trials = search_result.get("studies", [])

        if not basic_trials:
            raise ValueError(f"No trials found for condition: {condition}")

        nct_ids = []
        for trial in basic_trials:
            protocol_section = trial.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            nct_id = identification.get("nctId")
            if nct_id:
                nct_ids.append(nct_id)

        if not nct_ids:
            raise ValueError("No valid NCT IDs found in search results")

        full_trials = []
        tasks = []
        for i, nct_id in enumerate(nct_ids):
            task = create_task(
                task_id=f"full_fetch_{i}",
                func=self._fetch_single_trial_data,
                nct_id=nct_id,
            )
            tasks.append(task)

        task_queue = AsyncTaskQueue(max_concurrent=4, name="FullDataFetcherQueue")
        task_results = await task_queue.execute_tasks(tasks)

        for task_result in task_results:
            if task_result.success and task_result.result:
                full_trials.append(task_result.result)

        if not full_trials:
            raise ValueError("No full trial data could be fetched")

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
                "full_data_fetched": len(full_trials),
                "data_quality": data_quality,
            },
        }

    async def _fetch_single_trial_async(self, nct_id: str) -> Dict[str, Any]:
        import asyncio

        trial = await asyncio.to_thread(get_full_study, nct_id)

        if not trial:
            raise ValueError(f"Trial not found: {nct_id}")

        return {
            "success": True,
            "type": "single_trial",
            "data": [trial],
            "metadata": {"nct_id": nct_id},
        }

    def _fetch_single_trial(self, nct_id: str) -> Dict[str, Any]:
        """Fetch a single trial by NCT ID."""
        import asyncio

        return asyncio.run(self._fetch_single_trial_async(nct_id))

    async def _fetch_multiple_trials_async(self, nct_ids: List[str]) -> Dict[str, Any]:
        batch_results = await get_full_studies_batch(nct_ids, max_workers=8)

        successful_trials = []
        failed_trials = []

        for nct_id, result in batch_results.items():
            if isinstance(result, dict) and "error" not in result:
                successful_trials.append(result)
            else:
                failed_trials.append(nct_id)

        if not successful_trials:
            raise ValueError("No trials could be fetched from the provided NCT IDs")

        success_rate = len(successful_trials) / len(nct_ids) if nct_ids else 0

        return {
            "success": True,
            "type": "multiple_trials_batch",
            "data": successful_trials,
            "metadata": {
                "total_requested": len(nct_ids),
                "successful": len(successful_trials),
                "failed": len(failed_trials),
                "success_rate": success_rate,
                "failed_trials": failed_trials,
                "batch_workers": 8,
                "processing_method": "batch_api_calls",
            },
        }

    async def _fetch_single_trial_data(self, nct_id: str) -> Dict[str, Any]:
        import asyncio

        trial = await asyncio.to_thread(get_full_study, nct_id)
        return trial

    def _save_results(self, data: List[Dict[str, Any]], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    def _assess_data_quality(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trials:
            return {"status": "no_data", "completeness_score": 0.0}

        total_trials = len(trials)
        complete_trials = 0
        quality_scores = []

        for trial in trials:
            score = self._calculate_trial_completeness(trial)
            quality_scores.append(score)
            if score >= 0.8:
                complete_trials += 1

        avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        quality_assessment = {
            "status": (
                "good" if avg_score >= 0.8 else "partial" if avg_score >= 0.5 else "incomplete"
            ),
            "completeness_score": round(avg_score, 2),
            "complete_trials": complete_trials,
            "total_trials": total_trials,
            "completeness_percentage": (
                round((complete_trials / total_trials) * 100, 1) if total_trials > 0 else 0.0
            ),
        }

        return quality_assessment

    def _calculate_trial_completeness(self, trial: Dict[str, Any]) -> float:
        protocol_section = trial.get("protocolSection", {})

        expected_fields = {
            "identificationModule": ["nctId", "briefTitle", "officialTitle"],
            "statusModule": [
                "overallStatus",
                "startDateStruct",
                "completionDateStruct",
            ],
            "descriptionModule": ["briefSummary", "detailedDescription"],
            "eligibilityModule": [
                "eligibilityCriteria",
                "minimumAge",
                "maximumAge",
                "sex",
            ],
            "conditionsModule": ["conditions"],
            "armsInterventionsModule": ["interventions"],
            "designModule": ["studyType", "phases", "primaryPurpose"],
            "sponsorCollaboratorsModule": ["leadSponsor"],
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

    def _get_cli_args(self) -> Any:
        return getattr(self, "_cli_args", None)

    def _set_cli_args(self, args: Any) -> None:
        self._cli_args = args


def main(args):
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config.heysol_config import get_config

    get_config()

    workflow = TrialsFetcherWorkflow()

    kwargs = {}
    if hasattr(args, "condition") and args.condition:
        kwargs["condition"] = args.condition
    if hasattr(args, "nct_id") and args.nct_id:
        kwargs["nct_ids"] = [args.nct_id]
    if hasattr(args, "nct_ids") and args.nct_ids:
        kwargs["nct_ids"] = args.nct_ids
    if hasattr(args, "limit") and args.limit:
        kwargs["limit"] = args.limit
    if hasattr(args, "output_path") and args.output_path:
        kwargs["output_path"] = args.output_path

    result = workflow.execute(**kwargs)

    if result.success:
        print("✅ Trials fetch completed successfully!")
        if result.data:
            print(f"Total trials fetched: {len(result.data)}")
        if result.metadata:
            print(f"Metadata: {result.metadata}")
        sys.exit(0)
    else:
        print(f"❌ Trials fetch failed: {result.error_message}")
        sys.exit(1)
