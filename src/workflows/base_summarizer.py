"""
Base Summarizer Workflow - Shared functionality for generating natural language summaries.

This module provides a common base class for both patient and trial summarizers
to reduce code duplication and ensure consistent summarization behavior.
"""

from typing import Any, Dict, List, Optional

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import OncoCoreMemory

from .base_workflow import SummarizerWorkflow, WorkflowResult


class BaseSummarizerWorkflow(SummarizerWorkflow):
    """
    Base class for summarizer workflows with shared functionality.

    Provides common summarization logic, metadata extraction, and CORE memory storage
    for both patient and trial summarizers.
    """

    def __init__(
        self, config: Any, memory_storage: Optional[OncoCoreMemory] = None
    ):
        """
        Initialize the base summarizer workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()

    def execute_summarization(
        self,
        data_items: List[Dict[str, Any]],
        store_in_memory: bool = False,
        data_type: str = "data"  # "patients" or "trials"
    ) -> WorkflowResult:
        """
        Execute summarization for a list of data items.

        Args:
            data_items: List of data items to summarize
            store_in_memory: Whether to store results in CORE memory
            data_type: Type of data being summarized ("patients" or "trials")

        Returns:
            WorkflowResult: Summarization results
        """
        try:
            self.logger.info(f"Starting {data_type} summarizer workflow execution")

            if not data_items:
                return self._create_result(
                    success=False,
                    error_message=f"No {data_type} data provided for summarization.",
                )

            # Generate summaries
            processed_items = []
            successful_count = 0
            failed_count = 0

            self.logger.info(f"📝 Generating summaries for {len(data_items)} {data_type}")

            for item in data_items:
                try:
                    # Generate summary using subclass-specific logic
                    summary = self._generate_item_summary(item)

                    # Create processed item with summary
                    processed_item = item.copy()
                    if "McodeResults" not in processed_item:
                        processed_item["McodeResults"] = {}
                    processed_item["McodeResults"]["natural_language_summary"] = summary

                    # Store in CORE Memory if requested
                    if store_in_memory and self.memory_storage:
                        self._store_item_summary(item, summary, data_type)

                    processed_items.append(processed_item)
                    successful_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to summarize {data_type} item: {e}")
                    # Add error information to item
                    error_item = item.copy()
                    error_item["SummaryError"] = str(e)
                    processed_items.append(error_item)
                    failed_count += 1

            # Calculate success rate
            total_count = len(data_items)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Summarization complete: {successful_count}/{total_count} {data_type} successful"
            )

            return self._create_result(
                success=successful_count > 0,
                data=processed_items,
                metadata={
                    f"total_{data_type}": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, f"{data_type} summarization")

    def process_single_item(
        self, item: Dict[str, Any], **kwargs: Any
    ) -> WorkflowResult:
        """
        Process a single item for summarization.

        Args:
            item: Data item to summarize
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Summarization result
        """
        result = self.execute_summarization([item], **kwargs)

        # Return single item result
        if (
            result.success
            and result.data
            and isinstance(result.data, list)
            and len(result.data) > 0
        ):
            return self._create_result(
                success=True, data=result.data[0], metadata=result.metadata
            )
        else:
            return result

    def _generate_item_summary(self, item: Dict[str, Any]) -> str:
        """
        Generate summary for a single item.

        This method should be implemented by subclasses to provide
        data-type specific summarization logic.

        Args:
            item: Data item to summarize

        Returns:
            str: Generated summary
        """
        raise NotImplementedError("Subclasses must implement _generate_item_summary")

    def _store_item_summary(
        self, item: Dict[str, Any], summary: str, data_type: str
    ) -> None:
        """
        Store item summary in CORE memory.

        This method should be implemented by subclasses to provide
        data-type specific storage logic.

        Args:
            item: Original data item
            summary: Generated summary
            data_type: Type of data ("patients" or "trials")
        """
        raise NotImplementedError("Subclasses must implement _store_item_summary")