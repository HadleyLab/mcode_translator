# CLI package for mCODE translator command-line interfaces

# Import CLI modules to make them available in the src.cli namespace
# Import shared CLI utilities
from ..shared.cli_utils import McodeCLI
from ..workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from ..workflows.patients_processor_workflow import PatientsProcessorWorkflow
from ..workflows.patients_summarizer_workflow import PatientsSummarizerWorkflow
from ..workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from ..workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow
# Import workflow classes
from ..workflows.trials_processor_workflow import \
    ClinicalTrialsProcessorWorkflow
from ..workflows.trials_summarizer_workflow import TrialsSummarizerWorkflow
from . import (cli, data_commands, data_downloader, patients_commands,
               patients_fetcher, patients_processor, patients_summarizer,
               test_commands, test_runner, trials_commands, trials_fetcher,
               trials_optimizer, trials_processor, trials_summarizer)

# Make commonly used classes available at package level
__all__ = [
    "McodeCLI",
    "ClinicalTrialsProcessorWorkflow",
    "TrialsFetcherWorkflow",
    "TrialsSummarizerWorkflow",
    "TrialsOptimizerWorkflow",
    "PatientsFetcherWorkflow",
    "PatientsProcessorWorkflow",
    "PatientsSummarizerWorkflow",
    "cli",
    "data_commands",
    "data_downloader",
    "patients_commands",
    "patients_fetcher",
    "patients_processor",
    "patients_summarizer",
    "test_commands",
    "test_runner",
    "trials_commands",
    "trials_fetcher",
    "trials_optimizer",
    "trials_processor",
    "trials_summarizer",
]
