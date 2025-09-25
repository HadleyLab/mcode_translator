# CLI package for mCODE translator command-line interfaces

# Import CLI modules to make them available in the src.cli namespace
from . import click_cli
from . import data_commands
from . import data_downloader
from . import patients_commands
from . import patients_fetcher
from . import patients_processor
from . import patients_summarizer
from . import test_commands
from . import test_runner
from . import trials_commands
from . import trials_fetcher
from . import trials_optimizer
from . import trials_processor
from . import trials_summarizer

# Import shared CLI utilities
from ..shared.cli_utils import McodeCLI

# Import workflow classes
from ..workflows.trials_processor_workflow import ClinicalTrialsProcessorWorkflow
from ..workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from ..workflows.trials_summarizer_workflow import TrialsSummarizerWorkflow
from ..workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow
from ..workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from ..workflows.patients_processor_workflow import PatientsProcessorWorkflow
from ..workflows.patients_summarizer_workflow import PatientsSummarizerWorkflow

# Make commonly used classes available at package level
__all__ = [
    'McodeCLI',
    'ClinicalTrialsProcessorWorkflow',
    'TrialsFetcherWorkflow',
    'TrialsSummarizerWorkflow',
    'TrialsOptimizerWorkflow',
    'PatientsFetcherWorkflow',
    'PatientsProcessorWorkflow',
    'PatientsSummarizerWorkflow',
]
