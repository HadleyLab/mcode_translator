"""
Streamlined Workflow - DEPRECATED

This module has been obsoleted as of 2025-09-17.
Use the inheritance-based workflow pattern from base_workflow.py instead.

Previous functionality:
- StreamlinedTrialProcessor
- StreamlinedWorkflowCoordinator
- create_trial_processor()
- create_workflow_coordinator()

All functionality has been migrated to the standardized workflow hierarchy:
- TrialsProcessorWorkflow (inherits from ProcessorWorkflow)
- PatientsProcessorWorkflow (inherits from ProcessorWorkflow)
- TrialsFetcherWorkflow (inherits from FetcherWorkflow)
- PatientsFetcherWorkflow (inherits from FetcherWorkflow)

Migration benefits:
- Consistent interfaces across all workflow types
- Standardized error handling and logging
- Automatic CORE memory space isolation
- Better testability and maintainability

For migration examples, see the CLI tools in src/cli/ which demonstrate
the new inheritance-based workflow pattern.
"""
