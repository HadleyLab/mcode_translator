"""
Dependency Injection Container for mCODE Translator.

This module provides a centralized container for managing dependencies
and creating configured pipeline components.
"""

from typing import Any, Dict, Optional

from src.pipeline import McodePipeline

# Removed unified_pipeline imports - using simplified pipeline directly
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config


class DependencyContainer:
    """
    Centralized dependency injection container.

    Manages the creation and configuration of all pipeline components,
    ensuring proper dependency injection and component lifecycle.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the dependency container.

        Args:
            config: Application configuration
        """
        self.config = config or Config()
        self._components: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}

    def register_component(self, name: str, component: Any, singleton: bool = False):
        """
        Register a component in the container.

        Args:
            name: Component name/key
            component: The component instance
            singleton: Whether this should be a singleton
        """
        if singleton:
            self._singletons[name] = component
        else:
            self._components[name] = component

    def get_component(self, name: str) -> Any:
        """
        Get a component from the container.

        Args:
            name: Component name/key

        Returns:
            The component instance
        """
        if name in self._singletons:
            return self._singletons[name]
        return self._components.get(name)

    def create_mcode_processor(self, **kwargs) -> McodePipeline:
        """
        Create an mCODE processing component.

        Args:
            **kwargs: Configuration parameters for the processor

        Returns:
            Configured McodePipeline processor
        """
        # Extract configuration from kwargs or use defaults
        prompt_name = kwargs.get("prompt_name", "direct_mcode_evidence_based_concise")
        model_name = kwargs.get(
            "model_name", "deepseek-coder"
        )  # Use configured default model

        # Create and return the processor
        return McodePipeline(prompt_name=prompt_name, model_name=model_name)

    def create_memory_storage(self):
        """
        Create a memory storage component.

        Returns:
            Configured McodeMemoryStorage instance
        """
        if "memory_storage" not in self._singletons:
            self._singletons["memory_storage"] = McodeMemoryStorage()
        return self._singletons["memory_storage"]

    def create_clinical_trial_pipeline(
        self,
        processor_config: Optional[Dict[str, Any]] = None,
        include_storage: bool = True,
    ) -> McodePipeline:
        """
        Create a clinical trial processing pipeline.

        Args:
            processor_config: Configuration for the processor component
            include_storage: Whether to include storage component (ignored in simplified architecture)

        Returns:
            Configured McodePipeline
        """
        processor_config = processor_config or {}

        # Create and return simplified pipeline
        return self.create_mcode_processor(**processor_config)

    def create_patient_data_pipeline(
        self,
        processor_config: Optional[Dict[str, Any]] = None,
        include_storage: bool = True,
    ) -> McodePipeline:
        """
        Create a patient data processing pipeline.

        Args:
            processor_config: Configuration for the processor component
            include_storage: Whether to include storage component (ignored in simplified architecture)

        Returns:
            Configured McodePipeline
        """
        # For now, reuse the same processor (could be specialized later)
        processor_config = processor_config or {}
        return self.create_mcode_processor(**processor_config)


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get the global dependency container instance."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def set_container(container: DependencyContainer):
    """Set the global dependency container instance."""
    global _container
    _container = container


def reset_container():
    """Reset the global container (useful for testing)."""
    global _container
    _container = None


# Convenience functions for common pipeline creation
def create_trial_pipeline(
    processor_config: Optional[Dict[str, Any]] = None, include_storage: bool = True
) -> McodePipeline:
    """
    Create a clinical trial processing pipeline using the global container.

    Args:
        processor_config: Processor configuration
        include_storage: Whether to include storage (ignored in simplified architecture)

    Returns:
        Configured pipeline
    """
    return get_container().create_clinical_trial_pipeline(
        processor_config=processor_config, include_storage=include_storage
    )


def create_patient_pipeline(
    processor_config: Optional[Dict[str, Any]] = None, include_storage: bool = True
) -> McodePipeline:
    """
    Create a patient data processing pipeline using the global container.

    Args:
        processor_config: Processor configuration
        include_storage: Whether to include storage (ignored in simplified architecture)

    Returns:
        Configured pipeline
    """
    return get_container().create_patient_data_pipeline(
        processor_config=processor_config, include_storage=include_storage
    )
