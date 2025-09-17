"""
Unit tests for DependencyContainer with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.dependency_container import DependencyContainer, get_container, set_container, reset_container
from src.shared.models import PipelineResult


@pytest.mark.mock
class TestDependencyContainer:
    """Test DependencyContainer functionality with mocks."""

    def setup_method(self):
        """Reset container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset container after each test."""
        reset_container()

    @patch('src.core.dependency_container.Config')
    def test_init_with_config(self, mock_config_class):
        """Test initialization with config."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        container = DependencyContainer(mock_config)

        assert container.config == mock_config
        mock_config_class.assert_not_called()  # Should use provided config

    @patch('src.core.dependency_container.Config')
    def test_init_without_config(self, mock_config_class):
        """Test initialization without config creates default."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        container = DependencyContainer()

        assert container.config == mock_config
        mock_config_class.assert_called_once()

    def test_register_component_singleton(self):
        """Test registering a singleton component."""
        container = DependencyContainer()
        component = Mock()

        container.register_component("test", component, singleton=True)

        assert container.get_component("test") == component

    def test_register_component_non_singleton(self):
        """Test registering a non-singleton component."""
        container = DependencyContainer()
        component_factory = Mock()

        # Mock the factory to return different instances
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        component_factory.side_effect = [mock_instance1, mock_instance2]

        container.register_component("test", component_factory, singleton=False)

        comp1 = container.get_component("test")
        comp2 = container.get_component("test")

        # For non-singletons, the container should return the factory function itself
        # The caller is responsible for calling it
        assert comp1 == component_factory
        assert comp2 == component_factory
        assert comp1 == comp2  # Same factory function returned

    def test_get_component_not_found(self):
        """Test getting non-existent component returns None."""
        container = DependencyContainer()

        result = container.get_component("nonexistent")
        assert result is None

    @patch('src.core.dependency_container.McodeMemoryStorage')
    @patch('src.core.dependency_container.Config')
    def test_create_memory_storage(self, mock_config_class, mock_storage_class):
        """Test creating memory storage with mock."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        container = DependencyContainer(mock_config)
        result = container.create_memory_storage()

        assert result == mock_storage
        mock_storage_class.assert_called_once()

    @patch('src.core.dependency_container.create_clinical_trial_pipeline')
    @patch('src.core.dependency_container.McodePipeline')
    @patch('src.core.dependency_container.Config')
    def test_create_clinical_trial_pipeline(self, mock_config_class, mock_pipeline_class, mock_create_pipeline):
        """Test creating clinical trial pipeline with mock."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_create_pipeline.return_value = Mock()

        container = DependencyContainer(mock_config)
        result = container.create_clinical_trial_pipeline()

        assert result is not None
        mock_create_pipeline.assert_called_once()

    @patch('src.pipeline.unified_pipeline.create_patient_data_pipeline')
    @patch('src.core.dependency_container.McodePipeline')
    @patch('src.core.dependency_container.Config')
    def test_create_patient_data_pipeline(self, mock_config_class, mock_pipeline_class, mock_create_pipeline):
        """Test creating patient data pipeline with mock."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_create_pipeline.return_value = Mock()

        container = DependencyContainer(mock_config)
        result = container.create_patient_data_pipeline()

        assert result is not None
        mock_create_pipeline.assert_called_once()


@pytest.mark.mock
class TestGlobalContainerFunctions:
    """Test global container management functions."""

    def teardown_method(self):
        """Reset container after each test."""
        reset_container()

    def test_get_container_creates_default(self):
        """Test get_container creates default when none set."""
        reset_container()

        with patch('src.core.dependency_container.DependencyContainer') as mock_container_class:
            mock_container = Mock()
            mock_container_class.return_value = mock_container

            result = get_container()

            assert result == mock_container
            mock_container_class.assert_called_once()

    def test_set_container(self):
        """Test setting custom container."""
        custom_container = Mock()

        set_container(custom_container)
        result = get_container()

        assert result == custom_container

    def test_reset_container(self):
        """Test resetting container."""
        custom_container = Mock()
        set_container(custom_container)

        reset_container()
        result = get_container()

        assert result != custom_container