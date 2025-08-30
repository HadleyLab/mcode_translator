"""
Prompt and Model Library Interface for mCODE Translator
Provides setters and defaults for prompt templates and model configurations
"""

import sys
import os
from typing import Optional, Dict, Any

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.prompt_loader import PromptLoader
from src.utils.model_loader import ModelLoader, ModelConfig
from src.utils.config import Config
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.pipeline.mcode_mapper import StrictMcodeMapper
from src.pipeline.nlp_engine import StrictNlpExtractor

# Global instances for easy access
prompt_loader = PromptLoader()
model_loader = ModelLoader()
config = Config()


class PromptModelInterface:
    """
    Interface for managing prompt templates and model configurations
    Provides setters and defaults for the mCODE Translator pipeline
    """
    
    def __init__(self):
        """Initialize the prompt and model interface"""
        self.prompt_loader = prompt_loader
        self.model_loader = model_loader
        self.config = config
    
    # Prompt Management Methods
    
    def set_extraction_prompt(self, prompt_name: str) -> None:
        """
        Set custom extraction prompt template for the NLP engine
        
        Args:
            prompt_name: Name of the prompt template to use from the library
            
        Raises:
            ValueError: If prompt name is invalid or missing required placeholders
        """
        # Validate prompt exists and has required structure
        prompt_template = self.prompt_loader.get_prompt(prompt_name)
        
        # Check for required placeholder
        if "{clinical_text}" not in prompt_template:
            raise ValueError(f"Extraction prompt '{prompt_name}' must contain '{{clinical_text}}' placeholder")
        
        # Store the prompt template for use in pipeline initialization
        self._extraction_prompt_template = prompt_template
        self._extraction_prompt_name = prompt_name
    
    def set_mapping_prompt(self, prompt_name: str) -> None:
        """
        Set custom mapping prompt template for the mCODE mapper
        
        Args:
            prompt_name: Name of the prompt template to use from the library
            
        Raises:
            ValueError: If prompt name is invalid or missing required placeholders
        """
        # Validate prompt exists and has required structure
        prompt_template = self.prompt_loader.get_prompt(prompt_name)
        
        # Check for required placeholders
        if "{entities_json}" not in prompt_template or "{trial_context}" not in prompt_template:
            raise ValueError(f"Mapping prompt '{prompt_name}' must contain '{{entities_json}}' and '{{trial_context}}' placeholders")
        
        # Store the prompt template for use in pipeline initialization
        self._mapping_prompt_template = prompt_template
        self._mapping_prompt_name = prompt_name
    
    def get_extraction_prompt(self) -> Optional[str]:
        """
        Get the current extraction prompt template
        
        Returns:
            Current extraction prompt template or None if not set
        """
        return getattr(self, '_extraction_prompt_template', None)
    
    def get_mapping_prompt(self) -> Optional[str]:
        """
        Get the current mapping prompt template
        
        Returns:
            Current mapping prompt template or None if not set
        """
        return getattr(self, '_mapping_prompt_template', None)
    
    def get_prompt_names(self) -> Dict[str, list]:
        """
        Get all available prompt names from the library
        
        Returns:
            Dictionary with extraction and mapping prompt names
        """
        all_prompts = self.prompt_loader.list_available_prompts()
        
        extraction_prompts = []
        mapping_prompts = []
        
        for name, metadata in all_prompts.items():
            prompt_type = metadata.get('prompt_type')
            if prompt_type == 'NLP_EXTRACTION':
                extraction_prompts.append(name)
            elif prompt_type == 'MCODE_MAPPING':
                mapping_prompts.append(name)
        
        return {
            'extraction': extraction_prompts,
            'mapping': mapping_prompts
        }
    
    # Model Management Methods
    
    def set_model(self, model_key: str) -> None:
        """
        Set model configuration for the pipeline
        
        Args:
            model_key: Key identifying the model in the model library
            
        Raises:
            ValueError: If model configuration is invalid
        """
        # Validate model exists and is properly configured
        model_config = self.model_loader.get_model(model_key)
        
        # Store the model configuration for use in pipeline initialization
        self._model_config = model_config
        self._model_key = model_key
    
    def get_model(self) -> Optional[ModelConfig]:
        """
        Get the current model configuration
        
        Returns:
            Current model configuration or None if not set
        """
        return getattr(self, '_model_config', None)
    
    def get_model_names(self) -> list:
        """
        Get all available model names from the library
        
        Returns:
            List of available model names
        """
        all_models = self.model_loader.list_available_models()
        return list(all_models.keys())
    
    def get_model_parameters(self, model_key: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific model
        
        Args:
            model_key: Key identifying the model in the model library
            
        Returns:
            Dictionary with model parameters
        """
        model_config = self.model_loader.get_model(model_key)
        return model_config.default_parameters
    
    # Pipeline Configuration Methods
    
    def create_pipeline_with_config(self) -> StrictDynamicExtractionPipeline:
        """
        Create a pipeline instance with the configured prompts and models
        
        Returns:
            Configured StrictDynamicExtractionPipeline instance
            
        Raises:
            ValueError: If required configuration is missing
        """
        # Get prompt names if set
        extraction_prompt_name = getattr(self, '_extraction_prompt_name', None)
        mapping_prompt_name = getattr(self, '_mapping_prompt_name', None)
        
        # Create pipeline with configured prompts
        pipeline = StrictDynamicExtractionPipeline(
            extraction_prompt_name=extraction_prompt_name,
            mapping_prompt_name=mapping_prompt_name
        )
        
        # If model is configured, update the pipeline components
        if hasattr(self, '_model_config'):
            model_config = self._model_config
            
            # Update NLP engine with model configuration
            if hasattr(pipeline, 'nlp_engine'):
                pipeline.nlp_engine.model_name = model_config.model_identifier
                pipeline.nlp_engine.base_url = model_config.base_url
                # Update default parameters
                default_params = model_config.default_parameters
                if 'temperature' in default_params:
                    pipeline.nlp_engine.temperature = default_params['temperature']
                if 'max_tokens' in default_params:
                    pipeline.nlp_engine.max_tokens = default_params['max_tokens']
            
            # Update mCODE mapper with model configuration
            if hasattr(pipeline, 'llm_mapper'):
                pipeline.llm_mapper.model_name = model_config.model_identifier
                pipeline.llm_mapper.base_url = model_config.base_url
                # Update default parameters
                default_params = model_config.default_parameters
                if 'temperature' in default_params:
                    pipeline.llm_mapper.temperature = default_params['temperature']
                if 'max_tokens' in default_params:
                    pipeline.llm_mapper.max_tokens = default_params['max_tokens']
        
        return pipeline
    
    def reset_configuration(self) -> None:
        """Reset all configuration to defaults"""
        if hasattr(self, '_extraction_prompt_template'):
            delattr(self, '_extraction_prompt_template')
        if hasattr(self, '_extraction_prompt_name'):
            delattr(self, '_extraction_prompt_name')
        if hasattr(self, '_mapping_prompt_template'):
            delattr(self, '_mapping_prompt_template')
        if hasattr(self, '_mapping_prompt_name'):
            delattr(self, '_mapping_prompt_name')
        if hasattr(self, '_model_config'):
            delattr(self, '_model_config')
        if hasattr(self, '_model_key'):
            delattr(self, '_model_key')


# Global instance for easy access
prompt_model_interface = PromptModelInterface()


def set_extraction_prompt(prompt_name: str) -> None:
    """
    Set custom extraction prompt template globally
    
    Args:
        prompt_name: Name of the prompt template to use from the library
        
    Raises:
        ValueError: If prompt name is invalid or missing required placeholders
    """
    prompt_model_interface.set_extraction_prompt(prompt_name)


def set_mapping_prompt(prompt_name: str) -> None:
    """
    Set custom mapping prompt template globally
    
    Args:
        prompt_name: Name of the prompt template to use from the library
        
    Raises:
        ValueError: If prompt name is invalid or missing required placeholders
    """
    prompt_model_interface.set_mapping_prompt(prompt_name)


def set_model(model_key: str) -> None:
    """
    Set model configuration globally
    
    Args:
        model_key: Key identifying the model in the model library
        
    Raises:
        ValueError: If model configuration is invalid
    """
    prompt_model_interface.set_model(model_key)


def create_configured_pipeline() -> StrictDynamicExtractionPipeline:
    """
    Create a pipeline instance with the global configuration
    
    Returns:
        Configured StrictDynamicExtractionPipeline instance
    """
    return prompt_model_interface.create_pipeline_with_config()


# Example usage
if __name__ == "__main__":
    # Example of how to use the interface
    try:
        # Set prompts
        set_extraction_prompt("generic_extraction")
        set_mapping_prompt("generic_mapping")
        
        # Set model
        set_model("deepseek-coder")
        
        # Create configured pipeline
        pipeline = create_configured_pipeline()
        
        print("Pipeline created successfully with custom configuration")
        
    except Exception as e:
        print(f"Error configuring pipeline: {str(e)}")