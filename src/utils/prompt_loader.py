"""
Prompt Loader Utility

This module provides functionality to load prompts from the file-based prompt library
instead of using hardcoded prompts in the source code.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import centralized logging configuration
from src.utils.logging_config import get_logger

# Use centralized logger
logger = get_logger(__name__)


class PromptLoader:
    """Utility class for loading prompts from the file-based prompt library"""

    # Strict placeholder requirements by prompt type
    PROMPT_REQUIREMENTS = {
        "NLP_EXTRACTION": ["{clinical_text}"],
        "MCODE_MAPPING": ["{entities_json}", "{trial_context}"],
        "DIRECT_MCODE": ["{clinical_text}"],
    }

    # Expected JSON response structures by prompt type
    RESPONSE_STRUCTURE_REQUIREMENTS = {
        "NLP_EXTRACTION": {
            "required_fields": ["entities"],
            "field_types": {"entities": "array"},
            "error_message": "NLP extraction prompts must produce JSON with 'entities' array field",
        },
        "MCODE_MAPPING": {
            "required_fields": ["mcode_mappings"],
            "field_types": {"mcode_mappings": "array"},
            "error_message": "mCODE mapping prompts must produce JSON with 'mcode_mappings' array field",
        },
        "DIRECT_MCODE": {
            "required_fields": ["mcode_mappings"],
            "field_types": {"mcode_mappings": "array"},
            "error_message": "Direct mCODE mapping prompts must produce JSON with 'mcode_mappings' array field",
        },
    }

    def __init__(self, prompts_config_path: str = "prompts/prompts_config.json"):
        self.prompts_config_path = Path(prompts_config_path)
        self.prompts_config = self._load_prompts_config()
        self.pipelines_config = self._load_pipelines_config()

    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load the prompts configuration JSON file and flatten the structure"""
        try:
            if not self.prompts_config_path.exists():
                raise FileNotFoundError(
                    f"Prompts config file not found: {self.prompts_config_path}"
                )

            with open(self.prompts_config_path, "r") as f:
                config_data = json.load(f)

            # Handle the new flat structure: { "prompts": { "category": [prompt1, prompt2, ...] } }
            if "prompts" in config_data:
                prompts_config = config_data["prompts"]
                flattened_config = {}

                # Iterate through categories and their prompt lists
                for category, prompt_list in prompts_config.items():
                    for prompt_info in prompt_list:
                        prompt_name = prompt_info["name"]
                        flattened_config[prompt_name] = prompt_info

                return flattened_config
            else:
                logger.warning(
                    "Prompt library structure not found in config, using raw config"
                )
                return config_data

        except Exception as e:
            logger.error(f"Failed to load prompts config: {str(e)}")
            return {}

    def _load_pipelines_config(self) -> Dict[str, Any]:
        """Load the pipelines configuration from the JSON file"""
        try:
            if not self.prompts_config_path.exists():
                raise FileNotFoundError(
                    f"Prompts config file not found: {self.prompts_config_path}"
                )

            with open(self.prompts_config_path, "r") as f:
                config_data = json.load(f)

            # Extract pipelines configuration
            if "pipelines" in config_data:
                return config_data["pipelines"]
            else:
                logger.warning("Pipelines configuration not found in config")
                return {}

        except Exception as e:
            logger.error(f"Failed to load pipelines config: {str(e)}")
            return {}

    def get_prompt(self, prompt_key: str, **format_kwargs) -> str:
        """
        Get a prompt by key, optionally formatting it with provided arguments

        Args:
            prompt_key: The key identifying the prompt in the config
            **format_kwargs: Keyword arguments to format the prompt template

        Returns:
            The loaded prompt content, formatted if arguments provided

        Raises:
            ValueError: If prompt validation fails or required placeholders are missing
        """
        try:
            # Load from file (no caching - always read from disk)
            prompt_config = self.prompts_config.get(prompt_key)
            if not prompt_config:
                raise ValueError(f"Prompt key '{prompt_key}' not found in config")

            prompt_file_path = prompt_config.get("prompt_file")
            if not prompt_file_path:
                raise ValueError(
                    f"No prompt_file specified for prompt key '{prompt_key}'"
                )

            # Load prompt content from file - paths are relative to prompts directory
            prompt_file = self.prompts_config_path.parent / prompt_file_path
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

            with open(prompt_file, "r") as f:
                prompt_content = f.read().strip()

            # Validate prompt content against strict requirements
            self._validate_prompt_strict(prompt_content, prompt_config)

            # Validate that prompt produces correct JSON response structure
            self._validate_prompt_response_structure(prompt_content, prompt_config)

            # Format the prompt if arguments provided
            if format_kwargs:
                try:
                    return prompt_content.format(**format_kwargs)
                except KeyError as e:
                    logger.warning(
                        f"Missing format argument {e} for prompt '{prompt_key}'. Using unformatted prompt."
                    )
                    return prompt_content
            else:
                return prompt_content

        except ValueError as e:
            # Re-raise validation errors to be handled by caller
            raise
        except Exception as e:
            logger.error(f"Error loading prompt '{prompt_key}': {str(e)}")
            raise

    def _validate_prompt_strict(
        self, prompt_content: str, prompt_config: Dict[str, Any]
    ) -> None:
        """
        Validate prompt content against strict requirements based on prompt type

        Args:
            prompt_content: The loaded prompt content
            prompt_config: Prompt configuration metadata

        Raises:
            ValueError: If prompt fails validation against required placeholders
        """
        prompt_type = prompt_config.get("prompt_type")

        if not prompt_type:
            logger.warning(
                f"Prompt type not specified in config, skipping validation for prompt"
            )
            return

        required_placeholders = self.PROMPT_REQUIREMENTS.get(prompt_type)

        if not required_placeholders:
            logger.warning(
                f"No validation requirements defined for prompt type '{prompt_type}'"
            )
            return

        # Check for required placeholders
        missing_placeholders = []
        for placeholder in required_placeholders:
            if placeholder not in prompt_content:
                missing_placeholders.append(placeholder)

        if missing_placeholders:
            error_message = (
                f"TEMPLATE VALIDATION FAILED for '{prompt_config.get('name', 'unknown')}'. "
                f"Missing required placeholders: {missing_placeholders}. "
                f"Prompt type '{prompt_type}' requires: {required_placeholders}. "
                f"This prompt cannot be loaded."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        logger.debug(f"Prompt '{prompt_config.get('name')}' validation completed")

    def _validate_prompt_response_structure(
        self, prompt_content: str, prompt_config: Dict[str, Any]
    ) -> None:
        """
        Validate that prompt produces correct JSON response structure based on prompt type

        Args:
            prompt_content: The loaded prompt content
            prompt_config: Prompt configuration metadata

        Raises:
            ValueError: If prompt fails validation against required response structure
        """
        prompt_type = prompt_config.get("prompt_type")

        if not prompt_type:
            logger.warning(
                f"Prompt type not specified in config, skipping response structure validation for prompt"
            )
            return

        response_requirements = self.RESPONSE_STRUCTURE_REQUIREMENTS.get(prompt_type)

        if not response_requirements:
            logger.warning(
                f"No response structure requirements defined for prompt type '{prompt_type}'"
            )
            return

        # Extract JSON structure examples from prompt content
        json_examples = self._extract_json_examples_from_prompt(prompt_content)

        if not json_examples:
            logger.warning(
                f"Prompt '{prompt_config.get('name', 'unknown')}' contains no JSON examples. "
                f"Cannot validate response structure for prompt type '{prompt_type}'. "
                f"Expected structure: {response_requirements['required_fields']}"
            )
            return

        # Validate each JSON example against required structure
        validation_errors = []
        for json_example in json_examples:
            try:
                # First try to parse as JSON - this will work for complete JSON examples
                parsed_json = json.loads(json_example)
                errors = self._validate_json_structure(
                    parsed_json, response_requirements
                )
                if errors:
                    validation_errors.extend(errors)
            except json.JSONDecodeError:
                # If JSON parsing fails, try structural validation using regex
                # This handles template content with placeholders and comments
                structural_errors = self._validate_json_structure_regex(
                    json_example, response_requirements
                )
                if structural_errors:
                    validation_errors.extend(structural_errors)
                else:
                    # If structural validation passes, it's likely a valid template
                    logger.debug(
                        f"Template validation passed for example with placeholders: {json_example[:100]}..."
                    )

        if validation_errors:
            error_message = (
                f"RESPONSE STRUCTURE VALIDATION FAILED for '{prompt_config.get('name', 'unknown')}'. "
                f"Prompt type '{prompt_type}' requires: {response_requirements['error_message']}. "
                f"Validation errors: {validation_errors}. "
                f"This prompt will be skipped to prevent runtime errors."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        logger.debug(
            f"Prompt '{prompt_config.get('name')}' response structure validation completed"
        )

    def _extract_json_examples_from_prompt(self, prompt_content: str) -> List[str]:
        """
        Extract JSON examples from prompt content for structure validation

        Args:
            prompt_content: The prompt content to analyze

        Returns:
            List of JSON strings found in the prompt
        """
        import re

        # Look for JSON examples in the prompt content
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # JSON code blocks
            r"{{([\s\S]*?)}}",  # Double curly braces (template format)
            r"\{([\s\S]*?)\}",  # Single curly braces (JSON objects)
            r"REQUIRED OUTPUT FORMAT.*?({[\s\S]*?})",  # Output format sections
        ]

        json_examples = []
        logger.debug(
            f"Extracting JSON examples from prompt content (length: {len(prompt_content)})"
        )

        for pattern in json_patterns:
            matches = re.findall(pattern, prompt_content, re.IGNORECASE | re.DOTALL)
            logger.debug(f"Pattern '{pattern}' found {len(matches)} matches")

            for i, match in enumerate(matches):
                # Clean up the match and try to parse as JSON
                cleaned_match = match.strip()
                logger.debug(f"Match {i}: '{cleaned_match[:50]}...'")

                # Handle template syntax: convert ALL double braces to single braces for validation
                # This is necessary because prompts use {{ }} template syntax but we need { } for JSON validation
                if "{{" in cleaned_match or "}}" in cleaned_match:
                    # Convert all double braces to single braces for JSON validation
                    original_match = cleaned_match
                    cleaned_match = cleaned_match.replace("{{", "{").replace("}}", "}")
                    logger.debug(
                        f"Converted template syntax: '{original_match[:50]}...' -> '{cleaned_match[:50]}...'"
                    )

                if cleaned_match and self._looks_like_json(cleaned_match):
                    json_examples.append(cleaned_match)
                    logger.debug(
                        f"Added JSON example for validation: {cleaned_match[:100]}..."
                    )
                elif cleaned_match:
                    logger.debug(
                        f"Found text that doesn't look like JSON: {cleaned_match[:100]}..."
                    )

        logger.debug(f"Extracted {len(json_examples)} JSON examples from prompt")
        return json_examples

    def _looks_like_json(self, text: str) -> bool:
        """
        Check if text looks like a JSON object or array

        Args:
            text: Text to check

        Returns:
            True if text appears to be JSON
        """
        text = text.strip()
        # More lenient check - just look for JSON-like structure with placeholders
        # This allows for template content with placeholders and comments
        return (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        )

    def _validate_json_structure(
        self, parsed_json: Dict[str, Any], requirements: Dict[str, Any]
    ) -> List[str]:
        """
        Validate JSON structure against requirements

        Args:
            parsed_json: Parsed JSON object to validate
            requirements: Structure requirements

        Returns:
            List of validation errors, empty if valid
        """
        errors = []

        # Check required fields
        for field in requirements.get("required_fields", []):
            if field not in parsed_json:
                errors.append(f"Missing required field: '{field}'")

        # Check field types
        field_types = requirements.get("field_types", {})
        for field, expected_type in field_types.items():
            if field in parsed_json:
                actual_value = parsed_json[field]
                if expected_type == "array" and not isinstance(actual_value, list):
                    errors.append(
                        f"Field '{field}' must be an array, got {type(actual_value).__name__}"
                    )
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    errors.append(
                        f"Field '{field}' must be an object, got {type(actual_value).__name__}"
                    )
                elif expected_type == "string" and not isinstance(actual_value, str):
                    errors.append(
                        f"Field '{field}' must be a string, got {type(actual_value).__name__}"
                    )

        return errors

    def _validate_json_structure_regex(
        self, json_text: str, requirements: Dict[str, Any]
    ) -> List[str]:
        """
        Validate JSON structure using regex for template content with placeholders

        Args:
            json_text: JSON-like text with placeholders
            requirements: Structure requirements

        Returns:
            List of validation errors, empty if valid
        """
        import re

        errors = []

        # Check for required fields using regex (allowing for template placeholders)
        for field in requirements.get("required_fields", []):
            # Look for field pattern with optional quotes and possible template syntax
            field_pattern = rf'"{field}"\s*:\s*[^,}}]+'
            if not re.search(field_pattern, json_text):
                errors.append(f"Missing required field: '{field}'")

        # For field types, we can't validate templates with placeholders, so skip type validation
        # The actual validation will happen at runtime when the LLM produces real JSON

        return errors

    def reload_config(self) -> None:
        """Reload the prompts configuration from disk"""
        self.prompts_config = self._load_prompts_config()
        logger.info("Prompt configuration reloaded from disk")

    def get_prompt_metadata(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific prompt"""
        return self.prompts_config.get(prompt_key)

    def get_prompts_by_pipeline(
        self, pipeline_key: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get prompts organized by prompt type for a specific pipeline

        Args:
            pipeline_key: The pipeline key (e.g., "NlpMcodePipeline", "McodePipeline")

        Returns:
            Dictionary mapping prompt types to lists of prompt metadata
        """
        if pipeline_key not in self.pipelines_config:
            raise ValueError(f"Pipeline '{pipeline_key}' not found in configuration")

        pipeline_config = self.pipelines_config[pipeline_key]
        required_types = pipeline_config.get("required_prompt_types", [])

        pipeline_prompts = {}
        for prompt_name, prompt_metadata in self.prompts_config.items():
            prompt_type = prompt_metadata.get("prompt_type")
            compatible_pipelines = prompt_metadata.get("compatible_pipelines", [])

            # Check if prompt is compatible with this pipeline and of required type
            if prompt_type in required_types and pipeline_key in compatible_pipelines:
                if prompt_type not in pipeline_prompts:
                    pipeline_prompts[prompt_type] = []
                pipeline_prompts[prompt_type].append(prompt_metadata)

        return pipeline_prompts

    def get_pipeline_config(self, pipeline_key: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pipeline

        Args:
            pipeline_key: The pipeline key

        Returns:
            Pipeline configuration dictionary
        """
        if pipeline_key not in self.pipelines_config:
            raise ValueError(f"Pipeline '{pipeline_key}' not found in configuration")

        return self.pipelines_config[pipeline_key].copy()

    def list_available_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available pipelines with their configuration

        Returns:
            Dictionary mapping pipeline keys to pipeline configurations
        """
        return self.pipelines_config.copy()

    def list_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available prompts with their metadata

        Returns:
            Dictionary mapping prompt names to prompt metadata
        """
        return self.prompts_config.copy()


# Global instance for easy access
prompt_loader = PromptLoader()


def load_prompt(prompt_key: str, **format_kwargs) -> str:
    """
    Convenience function to load a prompt using the global loader

    Args:
        prompt_key: The key identifying the prompt in the config
        **format_kwargs: Keyword arguments to format the prompt template

    Returns:
        The loaded prompt content
    """
    return prompt_loader.get_prompt(prompt_key, **format_kwargs)


def reload_prompts_config() -> None:
    """Reload the prompts configuration using the global loader"""
    prompt_loader.reload_config()


# Example usage
if __name__ == "__main__":
    # Test the prompt loader
    try:
        # Load a specific prompt
        extraction_prompt = load_prompt("generic_extraction")
        print(f"Loaded extraction prompt: {extraction_prompt[:100]}...")

    except Exception as e:
        print(f"Error testing prompt loader: {str(e)}")
