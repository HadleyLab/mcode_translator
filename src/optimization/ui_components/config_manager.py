"""
Configuration Management Components for Modern Optimization UI
"""

from typing import Dict, Any, List
import json
from pathlib import Path

class PromptLibraryManager:
    """Manage prompt library operations"""
    
    def __init__(self, config_path: str = "prompts/prompts_config.json"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load prompt configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_prompts_by_type(self, prompt_type: str) -> List[Dict[str, Any]]:
        """Get all prompts of a specific type"""
        prompts = []
        library = self.config_data.get("prompt_library", {}).get("prompts", {})
        for category in library.values():
            if prompt_type in category:
                prompts.extend(category[prompt_type])
        return prompts
    
    def set_default_prompt(self, prompt_type: str, prompt_name: str) -> None:
        """Set a prompt as default for its type"""
        library = self.config_data["prompt_library"]["prompts"]
        for category in library.values():
            if prompt_type in category:
                for prompt in category[prompt_type]:
                    prompt["default"] = prompt["name"] == prompt_name
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def get_default_prompt(self, prompt_type: str) -> str:
        """Get the default prompt for a type"""
        library = self.config_data["prompt_library"]["prompts"]
        for category in library.values():
            if prompt_type in category:
                for prompt in category[prompt_type]:
                    if prompt.get("default", False):
                        return prompt["name"]
        return ""

class ModelLibraryManager:
    """Manage model library operations"""
    
    def __init__(self, config_path: str = "models/models_config.json"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get all models of a specific type"""
        models = []
        library = self.config_data.get("model_library", {}).get("models", {})
        for category in library.values():
            for subcategory, model_list in category.items():
                if subcategory == model_type:
                    models.extend(model_list)
        return models
    
    def set_default_model(self, model_name: str) -> None:
        """Set a model as default"""
        library = self.config_data["model_library"]["models"]
        for category in library.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    model["default"] = model["name"] == model_name
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def get_default_model(self) -> str:
        """Get the default model"""
        library = self.config_data["model_library"]["models"]
        for category in library.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    if model.get("default", False):
                        return model["name"]
        return ""