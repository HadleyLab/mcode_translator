import json
from pathlib import Path
from typing import Dict

def load_regex_rules() -> Dict[str, str]:
    """Loads regex rules from the patterns_config.json file."""
    config_path = Path(__file__).parent / "patterns_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    rules = {}
    for category in config["patterns"].values():
        for rule_name, rule_details in category.items():
            rules[rule_name] = rule_details["pattern"]
    return rules