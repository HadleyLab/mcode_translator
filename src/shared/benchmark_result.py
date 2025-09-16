"""
Shared BenchmarkResult class to avoid circular imports.
This class contains the result structure for benchmark runs.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.utils import Loggable

from .models import BenchmarkResult as PydanticBenchmarkResult

# Use the Pydantic BenchmarkResult model
BenchmarkResult = PydanticBenchmarkResult

# Note: For backward compatibility, you can add extension methods here if needed
# The Pydantic model provides model_dump() method for serialization
