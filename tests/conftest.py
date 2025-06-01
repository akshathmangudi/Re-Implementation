import os
import sys

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import refrakt_core.models
# Import the necessary modules to ensure registry is properly initialized
from refrakt_core.registry.model_registry import (MODEL_REGISTRY, get_model,
                                                  register_model)
