import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the necessary modules to ensure registry is properly initialized
from refrakt_core.registry.model_registry import MODEL_REGISTRY, register_model, get_model
import refrakt_core.models