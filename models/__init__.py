import os
import importlib
from registry.model_registry import register_model

# Make register_model available to all model modules
__all__ = ["register_model"]

def auto_import_models():
    model_dir = os.path.dirname(__file__)
    for filename in os.listdir(model_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{__name__}.{filename[:-3]}"
            importlib.import_module(module_name)

auto_import_models()

