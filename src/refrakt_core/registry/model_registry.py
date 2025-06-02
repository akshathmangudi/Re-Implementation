MODEL_REGISTRY = {}
_imported = False

from refrakt_core.logging import get_global_logger


def register_model(name):
    def decorator(cls):
        logger = get_global_logger()
        if name in MODEL_REGISTRY:
            logger.debug(f"Warning: Model '{name}' already registered. Skipping.")
            return cls
        logger.debug(f"Registering model: {name}")
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name, *args, **kwargs):
    global _imported
    if not _imported:
        # Trigger import of models
        import refrakt_core.models

        _imported = True
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](*args, **kwargs)


def log_registry_id():
    logger = get_global_logger()
    logger.debug(f"MODEL REGISTRY ID: {id(MODEL_REGISTRY)}")
