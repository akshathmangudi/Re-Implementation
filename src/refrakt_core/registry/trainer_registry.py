TRAINER_REGISTRY = {}
_imported = False

from refrakt_core.logging import get_global_logger

def register_trainer(name):
    def decorator(cls):
        logger = get_global_logger()
        logger.debug(f"Registering trainer: {name}")
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator


def get_trainer(name):
    global _imported
    if not _imported:
        # Trigger import of trainers
        import refrakt_core.trainer
        _imported = True
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name]  # Return the class, not an instance

def log_registry_id():
    logger = get_global_logger()
    logger.debug(f"TRAINER REGISTRY ID: {id(TRAINER_REGISTRY)}")
