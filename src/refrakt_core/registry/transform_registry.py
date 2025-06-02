# transform_registry.py
TRANSFORM_REGISTRY = {}
_imported = False

from refrakt_core.logging import get_global_logger

def register_transform(name):
    """Decorator to register a transform class with the given name."""
    def decorator(cls):
        logger = get_global_logger()
        if name in TRANSFORM_REGISTRY:
            logger.debug(f"Warning: Transform '{name}' already registered. Skipping.")
            return cls
        logger.debug(f"Registering transform: {name}")
        TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator

def get_transform(name, *args, **kwargs):
    """Get transform instance by name with optional arguments."""
    global _imported
    if not _imported:
        # Trigger import of transforms module to register custom transforms
        import refrakt_core.transforms
        _imported = True
    
    if name not in TRANSFORM_REGISTRY:
        # Try to find in torchvision transforms as fallback
        try:
            from torchvision import transforms
            if hasattr(transforms, name):
                return getattr(transforms, name)(*args, **kwargs)
        except ImportError:
            pass
        
        available = list(TRANSFORM_REGISTRY.keys()) + ["ToTensor", "Normalize", "Compose"]  # Example torchvision names
        raise ValueError(
            f"Transform '{name}' not found. Available: {available}"
        )
    
    return TRANSFORM_REGISTRY[name](*args, **kwargs)

def log_registry_id():
    logger = get_global_logger()
    logger.debug(f"TRANSFORM REGISTRY ID: {id(TRANSFORM_REGISTRY)}")

