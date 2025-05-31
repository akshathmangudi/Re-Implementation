# transform_registry.py
TRANSFORM_REGISTRY = {}
_imported = False

def register_transform(name):
    """Decorator to register a transform class with the given name."""
    def decorator(cls):
        if name in TRANSFORM_REGISTRY:
            print(f"Warning: Transform '{name}' already registered. Skipping.")
            return cls
        print(f"Registering transform: {name}")
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

print("TRANSFORM_REGISTRY ID:", id(TRANSFORM_REGISTRY))