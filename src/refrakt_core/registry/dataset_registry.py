DATASET_REGISTRY = {}
_imported = False

def register_dataset(name):
    def decorator(cls):
        if name in DATASET_REGISTRY:
            print(f"Warning: Dataset '{name}' already registered. Skipping.")
            return cls
        print(f"Registering dataset: {name}")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name, *args, **kwargs):
    global _imported
    if not _imported:
        # Trigger import of datasets module to register custom datasets
        import refrakt_core.datasets
        _imported = True
        
    if name not in DATASET_REGISTRY:
        # Try to find in torchvision datasets as fallback
        try:
            from torchvision import datasets
            if hasattr(datasets, name):
                return getattr(datasets, name)(*args, **kwargs)
        except ImportError:
            pass
        
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name](*args, **kwargs)

print("DATASET_REGISTRY ID:", id(DATASET_REGISTRY))