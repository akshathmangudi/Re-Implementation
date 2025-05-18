MODEL_REGISTRY = {}
_imported = False

def register_model(name):
    def decorator(cls):
        if name in MODEL_REGISTRY:
            print(f"Warning: Model '{name}' already registered. Skipping.")
            return cls
        print(f"Registering model: {name}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name, *args, **kwargs):
    global _imported
    if not _imported:
        # Lazy-load models on first use
        import models  # this will trigger auto_import_models()
        _imported = True
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](*args, **kwargs)

print("MODEL_REGISTRY ID:", id(MODEL_REGISTRY))
