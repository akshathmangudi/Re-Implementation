TRAINER_REGISTRY = {}
_imported = False

def register_trainer(name):
    def decorator(cls):
        if name in TRAINER_REGISTRY:
            print(f"Warning: Trainer '{name}' already registered. Skipping.")
            return cls
        print(f"Registering trainer: {name}")
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator

def get_trainer(name, *args, **kwargs):
    global _imported
    if not _imported:
        # Trigger import of trainers
        import refrakt_core.trainer
        _imported = True
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' not found. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name]  # <-- Just return the class, let caller instanti

print("TRAINER_REGISTRY ID:", id(TRAINER_REGISTRY))
