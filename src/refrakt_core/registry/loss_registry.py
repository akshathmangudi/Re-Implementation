LOSS_REGISTRY = {}
_imported = False

def register_loss(name):
    def decorator(cls_or_fn):
        if name in LOSS_REGISTRY:
            print(f"Warning: Loss '{name}' already registered. Skipping.")
            return cls_or_fn
        print(f"Registering loss: {name}")
        LOSS_REGISTRY[name] = cls_or_fn
        return cls_or_fn
    return decorator

def get_loss(name, *args, **kwargs):
    global _imported
    if not _imported:
        # Auto-import custom losses
        import refrakt_core.losses
        _imported = True
        
        # Add standard PyTorch losses to registry
        import torch.nn as nn
        standard_losses = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'bce': nn.BCELoss,
        }
        
        for loss_name, loss_class in standard_losses.items():
            if loss_name not in LOSS_REGISTRY:
                register_loss(loss_name)(loss_class)

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss '{name}' not found. Available: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[name](*args, **kwargs)

print("LOSS_REGISTRY ID:", id(LOSS_REGISTRY))