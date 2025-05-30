from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from refrakt_core.datasets import ContrastiveDataset
from omegaconf import DictConfig
import torch.nn as nn

# Register any wrappers here
WRAPPER_REGISTRY = {
    "contrastive": ContrastiveDataset,
    None: lambda ds, **kwargs: ds
}

def build_transform(transform_cfg):
    """Build transform pipeline from config, handling nested transforms properly."""
    transform_list = []
    
    for t in transform_cfg:
        name = t["name"]
        params = t.get("params", {})
        
        # Handle special case for RandomApply which has nested transforms
        if name == "RandomApply":
            # Extract the nested transforms
            nested_transforms = params.get("transforms", [])
            # Build the nested transform list
            nested_transform_list = []
            for nested_t in nested_transforms:
                nested_name = nested_t["name"]
                nested_params = nested_t.get("params", {})
                nested_t_cls = getattr(transforms, nested_name)
                nested_transform_list.append(nested_t_cls(**nested_params))
            
            # Create RandomApply with the built nested transforms
            p = params.get("p", 0.5)  # default probability
            t_cls = getattr(transforms, name)
            transform_list.append(t_cls(nested_transform_list, p=p))
        else:
            # Handle regular transforms
            t_cls = getattr(transforms, name)
            transform_list.append(t_cls(**params))
    
    return transforms.Compose(transform_list)


def build_dataset(cfg):
    dataset_cls = getattr(datasets, cfg.name)  # e.g., CIFAR10, STL10
    dataset = dataset_cls(**cfg.params)

    # Apply transform wrapper (e.g., ContrastiveDataset)
    wrapper_name = cfg.get("wrapper", None)
    wrapper_cls = WRAPPER_REGISTRY.get(wrapper_name)

    if wrapper_cls is None:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'")
    
    transform_fn = build_transform(cfg.transform) if "transform" in cfg else None
    return wrapper_cls(dataset, transform=transform_fn)


def build_dataloader(dataset, cfg: DictConfig):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        drop_last=cfg.drop_last
    )