import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import refrakt_core.datasets
import refrakt_core.transforms
from refrakt_core.datasets import ContrastiveDataset, SuperResolutionDataset
from refrakt_core.registry.dataset_registry import \
    DATASET_REGISTRY  # Add this import
from refrakt_core.registry.dataset_registry import get_dataset
from refrakt_core.registry.transform_registry import get_transform


def build_transform(cfg):
    """Build transform pipeline from config using transform registry"""
    transform_list = []
    
    # Handle different transform configuration styles
    if isinstance(cfg, (list, ListConfig)):
        transform_sequence = cfg
    elif isinstance(cfg, dict) and "views" in cfg:
        transform_sequence = cfg["views"][0]
    else:
        raise ValueError(f"Unsupported transform configuration: {type(cfg)}")
    
    for t in transform_sequence:
        name = t["name"]
        params = t.get("params", {})
        
        # Special handling for transforms with nested transforms
        if name == "RandomApply":
            # Recursively build nested transforms
            nested_transforms_cfg = params.get("transforms", [])
            nested_transforms = build_transform(nested_transforms_cfg).transforms
            transform_list.append(get_transform(
                "RandomApply", 
                transforms=nested_transforms,
                p=params.get("p", 0.5)
            ))
        else:
            # Handle all other transforms through the registry
            transform = get_transform(name, **params)
            
            # Remove the problematic flatten check since it's not needed
            transform_list.append(transform)
    
    # Special handling for paired transforms
    if any(tform.__class__.__name__ == "PairedTransform" for tform in transform_list):
        return transform_list[0]  # Return the paired transform directly
    else:
        return transforms.Compose(transform_list)

def build_dataset(cfg):
    # Convert to native Python types for compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract dataset parameters
    dataset_params = cfg_dict.get("params", {}).copy()
    dataset_name = cfg_dict["name"]
    wrapper_name = cfg_dict.get("wrapper", None)
    
    # Handle transform separately
    transform_cfg = cfg_dict.get("transform", None)
    transform_fn = build_transform(transform_cfg) if transform_cfg else None
    
    # Handle wrapped datasets (e.g., contrastive)
    if wrapper_name:
        # Create base dataset without transform
        base_dataset = get_dataset(dataset_name, **dataset_params)
        
        # Get wrapper class directly from registry
        if wrapper_name not in DATASET_REGISTRY:
            raise ValueError(f"Wrapper dataset '{wrapper_name}' not found in registry")
            
        wrapper_cls = DATASET_REGISTRY[wrapper_name]
        
        # Pass base_dataset as first positional argument
        return wrapper_cls(base_dataset, transform=transform_fn)
    else:
        # For non-wrapped datasets, apply transform directly
        if transform_fn:
            dataset_params["transform"] = transform_fn
        return get_dataset(dataset_name, **dataset_params)

def build_dataloader(dataset, cfg: DictConfig):
    # Convert to native Python types for compatibility
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract parameters from the 'params' sub-dictionary or use top-level
    params = cfg.get("params", cfg)
    
    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=params["shuffle"],
        num_workers=params["num_workers"],
        drop_last=params.get("drop_last", False)
    )