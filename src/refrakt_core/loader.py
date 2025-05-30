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

def build_dataset(cfg):
    dataset_cls = getattr(datasets, cfg.name)
    dataset = dataset_cls(**cfg.params)

    wrapper_name = cfg.get("wrapper", None)
    wrapper_cls = WRAPPER_REGISTRY.get(wrapper_name)
    if wrapper_cls is None:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'")

    # Handle new structure: transform.mode and transform.views
    transform_cfg = cfg.get("transform", None)
    transform_fn = None

    if transform_cfg:
        mode = transform_cfg.get("mode", "single")
        views = transform_cfg.get("views", [])

        if mode == "single":
            transform_fn = build_transform(views[0])  # views[0] is the only view list
        else:
            raise NotImplementedError(f"Transform mode '{mode}' is not supported")

    return wrapper_cls(dataset, transform=transform_fn)



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