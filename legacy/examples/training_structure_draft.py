import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

def run_main(config_path: str):
    # === Imports moved inside for isolation ===
    from refrakt_core.registry.trainer_registry import get_trainer
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.loader import build_dataset, build_dataloader
    import refrakt_core.models
    import refrakt_core.trainer
    import refrakt_core.losses

    cfg = OmegaConf.load(config_path)

    def main():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # === Dataset & Dataloader ===
        train_dataset = build_dataset(cfg.dataset)
        val_dataset = build_dataset(cfg.get("val_dataset", cfg.dataset))  # fallback to train

        train_loader = build_dataloader(train_dataset, cfg.dataloader)
        val_loader = build_dataloader(val_dataset, cfg.dataloader)

        # === Model ===
        model_config = cfg.model
        model_params = dict(model_config.get("params", {})) or {
            k: v for k, v in model_config.items() if k != "name"
        }

        model = get_model(model_config.name, **model_params).to(device)

        # === Loss Function ===
        loss_fn = get_loss(cfg.loss.name, **cfg.loss.params).to(device)

        # === Optimizer ===
        opt_name_raw = cfg.optimizer.name.lower()
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,
        }

        if opt_name_raw not in opt_map:
            raise ValueError(f"Unsupported optimizer: {opt_name_raw}")

        optimizer_cls = opt_map[opt_name_raw]

        optimizer_params = dict(cfg.optimizer.get("params", {}))
        optimizer = optimizer_cls(model.parameters(), **optimizer_params)

        # === Scheduler ===
        scheduler = None
        if "scheduler" in cfg and cfg.scheduler:
            sched_name = cfg.scheduler.name.lower()
            if sched_name == "cosine":
                scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
            else:
                scheduler_cls = getattr(torch.optim.lr_scheduler, sched_name)
            scheduler = scheduler_cls(optimizer, **cfg.scheduler.params)

        # === Trainer ===
        trainer_cls = get_trainer(cfg.trainer.name)
        trainer = trainer_cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer_cls=optimizer,
            device=device
        )

        # === Train + Evaluate ===
        trainer.train(num_epochs=cfg.trainer.params.num_epochs)
        trainer.evaluate()

    main()