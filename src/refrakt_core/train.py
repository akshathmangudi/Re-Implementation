import os
import gc
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
gc.collect()
torch.cuda.empty_cache()


def main(config_path: str):
    # === Import within function to handle path issues ===
    from refrakt_core.registry.trainer_registry import get_trainer
    from refrakt_core.registry.loss_registry import get_loss
    from refrakt_core.registry.model_registry import get_model
    from refrakt_core.loader import build_dataset, build_dataloader
    import refrakt_core.models
    import refrakt_core.trainer
    import refrakt_core.losses
    import refrakt_core.registry
    import refrakt_core.datasets

    try:
        # Load configuration
        cfg = OmegaConf.load(config_path)
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # === Dataset & DataLoader ===
        print("Building datasets...")
        train_dataset = build_dataset(cfg.dataset)
        # For validation, use same dataset config but with train=False
        val_cfg = OmegaConf.merge(cfg.dataset, OmegaConf.create({"params": {"train": False}}))
        val_dataset = build_dataset(val_cfg)
        
        print("Building data loaders...")
        train_loader = build_dataloader(train_dataset, cfg.dataloader)
        val_loader = build_dataloader(val_dataset, cfg.dataloader)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # === Model ===
        print("Building model...")
        model_params = cfg.model.params or {}
        model = get_model(cfg.model.name, **model_params).to(device)
        print(f"Model: {cfg.model.name} with params: {model_params}")

        # ===== Updated Loss Handling =====
        print("Building loss function...")
        if cfg.loss.get("generator") or cfg.loss.get("discriminator"):
            # Handle GAN-style loss without explicit 'components' key
            loss_fn = {}
            for comp_name in ["generator", "discriminator"]:
                comp_cfg = cfg.loss.get(comp_name)
                if comp_cfg:
                    loss_name = comp_cfg["name"]
                    loss_params = comp_cfg.get("params", {})
                    loss_fn[comp_name] = get_loss(loss_name, **loss_params).to(device)
                    print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
        elif cfg.loss.get("components"):
            # Handle multi-component loss with explicit 'components' key
            loss_fn = {}
            for comp_name, comp_cfg in cfg.loss.components.items():
                loss_name = comp_cfg["name"]
                loss_params = comp_cfg.get("params", {})
                loss_fn[comp_name] = get_loss(loss_name, **loss_params).to(device)
                print(f"Loss ({comp_name}): {loss_name} with params: {loss_params}")
        else:
            # Standard single loss
            loss_name = cfg.loss.name
            loss_params = cfg.loss.get("params", {})
            loss_fn = get_loss(loss_name, **loss_params).to(device)
            print(f"Loss: {loss_name} with params: {loss_params}")
        
        # === Optimizer (updated for GAN support) ===
        print("Building optimizer...")
        opt_map = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adamw": torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }

        if cfg.optimizer.get("generator") or cfg.optimizer.get("discriminator"):
            # Handle GAN-style optimizer without explicit 'components' key
            optimizer = {}
            for comp_name in ["generator", "discriminator"]:
                comp_cfg = cfg.optimizer.get(comp_name)
                if comp_cfg:
                    opt_name = comp_cfg["name"]
                    opt_cls = opt_map.get(opt_name.lower())
                    if not opt_cls:
                        raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")
                    
                    opt_params = comp_cfg.get("params", {})
                    
                    # Get parameters for specific component
                    if comp_name == "generator":
                        parameters = model.generator.parameters()
                    elif comp_name == "discriminator":
                        parameters = model.discriminator.parameters()
                    else:
                        raise ValueError(f"Unknown optimizer component: {comp_name}")
                    
                    optimizer[comp_name] = opt_cls(parameters, **opt_params)
                    print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
                
        elif cfg.optimizer.get("components"):
            # Handle multi-component optimizer (GAN)
            optimizer = {}
            for comp_name, comp_cfg in cfg.optimizer.components.items():
                opt_name = comp_cfg["name"]
                opt_cls = opt_map.get(opt_name.lower())
                if not opt_cls:
                    raise ValueError(f"Unsupported optimizer for {comp_name}: {opt_name}")
                
                opt_params = comp_cfg.get("params", {})
                
                # Get parameters for specific component
                if comp_name == "generator":
                    parameters = model.generator.parameters()
                elif comp_name == "discriminator":
                    parameters = model.discriminator.parameters()
                else:
                    raise ValueError(f"Unknown optimizer component: {comp_name}")
                
                optimizer[comp_name] = opt_cls(parameters, **opt_params)
                print(f"Optimizer ({comp_name}): {opt_name} with params: {opt_params}")
        else:
            # Standard single optimizer (VAE, AE, etc.)
            opt_cls = opt_map.get(cfg.optimizer.name.lower())
            if not opt_cls:
                raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
            
            optimizer_params = cfg.optimizer.params or {}
            optimizer = opt_cls(model.parameters(), **optimizer_params)
            print(f"Optimizer: {cfg.optimizer.name} with params: {optimizer_params}")

        # === Scheduler ===
        scheduler = None
        if cfg.scheduler and cfg.scheduler.name:
            print("Building scheduler...")
            sched_map = {
                "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
                "steplr": torch.optim.lr_scheduler.StepLR,
                "multisteplr": torch.optim.lr_scheduler.MultiStepLR,
                "exponential": torch.optim.lr_scheduler.ExponentialLR,
            }
            scheduler_cls = sched_map.get(cfg.scheduler.name.lower())
            if not scheduler_cls:
                raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")

            scheduler_params = cfg.scheduler.params or {}
            scheduler = scheduler_cls(optimizer, **scheduler_params)
            print(f"Scheduler: {cfg.scheduler.name} with params: {scheduler_params}")

        # === Trainer Initialization ===
        print("Initializing trainer...")
        trainer_cls = get_trainer(cfg.trainer.name)
        trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True) if cfg.trainer.params else {}

        # Extract special parameters
        num_epochs = trainer_params.pop("num_epochs", 1)
        device_param = trainer_params.pop("device", device)
        final_device = device_param if device_param else device

        # Handle different trainer types
        if cfg.trainer.name != "gan":
            # For supervised trainers, pass optimizer class and arguments
            trainer = trainer_cls(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer_cls=opt_cls,  # Pass optimizer class
                optimizer_args=optimizer_params,  # Pass optimizer arguments
                device=final_device,
                scheduler=scheduler,
                **trainer_params
            )
        else:
            # For other trainers (GAN, etc.), pass optimizer instance or dict
            trainer = trainer_cls(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,  # Pass optimizer instance/dict
                device=final_device,
                scheduler=scheduler,
                **trainer_params
            )

        # === Training ===
        print(f"\nStarting training for {num_epochs} epochs...")  # Use extracted num_epochs
        trainer.train(num_epochs=num_epochs)
        
        # === Final Evaluation ===
        print("\nRunning final evaluation...")
        trainer.evaluate()
        
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    args = parser.parse_args()
    
    main(args.config)
    # config_path = "./src/refrakt_core/config/simclr.yaml"
    # main(config_path)