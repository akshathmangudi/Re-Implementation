import os
import traceback
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf

# Add direct imports for dataset and dataloader builders
from refrakt_core.api.builders.dataloader_builder import build_dataloader
from refrakt_core.api.builders.dataset_builder import build_dataset
from refrakt_core.api.builders.trainer_builder import initialize_trainer
from refrakt_core.api.core.utils import build_model_components, import_modules


def test(cfg: Union[str, OmegaConf], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Test/evaluate a model based on the provided configuration.
    
    Args:
        cfg: Either a path to a config file or an OmegaConf object
        model_path: Optional path to a saved model checkpoint
        
    Returns:
        Dict containing evaluation results
    """
    try:
        # Load configuration
        if isinstance(cfg, str):
            config = OmegaConf.load(cfg)
        else:
            config = cfg

        modules = import_modules()
        
        # === Build Dataset & DataLoader ===
        print("Building test datasets...")
        test_cfg = OmegaConf.merge(config.dataset, OmegaConf.create({"params": {"train": False}}))
        test_dataset = build_dataset(test_cfg)
        test_loader = build_dataloader(test_dataset, config.dataloader)
        print(f"Test batches: {len(test_loader)}")
        
        # Build model components
        components = build_model_components(config)
        
        # Load model checkpoint if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=components.device)
            components.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        
        # Initialize trainer for evaluation
        trainer = initialize_trainer(
            config, components.model, test_loader, test_loader,  # Use test_loader for both
            components.loss_fn, components.optimizer, components.scheduler,
            components.device, modules, save_dir=None
        )
        
        # Run evaluation
        print("\nRunning evaluation...")
        eval_results = trainer.evaluate()
        
        print("\nEvaluation completed successfully!")
        
        return {
            "model": components.model,
            "evaluation_results": eval_results,
            "config": config
        }
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        traceback.print_exc()
        raise