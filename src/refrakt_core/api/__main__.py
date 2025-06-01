import argparse
import os

from omegaconf import OmegaConf

from refrakt_core.api.inference import inference
from refrakt_core.api.test import test
from refrakt_core.api.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "inference", "pipeline"],
                        help="Mode to run: train, test, inference, or pipeline")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint (for test/inference)")
    args = parser.parse_args()

    if args.mode == "inference" and not args.model_path:
        raise ValueError("--model-path is required for inference mode")

    if args.mode == "train":
        train(args.config)
    elif args.mode == "test":
        test(args.config, args.model_path)
    elif args.mode == "inference":
        inference(args.config, args.model_path)
    elif args.mode == "pipeline":
        # Load configuration to get model name
        cfg = OmegaConf.load(args.config)
        save_dir = cfg.trainer.params.save_dir
        model_name = cfg.trainer.params.model_name
        
        # Determine model path using model_name from configuration
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        
        # Run training
        print("\n" + "="*50)
        print("🚀 Starting Training Phase")
        print("="*50)
        train(args.config)
        
        # Run testing
        print("\n" + "="*50)
        print("🧪 Starting Testing Phase")
        print("="*50)
        test(args.config, model_path)
        
        # Run inference
        print("\n" + "="*50)
        print("🔮 Starting Inference Phase")
        print("="*50)
        inference(args.config, model_path)