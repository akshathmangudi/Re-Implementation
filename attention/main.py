import warnings
from src.config import get_config
from src.train import train_model

if __name__ == "__main__":
    warnings.filterwarning("ignore")
    config = get_config()
    train_model(config)
