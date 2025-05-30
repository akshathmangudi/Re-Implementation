from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Union, Dict

# Dataset block
class DatasetConfig(BaseModel):
    name: str
    root: str
    train: bool = True
    download: bool = True
    transform: Optional[str] = None  # Points to a registered transform

# DataLoader block
class DataLoaderConfig(BaseModel):
    batch_size: int
    shuffle: bool
    num_workers: int
    drop_last: bool = False

# Model block
class ModelConfig(BaseModel):
    name: str  # e.g. "dino"
    backbone: str
    out_dim: Optional[int] = None
    extra: Dict[str, Union[str, float, int, bool]] = {}  # for model-specific keys

# Loss block
class LossConfig(BaseModel):
    name: str  # e.g. "dino"
    params: Dict[str, Union[float, int, bool]]  # e.g. teacher_temp, center_momentum

# Optimizer block
class OptimizerConfig(BaseModel):
    name: Literal["adam", "sgd", "adamw"]
    lr: float
    weight_decay: Optional[float] = None

# Scheduler block
class SchedulerConfig(BaseModel):
    name: Optional[str] = None  # e.g. cosine
    params: Optional[Dict[str, Union[float, int]]] = None

# Trainer block
class TrainerConfig(BaseModel):
    name: str  # e.g. "dino"
    num_epochs: int
    device: Literal["cuda", "cpu"] = "cuda"

# Root config
class RefraktConfig(BaseModel):
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig]
    trainer: TrainerConfig
