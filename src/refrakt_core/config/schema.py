from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    wrapper: Optional[str] = None
    transform: Optional[Union[str, List]] = None

class DataLoaderConfig(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class LossConfig(BaseModel):
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    components: Optional[Dict[str, Dict[str, Any]]] = None

class OptimizerConfig(BaseModel):
    name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    components: Optional[Dict[str, Dict[str, Any]]] = None

class SchedulerConfig(BaseModel):
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None

class TrainerConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class RefraktConfig(BaseModel):
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    trainer: TrainerConfig