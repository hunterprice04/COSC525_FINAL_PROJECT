from dataclasses import dataclass, field
from typing import Union

from src.data import Embeddings


@dataclass
class TrainingConfig:
    HIDDEN_STATE_SIZE: list = field(default_factory=list)
    EPOCHS: int = field(default_factory=int)
    BATCH_SIZE: int = field(default_factory=int)
    DROPOUT: int = field(default=0.5)
    LR: int = field(default=0.001)
    BUFFER_SIZE: int = field(default=10000)
    SAVE_DIR: str = field(default="models")
    PRED_EVERY: int = field(default=5)
    PRED_LEN: int = field(default=100)
    PRED_TEMP: float = field(default=0.75)


@dataclass
class DataConfig:
    WINDOW_SIZE: int = field(default_factory=int)
    STRIDE: int = field(default_factory=int)
    DATA_PATH: str = field(default_factory=str)


@dataclass
class Config:
    DATA: Union[DataConfig, dict]
    TRAINING: Union[TrainingConfig, dict]
    EMBED: Embeddings = field(default=None)

    def __post_init__(self):
        self.DATA = DataConfig(**self.DATA)
        self.TRAINING = TrainingConfig(**self.TRAINING)
        print(self.DATA)
        print(self.TRAINING)

    def set_embeddings(self, embed: Embeddings):
        self.EMBED = embed
        return self
