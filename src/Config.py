from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    VOCAB_SZ: int = field(default_factory=int)
    MAX_LEN: int = field(default_factory=int)
    ATT_HEADS: int = field(default_factory=int)
    DIM_EMB: int = field(default_factory=int)
    DIM_FFN: int = field(default_factory=int)
    NUM_LAYERS: int = field(default_factory=int)
    WARMUP_STEPS: int = field(default_factory=int)


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = field(default=32)
    EPOCHS: int = field(default=20)
    DATASET: list = field(default_factory=list)


class Config:
    def __init__(self, model_config: dict, training_config: dict):
        self.MODEL = ModelConfig(**model_config)
        self.TRAINING = TrainingConfig(**training_config)
