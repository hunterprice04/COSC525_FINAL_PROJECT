from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    VOCAB_SZ: int = field(default=20000)
    MAX_LEN: int = field(default=100)
    ATT_HEADS: int = field(default=2)
    DIM_EMB: int = field(default=128)
    DIM_FFN: int = field(default=512)
    WARMUP_STEPS: int = field(default=2000)


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = field(default=256)
    EPOCHS: int = field(default=20)
    DATASET: list = field(default_factory=list)


class Config:
    def __init__(self, model_config: dict, training_config: dict):
        self.MODEL = ModelConfig(**model_config)
        self.TRAINING = TrainingConfig(**training_config)
