class ModelConfig:
    M_VOCAB_SZ = 20000
    M_MAX_LEN = 100
    M_ATT_HEADS = 2
    M_DIM_EMB = 128
    M_DIM_FFN = 512
    M_WARMUP_STEPS = 2000


class TrainingConfig:
    T_BATCH_SIZE = 256
    T_EPOCHS = 20
    T_DATASET = ["data/rap.txt"]
