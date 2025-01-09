class ConfigDataset():
    PATH_DATASET = "squad"
    REVISION = None



class ConfigModel():
    BATCH_SIZE = 8
    MAX_INPUT_LENGTH = 384
    STRIDE = 128
    MODEL_TOKENIZER = "bert-base-cased"
    MODEL_NAME = "bert-base-cased"
    TRAIN_SIZE = 0.9
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    METRICS = "squad"
    PATH_TENSORBOARD = ""
    PATH_SAVE = "squad"
    NUM_WARMUP_STEPS = 0

class ConfigHelper():
    TOKEN_HF = "xxx"
    AUTHOR = "Chessmen"