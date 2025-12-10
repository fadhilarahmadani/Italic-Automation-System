# src/config.py
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Model settings
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 256
LABEL_LIST = ["O", "B", "I"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Training settings
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# Early stopping
EARLY_STOPPING_PATIENCE = 3
METRIC_FOR_BEST_MODEL = "f1"

# Logging
LOGGING_STEPS = 50
SAVE_STEPS = 100
EVAL_STEPS = 100

print("✅ Configuration loaded")
