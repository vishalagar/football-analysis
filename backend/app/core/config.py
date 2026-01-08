import os
import pathlib

# Centralized BASE_DIR resolution
# This should point to the root of the project (agent-ai-2035)
# backend/app/core/config.py -> core -> app -> backend -> agent-ai-2035
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.parent.resolve()

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mlcc")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
