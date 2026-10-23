"""Конфигурация проекта: пути, модель, seed и гиперпараметры."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRAIN_PATHS: dict[str, Path] = {
    "type1": DATA_DIR / "train_type1.jsonl",
    "type2": DATA_DIR / "train_type2.jsonl",
}
TEST_INPUTS_PATH = DATA_DIR / "test_inputs.jsonl"
OUTPUT_PATHS: dict[str, Path] = {
    "type1": PROJECT_ROOT / "outputs_type1.txt",
    "type2": PROJECT_ROOT / "outputs_type2.txt",
}

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_SEQ_LEN = 1024

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

ADAPTER_DIRS: dict[str, Path] = {
    "type1": ARTIFACTS_DIR / "lora_type1",
    "type2": ARTIFACTS_DIR / "lora_type2",
}

GEN_MAX_NEW_TOKENS = 320
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
GEN_REPETITION_PENALTY = 1.05
GEN_NUM_CANDIDATES = 3
FEW_SHOT_COUNT = 3
