"""Central configuration for Task 1: LoRA fine-tuning of Qwen2.5-0.5B-Instruct.

All hyperparameters, paths and schema policy constants live here so that
create_dataset.py, train_lora.py and evaluate.py stay consistent.
"""
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to task1_lora/)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_FILE = DATA_DIR / "train.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"
LORA_OUTPUT_DIR = BASE_DIR / "lora_adapter"
EVAL_RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You convert natural-language requests into strict JSON. "
    "Output ONLY a JSON object with this schema: "
    '{"intent": "<string>", "entities": {<string keys, string values>}}. '
    'Valid intents: set_reminder, create_event, get_weather, send_message, '
    "search_web, query_null. Use null for any missing entity value. "
    "Never add or remove entity keys beyond those required by the request."
)

# ---------------------------------------------------------------------------
# Dataset policy
# ---------------------------------------------------------------------------
N_TRAIN = 300
N_TEST = 60
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# LoRA / training hyperparameters (as specified in the assignment brief)
# ---------------------------------------------------------------------------


@dataclass
class LoraSettings:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainSettings:
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1   # effective batch size = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    logging_steps: int = 10
    # If Colab OOMs, drop to these values (record whichever actually ran):
    fallback_batch_size: int = 2
    fallback_grad_accum: int = 2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 128          # outputs are tiny JSON objects
GENERATION_TEMPERATURE = 0.0  # greedy decoding => deterministic evaluation
