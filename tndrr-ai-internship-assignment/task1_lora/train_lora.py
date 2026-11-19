"""Task 1 – LoRA / QLoRA fine-tuning of Qwen2.5-0.5B-Instruct.

Supports:
- Standard LoRA (fp16 / bf16) on GPU or CPU
- 4-bit QLoRA via bitsandbytes (--qlora flag)
- Deterministic training seed (RANDOM_SEED = 42)
- Telemetry recording (GPU model, peak VRAM, wall-clock time, packages, loss)
- Saves adapter weights to lora_adapter/ and metadata to results/train_log.json

Colab quick start (T4 GPU recommended):
    !pip install -q torch transformers peft datasets accelerate bitsandbytes
    !python create_dataset.py
    !python validate_dataset.py
    !python train_lora.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Guard against torchao version mismatch in Google Colab (pre-installed torchao < 0.16.0)
try:
    import torchao  # type: ignore
    import packaging.version  # type: ignore
    if packaging.version.parse(getattr(torchao, "__version__", "0.0.0")) < packaging.version.parse("0.16.0"):
        sys.modules["torchao"] = None
except Exception:
    sys.modules["torchao"] = None

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    EVAL_RESULTS_FILE,
    LORA_OUTPUT_DIR,
    MODEL_ID,
    RANDOM_SEED,
    TRAIN_FILE,
    LoraSettings,
    TrainSettings,
)


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """Read a chat-formatted jsonl file into an HF Dataset."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


def to_training_format(batch, tokenizer, max_len: int):
    """Map a chat batch -> token ids with loss computed ONLY on the assistant reply.

    We render the full ChatML prompt+response, then mask everything up to and
    including the assistant generation header (<|im_start|>assistant\\n) with -100
    so the model is trained strictly to produce the JSON answer.
    """
    input_ids, labels = [], []
    for msgs in batch["messages"]:
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        enc_full = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)
        enc_prompt = tokenizer(prompt_only, add_special_tokens=False)
        n_prompt = len(enc_prompt["input_ids"])

        ids = enc_full["input_ids"]
        # Mask prompt tokens with -100
        cutoff = min(n_prompt, len(ids))
        lab = [-100] * cutoff + ids[cutoff:]
        input_ids.append(ids)
        labels.append(lab)
    return {"input_ids": input_ids, "labels": labels}


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA / QLoRA fine-tuning for Task 1")
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit NormalFloat4 QLoRA quantization")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override max training steps (useful for smoke tests)")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()
    lora_cfg = LoraSettings()
    tr_cfg = TrainSettings()

    num_epochs = args_cli.epochs or tr_cfg.num_epochs
    batch_size = args_cli.batch_size or tr_cfg.per_device_batch_size
    learning_rate = args_cli.lr or tr_cfg.learning_rate

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"

    print("=" * 60)
    print(f"Task 1 Training | Model: {MODEL_ID}")
    print(f"Device: {device_name} | CUDA: {has_cuda} | QLoRA 4-bit: {args_cli.qlora}")
    print(f"Python {platform.python_version()} | torch {torch.__version__}")
    print("=" * 60)

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading configuration
    load_kwargs = {}
    if args_cli.qlora:
        if not has_cuda:
            print("Warning: QLoRA requested without CUDA; falling back to 32-bit CPU loading.")
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
    else:
        if has_cuda:
            load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

    if args_cli.qlora and has_cuda:
        model = prepare_model_for_kbit_training(model)
    else:
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = load_jsonl_as_dataset(TRAIN_FILE)
    print(f"Loaded {len(train_ds)} training examples from {TRAIN_FILE}")
    train_ds = train_ds.map(
        lambda b: to_training_format(b, tokenizer, tr_cfg.max_seq_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters

    total_steps = max(1, (len(train_ds) // (batch_size * tr_cfg.gradient_accumulation_steps)) * num_epochs)
    warmup_steps = max(1, int(tr_cfg.warmup_ratio * total_steps))

    raw_args = {
        "output_dir": str(LORA_OUTPUT_DIR.parent / "trainer_out"),
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": tr_cfg.gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": num_epochs,
        "max_steps": args_cli.max_steps,
        "warmup_steps": warmup_steps,
        "weight_decay": tr_cfg.weight_decay,
        "lr_scheduler_type": tr_cfg.lr_scheduler,
        "logging_steps": tr_cfg.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": has_cuda and not args_cli.qlora and not torch.cuda.is_bf16_supported(),
        "bf16": has_cuda and not args_cli.qlora and torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "dataloader_num_workers": 0,
        "seed": RANDOM_SEED,
    }
    # Pass only arguments accepted by the active transformers version
    filtered_args = {k: v for k, v in raw_args.items() if k in sig}
    training_args = TrainingArguments(**filtered_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    start_wall_time = time.time()
    train_result = trainer.train()
    elapsed_time_sec = round(time.time() - start_wall_time, 2)

    # Save adapter and tokenizer
    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"\nAdapter and tokenizer successfully saved to: {LORA_OUTPUT_DIR}")

    peak_vram_mb = 0.0
    if has_cuda:
        peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)

    # Build truthful run telemetry log
    try:
        import transformers
        transformers_ver = transformers.__version__
    except Exception:
        transformers_ver = "unknown"

    try:
        import peft
        peft_ver = peft.__version__
    except Exception:
        peft_ver = "unknown"

    log_entry = {
        "model_id": MODEL_ID,
        "hardware": {
            "device": device_name,
            "cuda_available": has_cuda,
            "peak_vram_mb": peak_vram_mb,
            "platform": platform.platform(),
        },
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers_ver,
            "peft": peft_ver,
        },
        "training_time_sec": elapsed_time_sec,
        "lora_hyperparameters": {
            "r": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
            "target_modules": list(lora_cfg.target_modules),
            "bias": lora_cfg.bias,
            "qlora_4bit": args_cli.qlora,
        },
        "train_hyperparameters": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": tr_cfg.gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler": tr_cfg.lr_scheduler,
            "seed": RANDOM_SEED,
        },
        "n_train_examples": len(train_ds),
        "global_steps": int(trainer.state.global_step),
        "train_loss_final": float(trainer.state.log_history[-1].get("loss", float("nan"))),
    }

    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_log_path = EVAL_RESULTS_FILE.parent / "train_log.json"
    train_log_path.write_text(json.dumps(log_entry, indent=2), encoding="utf-8")
    print(f"Training log written to {train_log_path}")
    print(json.dumps(log_entry, indent=2))


if __name__ == "__main__":
    main()
