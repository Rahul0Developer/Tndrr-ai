"""Task 1 – LoRA fine-tuning of Qwen2.5-0.5B-Instruct (run in Google Colab).

Colab quick start
-----------------
1. Runtime -> Change runtime type -> GPU: T4 (free tier OK).
2. Upload this folder (task1_lora/) or clone your repo, then run:

    !pip install -q torch transformers peft datasets accelerate
    !python create_dataset.py      # skip if data/ already committed
    !python train_lora.py

Outputs: ./lora_adapter/  (PEFT adapter weights) and ./results/train_log.json
(record of the settings ACTUALLY used, for honest reporting).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import (
    EVAL_RESULTS_FILE,
    LORA_OUTPUT_DIR,
    MODEL_ID,
    TRAIN_FILE,
    TEST_FILE,
    LoraSettings,
    TrainSettings,
)


def load_jsonl_as_dataset(path: Path) -> Dataset:
    """Read a chat-formatted jsonl file into an HF Dataset."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return Dataset.from_list(rows)


def to_training_format(batch, tokenizer, max_len: int):
    """Map a chat batch -> token ids with loss only on the assistant reply.

    We render the full ChatML prompt+response, then mask everything up to and
    including the assistant generation header so the model is trained only to
    produce the JSON answer.
    """
    input_ids, labels = [], []
    for msgs in batch["messages"]:
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        enc_full = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)
        n_prompt = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
        ids = enc_full["input_ids"]
        lab = [-100] * min(n_prompt, len(ids)) + ids[min(n_prompt, len(ids)):]
        input_ids.append(ids)
        labels.append(lab)
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    lora_cfg = LoraSettings()
    tr_cfg = TrainSettings()

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Python {platform.python_version()}, torch {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:          # Qwen uses eos as pad
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.enable_input_require_grads()       # required for LoRA grad flow w/ GC

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
    train_ds = train_ds.map(
        lambda b: to_training_format(b, tokenizer, tr_cfg.max_seq_length),
        batched=True, remove_columns=train_ds.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(LORA_OUTPUT_DIR.parent / "trainer_out"),
        per_device_train_batch_size=tr_cfg.per_device_batch_size,
        gradient_accumulation_steps=tr_cfg.gradient_accumulation_steps,
        learning_rate=tr_cfg.learning_rate,
        num_train_epochs=tr_cfg.num_epochs,
        warmup_ratio=tr_cfg.warmup_ratio,
        weight_decay=tr_cfg.weight_decay,
        lr_scheduler_type=tr_cfg.lr_scheduler,
        logging_steps=tr_cfg.logging_steps,
        save_strategy="epoch",
        report_to="none",
        # T4-safe defaults; if you still OOM, set these from fallback_* values
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()

    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"Adapter saved to {LORA_OUTPUT_DIR}")

    # Honest record of what actually ran -> used by README/report tables.
    log = {
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "lora": vars(lora_cfg),
        "train": vars(tr_cfg),
        "n_train_examples": len(train_ds),
        "train_loss_final": float(trainer.state.log_history[-1].get("loss", float("nan"))),
        "global_steps": int(trainer.state.global_step),
    }
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    (EVAL_RESULTS_FILE.parent / "train_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
