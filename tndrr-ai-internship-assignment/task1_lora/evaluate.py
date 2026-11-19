"""Task 1 – Evaluate base vs LoRA fine-tuned Qwen2.5-0.5B on the held-out set.

Metrics (deterministic, greedy decoding):
  1. Exact Match Accuracy   – parsed JSON equals gold after canonicalisation
                              (recursive key sort; null kept as None).
  2. JSON Validity Rate     – model output contains a parseable JSON object.
  3. Field-level accuracy   – intent accuracy + per-entity exact match.

Run in Colab after train_lora.py:
    !python evaluate.py
Writes results/evaluation_results.json including a `failure_cases` list
(the three+ worst fine-tuned predictions with expected/actual/analysis stubs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    EVAL_RESULTS_FILE,
    LORA_OUTPUT_DIR,
    MAX_NEW_TOKENS,
    MODEL_ID,
    SYSTEM_PROMPT,
    TEST_FILE,
)


def extract_json(text: str) -> dict[str, Any] | None:
    """Deterministic extraction of the first balanced JSON object in text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def canonical(obj: Any) -> str:
    """Canonical serialisation: recursively sorted keys, stable separators."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def field_scores(pred: dict | None, gold: dict) -> tuple[int, int, int, int]:
    """Return (intent_ok, entity_keys_ok, fields_correct, fields_total)."""
    g_int = gold["intent"]
    if not isinstance(pred, dict):
        return 0, 0, 0, len(gold["entities"]) + 1
    intent_ok = int(pred.get("intent") == g_int)
    ents = pred.get("entities")
    keys_ok = int(isinstance(ents, dict) and set(ents.keys()) == set(gold["entities"]))
    correct = intent_ok
    total = 1 + len(gold["entities"])
    if isinstance(ents, dict):
        for k, v in gold["entities"].items():
            if k in ents and ents[k] == v:
                correct += 1
    return intent_ok, keys_ok, correct, total


def generate(model, tokenizer, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def evaluate_split(model, tokenizer, examples: list[dict]) -> dict:
    n = len(examples)
    exact = valid = 0
    intent_ok = keys_ok = fields_correct = fields_total = 0
    per_example = []
    for ex in examples:
        user = ex["messages"][1]["content"]
        gold_raw = ex["messages"][2]["content"]
        gold = json.loads(gold_raw)
        raw_out = generate(model, tokenizer, user)
        pred = extract_json(raw_out)
        is_valid = pred is not None
        is_exact = is_valid and canonical(pred) == canonical(gold)
        i_ok, k_ok, f_c, f_t = field_scores(pred, gold)
        exact += int(is_exact); valid += int(is_valid)
        intent_ok += i_ok; keys_ok += k_ok
        fields_correct += f_c; fields_total += f_t
        per_example.append({
            "user": user, "expected": gold_raw, "raw_output": raw_out,
            "exact_match": bool(is_exact), "json_valid": bool(is_valid),
            "intent_ok": bool(i_ok), "fields_correct": f_c, "fields_total": f_t,
        })
    return {
        "n": n,
        "exact_match_accuracy": exact / n,
        "json_validity_rate": valid / n,
        "intent_accuracy": intent_ok / n,
        "entity_key_accuracy": keys_ok / n,
        "field_level_accuracy": fields_correct / fields_total,
        "per_example": per_example,
    }


def collect_failures(ft_results: dict, min_n: int = 3) -> list[dict]:
    """Genuine failure cases from the fine-tuned model run."""
    fails = [e for e in ft_results["per_example"] if not e["exact_match"]]
    # Sort hardest-first: invalid JSON, then most wrong fields.
    fails.sort(key=lambda e: (e["json_valid"], -e["fields_correct"]))
    out = []
    for e in fails[:max(min_n, 6)]:
        analysis = ("Output was not parseable JSON." if not e["json_valid"] else
                    "JSON parsed but at least one field differs from gold.")
        if not e["intent_ok"]:
            analysis += " Intent classification error."
        out.append({**e, "analysis": analysis})
    return out


def main() -> None:
    examples = [json.loads(l) for l in TEST_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Evaluating on {len(examples)} held-out test examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Base model -------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_res = evaluate_split(base, tokenizer, examples)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Fine-tuned model -------------------------------------------------
    ft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    ft = PeftModel.from_pretrained(ft_base, LORA_OUTPUT_DIR)
    ft_res = evaluate_split(ft, tokenizer, examples)

    summary = {
        "base_model": {k: v for k, v in base_res.items() if k != "per_example"},
        "fine_tuned_lora": {k: v for k, v in ft_res.items() if k != "per_example"},
        "failure_cases": collect_failures(ft_res),
        "settings_note": "Greedy decoding, deterministic canonical JSON comparison.",
    }
    EVAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(json.dumps(summary["base_model"], indent=2))
    print(json.dumps(summary["fine_tuned_lora"], indent=2))
    print(f"Wrote {EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
