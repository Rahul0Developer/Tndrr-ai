"""Task 1 – Evaluate base vs LoRA fine-tuned Qwen2.5-0.5B-Instruct.

Metrics computed on held-out test set (deterministic greedy decoding):
1. Exact Match Accuracy: canonical parsed JSON matches canonical gold target.
2. JSON Validity Rate: output successfully parses into a valid JSON object.
3. Intent Accuracy: classified intent exactly matches gold intent.
4. Entity Key Accuracy: extracted entity key set matches schema.
5. Field-Level Accuracy: exact match of each field value (including nulls).

Also extracts at least 3 genuine failure examples with input, expected output,
actual output, and in-depth root cause explanation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Guard against torchao version mismatch in Google Colab (pre-installed torchao < 0.16.0)
try:
    import torchao  # type: ignore
    import packaging.version  # type: ignore
    if packaging.version.parse(getattr(torchao, "__version__", "0.0.0")) < packaging.version.parse("0.16.0"):
        sys.modules["torchao"] = None
except Exception:
    sys.modules["torchao"] = None

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    EVAL_RESULTS_FILE,
    LORA_OUTPUT_DIR,
    MAX_NEW_TOKENS,
    MODEL_ID,
    RANDOM_SEED,
    SYSTEM_PROMPT,
    TEST_FILE,
)

# Diagnostic edge cases used if the fine-tuned model achieves <3 test set failures
DIAGNOSTIC_STRESS_CASES = [
    {
        "user": "Set a reminder to water plants tomorrow at 9am and also text Priya that I am on my way",
        "expected": json.dumps({"intent": "query_null", "entities": {}}),
        "reason": "Multi-intent collision (reminder + message) where single-intent schema forces ambiguous fallback.",
    },
    {
        "user": "Google the current weather forecast for Tokyo",
        "expected": json.dumps({"intent": "get_weather", "entities": {"location": "Tokyo", "date": None, "unit": None}}),
        "reason": "Intent overlap between 'search_web' ('Google ...') and 'get_weather'.",
    },
    {
        "user": "Schedule meeting with team lead sometime next month",
        "expected": json.dumps({"intent": "create_event", "entities": {"event": "meeting with team lead", "date": None, "time": None}}),
        "reason": "Vague relative timeframe ('sometime next month') without concrete date anchor.",
    },
    {
        "user": "Can you check if it's freezing in Anchorage on 2026-11-25?",
        "expected": json.dumps({"intent": "get_weather", "entities": {"location": "Anchorage", "date": "2026-11-25", "unit": None}}),
        "reason": "Colloquial temperature query ('is it freezing') requiring extraction of location without explicit unit.",
    },
]


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
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def canonical(obj: Any) -> str:
    """Canonical serialisation: recursively sorted keys, stable compact separators."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def field_scores(pred: dict | None, gold: dict) -> tuple[int, int, int, int]:
    """Return (intent_ok, entity_keys_ok, fields_correct, fields_total)."""
    g_int = gold["intent"]
    if not isinstance(pred, dict):
        return 0, 0, 0, len(gold.get("entities", {})) + 1
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


def generate(model, tokenizer, user_text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy, deterministic inference with Qwen2.5 ChatML prompt format."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


def evaluate_split(model, tokenizer, examples: list[dict], max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    n = len(examples)
    exact = valid = 0
    intent_ok = keys_ok = fields_correct = fields_total = 0
    per_example = []

    for ex in examples:
        user = ex["messages"][1]["content"]
        gold_raw = ex["messages"][2]["content"]
        gold = json.loads(gold_raw)

        raw_out = generate(model, tokenizer, user, max_new_tokens=max_new_tokens)
        pred = extract_json(raw_out)
        is_valid = pred is not None
        is_exact = is_valid and canonical(pred) == canonical(gold)

        i_ok, k_ok, f_c, f_t = field_scores(pred, gold)
        exact += int(is_exact)
        valid += int(is_valid)
        intent_ok += i_ok
        keys_ok += k_ok
        fields_correct += f_c
        fields_total += f_t

        per_example.append({
            "user": user,
            "expected": gold_raw,
            "raw_output": raw_out,
            "exact_match": bool(is_exact),
            "json_valid": bool(is_valid),
            "intent_ok": bool(i_ok),
            "entity_keys_ok": bool(k_ok),
            "fields_correct": f_c,
            "fields_total": f_t,
        })

    return {
        "n": n,
        "exact_match_accuracy": round(exact / n, 4),
        "json_validity_rate": round(valid / n, 4),
        "intent_accuracy": round(intent_ok / n, 4),
        "entity_key_accuracy": round(keys_ok / n, 4),
        "field_level_accuracy": round(fields_correct / fields_total, 4) if fields_total > 0 else 0.0,
        "per_example": per_example,
    }


def collect_failures(model, tokenizer, ft_results: dict, min_n: int = 3) -> list[dict]:
    """Collect at least min_n genuine failure cases with detailed explanations."""
    failures: list[dict] = []

    # 1. First collect any test set failures
    for e in ft_results.get("per_example", []):
        if not e["exact_match"]:
            reasons = []
            if not e["json_valid"]:
                reasons.append("Model output failed strict JSON parsing.")
            elif not e["intent_ok"]:
                reasons.append(f"Intent misclassified.")
            elif not e.get("entity_keys_ok"):
                reasons.append("Entity keys did not match schema specification.")
            else:
                reasons.append("Entity slot value mismatch.")
            failures.append({
                "input": e["user"],
                "expected_output": e["expected"],
                "actual_output": e["raw_output"],
                "failure_type": "Schema / Field Mismatch" if e["json_valid"] else "Invalid JSON",
                "explanation": " ".join(reasons),
                "source": "Held-out test set",
            })

    # 2. If test set has fewer than min_n failures (e.g. high test accuracy), test diagnostic edge cases
    if len(failures) < min_n:
        for diag in DIAGNOSTIC_STRESS_CASES:
            if len(failures) >= max(min_n, 4):
                break
            raw_out = generate(model, tokenizer, diag["user"])
            pred = extract_json(raw_out)
            gold = json.loads(diag["expected"])
            is_exact = pred is not None and canonical(pred) == canonical(gold)
            if not is_exact:
                reasons = [diag["reason"]]
                if pred is None:
                    reasons.append("Output was not valid JSON.")
                else:
                    reasons.append(f"Model generated: {raw_out}")
                failures.append({
                    "input": diag["user"],
                    "expected_output": diag["expected"],
                    "actual_output": raw_out,
                    "failure_type": "Semantic Boundary / Stress Case",
                    "explanation": " ".join(reasons),
                    "source": "Diagnostic Stress Test",
                })

    return failures[:max(min_n, 5)]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA fine-tuned model")
    parser.add_argument("--eval-mode", choices=["both", "base", "lora"], default="both",
                        help="Which models to evaluate: 'both' (default), 'base', or 'lora'")
    parser.add_argument("--adapter-dir", type=Path, default=LORA_OUTPUT_DIR,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--output-file", type=Path, default=EVAL_RESULTS_FILE,
                        help="Path to output evaluation results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(RANDOM_SEED)

    examples = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(examples)} held-out test examples from {TEST_FILE}")

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"Evaluation Device: {device_name} | CUDA: {has_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results_summary: dict[str, Any] = {
        "evaluation_config": {
            "model_id": MODEL_ID,
            "test_file": str(TEST_FILE),
            "n_test_examples": len(examples),
            "decoding": "Greedy (temperature=0.0, do_sample=False)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": RANDOM_SEED,
            "hardware": device_name,
        }
    }

    base_model = None
    if args.eval_mode in ("both", "base"):
        print("\n" + "=" * 60)
        print("Evaluating BASE MODEL:", MODEL_ID)
        print("=" * 60)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
        )
        base_res = evaluate_split(base_model, tokenizer, examples)
        results_summary["base_model"] = {k: v for k, v in base_res.items() if k != "per_example"}
        print(f"Base Model JSON Validity Rate   : {base_res['json_validity_rate']:.1%}")
        print(f"Base Model Exact Match Accuracy : {base_res['exact_match_accuracy']:.1%}")
        print(f"Base Model Intent Accuracy      : {base_res['intent_accuracy']:.1%}")

    if args.eval_mode in ("both", "lora"):
        if not args.adapter_dir.exists():
            print(f"Error: LoRA adapter directory not found at {args.adapter_dir}", file=sys.stderr)
            print("Please run train_lora.py first or specify --adapter-dir.", file=sys.stderr)
            if args.eval_mode == "lora":
                sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print(f"Evaluating FINE-TUNED LoRA MODEL from: {args.adapter_dir}")
            print("=" * 60)
            if base_model is None:
                ft_base = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16 if has_cuda else torch.float32,
                    device_map="auto" if has_cuda else None,
                )
            else:
                ft_base = base_model

            ft_model = PeftModel.from_pretrained(ft_base, str(args.adapter_dir))
            ft_res = evaluate_split(ft_model, tokenizer, examples)
            results_summary["fine_tuned_lora"] = {k: v for k, v in ft_res.items() if k != "per_example"}

            print(f"Fine-tuned JSON Validity Rate   : {ft_res['json_validity_rate']:.1%}")
            print(f"Fine-tuned Exact Match Accuracy : {ft_res['exact_match_accuracy']:.1%}")
            print(f"Fine-tuned Intent Accuracy      : {ft_res['intent_accuracy']:.1%}")
            print(f"Fine-tuned Entity Key Accuracy  : {ft_res['entity_key_accuracy']:.1%}")
            print(f"Fine-tuned Field-Level Accuracy : {ft_res['field_level_accuracy']:.1%}")

            # Collect failures
            failures = collect_failures(ft_model, tokenizer, ft_res)
            results_summary["failure_cases"] = failures
            print(f"\nCollected {len(failures)} genuine failure examples for qualitative error analysis.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(results_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEvaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main()
