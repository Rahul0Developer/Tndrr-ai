"""Task 1 – Dataset validation script.

Validates:
1. File existence and minimum counts (>=300 train, >=60 test).
2. ChatML message structure (roles: system, user, assistant).
3. JSON syntax and strict schema conformity for all target outputs.
4. Intent validity across all 6 specified categories.
5. Entity key correctness and canonical ordering per intent.
6. Value type constraints (only strings or null allowed).
7. Complete train/test split isolation (zero leakage, zero duplicates).
8. Intent distribution and missing value (null) metrics.

Run:
    python validate_dataset.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from config import N_TEST, N_TRAIN, TEST_FILE, TRAIN_FILE

CANONICAL_SCHEMA = {
    "set_reminder": ["task", "date", "time"],
    "create_event": ["event", "date", "time"],
    "get_weather": ["location", "date", "unit"],
    "send_message": ["recipient", "content"],
    "search_web": ["query"],
    "query_null": [],
}


def normalise_text(text: str) -> str:
    """Normalize text for rigorous duplicate and leakage detection."""
    t = unicodedata.normalize("NFKC", text.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def validate_file(path: Path, expected_min: int, split_name: str) -> tuple[list[dict], set[str], set[tuple[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < expected_min:
        raise ValueError(
            f"[{split_name}] Expected at least {expected_min} lines, found {len(lines)}"
        )

    examples: list[dict] = []
    normalized_inputs: set[str] = set()
    input_target_pairs: set[tuple[str, str]] = set()
    intent_counter: Counter = Counter()
    null_entity_counter: Counter = Counter()

    for idx, line in enumerate(lines, 1):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as err:
            raise ValueError(f"[{split_name} line {idx}] Invalid JSONL line: {err}") from err

        if "messages" not in ex or not isinstance(ex["messages"], list):
            raise ValueError(f"[{split_name} line {idx}] Missing 'messages' list")

        msgs = ex["messages"]
        if len(msgs) != 3:
            raise ValueError(f"[{split_name} line {idx}] Expected 3 messages, got {len(msgs)}")

        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(
                f"[{split_name} line {idx}] Invalid roles sequence: {roles}. "
                f"Expected ['system', 'user', 'assistant']"
            )

        system_msg, user_msg, assistant_msg = msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]
        if not user_msg.strip():
            raise ValueError(f"[{split_name} line {idx}] Empty user input")

        # Parse assistant JSON target
        try:
            target = json.loads(assistant_msg)
        except json.JSONDecodeError as err:
            raise ValueError(f"[{split_name} line {idx}] Assistant content is not valid JSON: {err}") from err

        if not isinstance(target, dict):
            raise ValueError(f"[{split_name} line {idx}] Assistant target must be a JSON object")

        if set(target.keys()) != {"intent", "entities"}:
            raise ValueError(
                f"[{split_name} line {idx}] Root keys must strictly be {{'intent', 'entities'}}, "
                f"got {set(target.keys())}"
            )

        intent = target["intent"]
        if intent not in CANONICAL_SCHEMA:
            raise ValueError(
                f"[{split_name} line {idx}] Unknown intent '{intent}'. Expected one of {list(CANONICAL_SCHEMA.keys())}"
            )

        entities = target["entities"]
        if not isinstance(entities, dict):
            raise ValueError(f"[{split_name} line {idx}] 'entities' must be a JSON dictionary")

        expected_entity_keys = CANONICAL_SCHEMA[intent]
        actual_entity_keys = list(entities.keys())
        if actual_entity_keys != expected_entity_keys:
            raise ValueError(
                f"[{split_name} line {idx}] Entity keys for intent '{intent}' mismatch. "
                f"Expected {expected_entity_keys}, got {actual_entity_keys}"
            )

        for k, v in entities.items():
            if v is not None and not isinstance(v, str):
                raise ValueError(
                    f"[{split_name} line {idx}] Entity '{k}' has invalid value {v!r} (type {type(v).__name__}). "
                    f"Must be a string or null."
                )
            if v is None:
                null_entity_counter[f"{intent}.{k}"] += 1

        norm_user = normalise_text(user_msg)
        if norm_user in normalized_inputs:
            raise ValueError(f"[{split_name} line {idx}] Exact duplicate user input detected: {user_msg!r}")
        normalized_inputs.add(norm_user)
        input_target_pairs.add((norm_user, assistant_msg))

        intent_counter[intent] += 1
        examples.append(ex)

    print(f"\n[{split_name.upper()}] Dataset Validation Passed:")
    print(f"  Total examples: {len(examples)}")
    print(f"  Intents represented ({len(intent_counter)}/6):")
    for int_name, count in sorted(intent_counter.items()):
        print(f"    - {int_name:15s}: {count:3d} ({count / len(examples):.1%})")
    print(f"  Null entity slots: {sum(null_entity_counter.values())} total")

    if len(intent_counter) < 6:
        raise ValueError(
            f"[{split_name}] Expected at least 6 intent categories, only found {len(intent_counter)}: "
            f"{list(intent_counter.keys())}"
        )

    return examples, normalized_inputs, input_target_pairs


def main() -> int:
    print("=" * 60)
    print("Task 1 Dataset Validation Suite")
    print("=" * 60)

    try:
        train_examples, train_inputs, train_pairs = validate_file(TRAIN_FILE, N_TRAIN, "train")
        test_examples, test_inputs, test_pairs = validate_file(TEST_FILE, N_TEST, "test")

        # Cross-split leakage checks
        overlap_inputs = train_inputs & test_inputs
        if overlap_inputs:
            raise ValueError(
                f"Train/Test leakage detected: {len(overlap_inputs)} identical user inputs found in both splits!\n"
                f"Sample: {list(overlap_inputs)[:3]}"
            )

        overlap_pairs = train_pairs & test_pairs
        if overlap_pairs:
            raise ValueError(
                f"Train/Test leakage detected: {len(overlap_pairs)} identical input-target pairs in both splits!"
            )

        print("\n[SPLIT ISOLATION]")
        print(f"  Train inputs count: {len(train_inputs)}")
        print(f"  Test inputs count : {len(test_inputs)}")
        print("  Input overlap     : ZERO (Clean split isolation)")
        print("  Pair overlap      : ZERO (Clean split isolation)")
        print("\nAll dataset checks passed successfully (100% compliant with schema).")
        return 0

    except Exception as exc:
        print(f"\n[VALIDATION FAILED] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
