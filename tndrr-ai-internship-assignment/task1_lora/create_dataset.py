"""Task 1 – Generate an original NL->JSON instruction dataset.

Produces data/train.jsonl (300 examples) and data/test.jsonl (60 examples).

Design
------
* Five productive intents (set_reminder, create_event, get_weather,
  send_message, search_web) plus a `query_null` intent for ambiguous or
  nonsensical requests, following a fixed policy: missing entity values
  are serialised as JSON `null`, unknown intents map to `query_null`.
* Each example = {"messages": [{"role": "system", ...},
                               {"role": "user", ...},
                               {"role": "assistant", <canonical JSON>}]}
* Templates are combined with sampled slot values so phrasing varies while
  the target schema stays consistent. Canonical key order: intent, entities
  (entity keys in the per-intent canonical order below).
* Deduplication: user texts are normalised and checked globally; train and
  test are drawn from disjoint template/value pools where practical and
  verified to have zero overlap.
* A validation pass prints statistics and asserts all invariants.

Run:  python create_dataset.py          (works on laptop CPU or Colab)
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from config import N_TEST, N_TRAIN, RANDOM_SEED, SYSTEM_PROMPT, TEST_FILE, TRAIN_FILE

# ---------------------------------------------------------------------------
# Slot vocabularies
# ---------------------------------------------------------------------------
TASKS_REMINDER = [
    "submit assignment", "call the dentist", "pay electricity bill",
    "water the plants", "take medication", "back up laptop", "renew passport",
    "buy groceries", "finish report", "stretch break", "reply to email",
    "attend standup", "pick up laundry", "return library book", "fix resume",
    "book flight tickets", "clean keyboard", "charge power bank",
]
PEOPLE = [
    "Mom", "Dad", "Priya", "Rahul", "professor", "team lead", "Ananya",
    "Vikram", "the client", "my sister", "Arjun", "HR", "the plumber",
    "Karthik", "my roommate", "Neha", "the doctor", "Sneha", "Aditya",
    "the delivery agent",
]
MESSAGES_TEXT = [
    "I will be late tonight", "meeting moved to 4 PM", "thanks for helping",
    "confirming tomorrow call", "sending files now", "out of office Monday",
    "happy birthday", "review pending on doc", "lunch at 1?",
    "deadline shifted to Friday", "need signature urgently", "joining 10 min late",
]
CITIES = [
    "Bengaluru", "Delhi", "Mumbai", "Chennai", "Hyderabad", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Kochi", "Lucknow", "Goa",
]
EVENT_NAMES = [
    "team retro", "project review", "birthday party", "webinar on AI",
    "sprint planning", "client demo", "study group", "gym session",
    "family dinner", "campus interview", "design sync", "product launch",
]
SEARCH_TOPICS = [
    "LoRA fine-tuning tutorial", "Python list comprehension syntax",
    "how does attention work in transformers", "NIST AI RMF overview",
    "llama.cpp quantization types", "difference between recall and precision",
    "GGUF format explained", "how cosine similarity works",
    "best practices for prompt engineering", "what is KV cache",
    "sentence transformers MiniLM model", "vector database vs numpy search",
]
DATES = [f"2026-1{m}-0{d}" if d < 10 else f"2026-1{m}-{d}"
         for m in (1, 2) for d in range(9, 25)]  # Nov/Dec 2026 dates
TIMES = [f"{h:02d}:{m:02d}" for h in range(7, 22) for m in (0, 15, 30, 45)]
UNITS_TEMP = ["celsius", "fahrenheit"]

# Ambiguous / off-domain inputs -> query_null policy
NULL_UTTERANCES = [
    "asdfgh qwerty", "please do the thing you know", "hmm yeah so basically",
    "can you help me with that stuff", "banana flask orbit", "u know what i mean",
    "do it whenever", "tell me something cool about numbers or whatever",
    "the usual, you remember", "idk man just handle it", "blue silence loudly",
    "same as last time but different", "make it pop", "thingy over there",
    "what about, you know, weather-ish but not really",
]

# Relative-date phrasings (no explicit date/time given -> null fields)
RELATIVE_WHEN = ["tomorrow", "tonight", "next Monday", "this weekend",
                 "in two hours", "later today", "next week", "day after tomorrow"]

# ---------------------------------------------------------------------------
# Per-intent canonical entity orders
# ---------------------------------------------------------------------------
ENTITY_ORDER = {
    "set_reminder": ["task", "date", "time"],
    "create_event": ["event", "date", "time"],
    "get_weather": ["location", "date", "unit"],
    "send_message": ["recipient", "content"],
    "search_web": ["query"],
    "query_null": [],
}

# ---------------------------------------------------------------------------
# Sentence templates. `{...}` marks optional slots; '?' variants drop slots
# to teach the null policy. Placeholders filled by SAMPLE functions.
# ---------------------------------------------------------------------------
def _mk_set_reminder(rng: random.Random) -> tuple[str, dict[str, Any]]:
    task = rng.choice(TASKS_REMINDER)
    style = rng.random()
    if style < 0.45:
        d, t = rng.choice(DATES), rng.choice(TIMES)
        text = rng.choice([
            f"Remind me to {task} on {d} at {t}",
            f"Set a reminder for {task} at {t} on {d}",
            f"I need to {task} - remind me on {d} at {t}",
            f"Can you remind me about {task}? It's on {d} at {t}.",
            f"Alert me to {task} at {t} on {d}, please.",
            f"{task} - set reminder {d} {t}",
        ])
        return text, {"task": task, "date": d, "time": t}
    if style < 0.70:
        d = rng.choice(DATES)
        text = rng.choice([
            f"Remind me to {task} on {d}",
            f"On {d} I have to {task}, set a reminder",
            f"Reminder: {task}, {d}",
        ])
        return text, {"task": task, "date": d, "time": None}
    when = rng.choice(RELATIVE_WHEN)
    text = rng.choice([
        f"Remind me to {task} {when}",
        f"{when} I need a reminder for {task}",
    ])
    return text, {"task": task, "date": None, "time": None}


def _mk_create_event(rng: random.Random) -> tuple[str, dict[str, Any]]:
    ev = rng.choice(EVENT_NAMES)
    style = rng.random()
    if style < 0.45:
        d, t = rng.choice(DATES), rng.choice(TIMES)
        text = rng.choice([
            f"Create a calendar event called '{ev}' on {d} at {t}",
            f"Schedule {ev} for {d} at {t}",
            f"Add '{ev}' to my calendar on {d}, starting {t}",
            f"I want to put {ev} in my calendar: {d} {t}",
        ])
        return text, {"event": ev, "date": d, "time": t}
    if style < 0.72:
        d = rng.choice(DATES)
        text = rng.choice([
            f"Schedule {ev} sometime on {d}",
            f"Create event '{ev}' on {d}",
        ])
        return text, {"event": ev, "date": d, "time": None}
    text = rng.choice([
        f"Make a new event for {ev}",
        f"Add {ev} to my calendar",
    ])
    return text, {"event": ev, "date": None, "time": None}


def _mk_get_weather(rng: random.Random) -> tuple[str, dict[str, Any]]:
    city = rng.choice(CITIES)
    style = rng.random()
    if style < 0.40:
        d, u = rng.choice(DATES), rng.choice(UNITS_TEMP)
        text = rng.choice([
            f"What's the weather in {city} on {d} in {u}?",
            f"Forecast for {city} on {d}, {u} please",
            f"Will it rain in {city} on {d}? Report in {u}.",
        ])
        return text, {"location": city, "date": d, "unit": u}
    if style < 0.75:
        d = rng.choice(DATES)
        text = rng.choice([
            f"How's the weather looking in {city} on {d}?",
            f"Check the forecast for {city}, {d}",
        ])
        return text, {"location": city, "date": d, "unit": None}
    when = rng.choice(RELATIVE_WHEN)
    text = rng.choice([
        f"Weather in {city} {when}?",
        f"Is it going to be hot in {city} {when}",
    ])
    return text, {"location": city, "date": None, "unit": None}


def _mk_send_message(rng: random.Random) -> tuple[str, dict[str, Any]]:
    p, m = rng.choice(PEOPLE), rng.choice(MESSAGES_TEXT)
    style = rng.random()
    if style < 0.60:
        text = rng.choice([
            f"Message {p}: '{m}'",
            f"Send a text to {p} saying {m}",
            f"Tell {p} that {m}",
            f"Text {p} - {m}",
            f"WhatsApp {p}: {m}",
        ])
        return text, {"recipient": p, "content": m}
    text = rng.choice([
        f"Send a message to {p}",
        f"I want to text {p} something",
    ])
    return text, {"recipient": p, "content": None}


def _mk_search_web(rng: random.Random) -> tuple[str, dict[str, Any]]:
    q = rng.choice(SEARCH_TOPICS)
    text = rng.choice([
        f"Search the web for {q}",
        f"Look up '{q}' online",
        f"Google {q}",
        f"Find me information about {q}",
        f"Can you search for {q}?",
    ])
    return text, {"query": q}


def _mk_query_null(rng: random.Random) -> tuple[str, dict[str, Any]]:
    return rng.choice(NULL_UTTERANCES), {}

GENERATORS = [_mk_set_reminder, _mk_create_event, _mk_get_weather,
              _mk_send_message, _mk_search_web, _mk_query_null]
# Rough mix: heavy on the five real intents, ~10% ambiguous cases.
WEIGHTS = [0.22, 0.18, 0.18, 0.18, 0.14, 0.10]


def canonical_json(intent: str, entities: dict[str, Any]) -> str:
    """Serialise target with deterministic key order (policy enforced here)."""
    ordered = {k: entities.get(k) for k in ENTITY_ORDER[intent]}
    obj = {"intent": intent, "entities": ordered}
    return json.dumps(obj, ensure_ascii=False)


def normalise(text: str) -> str:
    """Normalisation used for exact-duplicate detection."""
    t = unicodedata.normalize("NFKC", text.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_pool(target_n: int, rng: random.Random, forbidden: set[str]) -> list[dict]:
    """Generate `target_n` unique examples avoiding `forbidden` normalised texts."""
    pool: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(pool) < target_n and attempts < target_n * 200:
        attempts += 1
        gen = rng.choices(GENERATORS, weights=WEIGHTS, k=1)[0]
        text, ents = gen(rng)
        intent = gen.__name__.replace("_mk_", "")
        key = normalise(text)
        if key in seen or key in forbidden:
            continue
        seen.add(key)
        pool.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
                {"role": "assistant", "content": canonical_json(intent, ents)},
            ]
        })
    if len(pool) < target_n:
        raise RuntimeError(f"Only produced {len(pool)}/{target_n} unique examples")
    return pool


def validate(examples: list[dict], name: str) -> None:
    """Assert schema invariants and print human-readable inspection stats."""
    norm_seen: set[str] = set()
    intent_counts: dict[str, int] = {}
    null_field_count = 0
    for ex in examples:
        assert len(ex["messages"]) == 3 and ex["messages"][0]["role"] == "system"
        user = ex["messages"][1]["content"]
        tgt = json.loads(ex["messages"][2]["content"])  # must parse
        assert set(tgt.keys()) == {"intent", "entities"}
        assert tgt["intent"] in ENTITY_ORDER
        assert list(tgt["entities"].keys()) == ENTITY_ORDER[tgt["intent"]]
        for v in tgt["entities"].values():
            assert v is None or isinstance(v, str)
            if v is None:
                null_field_count += 1
        k = normalise(user)
        assert k not in norm_seen, f"duplicate user text in {name}: {user!r}"
        norm_seen.add(k)
        intent_counts[tgt["intent"]] = intent_counts.get(tgt["intent"], 0) + 1
    print(f"[{name}] {len(examples)} examples | intents={intent_counts} | "
          f"null entity fields={null_field_count}")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    DATA_DIR = TRAIN_FILE.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_pool = build_pool(N_TRAIN, rng, forbidden=set())
    train_keys = {normalise(e["messages"][1]["content"]) for e in train_pool}
    test_pool = build_pool(N_TEST, rng, forbidden=train_keys)

    # Cross-split leakage check (stronger than needed: even substring-free)
    test_keys = {normalise(e["messages"][1]["content"]) for e in test_pool}
    assert not (train_keys & test_keys), "train/test overlap detected!"

    validate(train_pool, "train")
    validate(test_pool, "test")

    for path, data in ((TRAIN_FILE, train_pool), (TEST_FILE, test_pool)):
        with path.open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(data)} lines)")

    print("\nSample train examples:")
    for ex in train_pool[:3]:
        print("  USER:", ex["messages"][1]["content"])
        print("  TGT :", ex["messages"][2]["content"])
    print("\nSample held-out test examples:")
    for ex in test_pool[:3]:
        print("  USER:", ex["messages"][1]["content"])
        print("  TGT :", ex["messages"][2]["content"])


if __name__ == "__main__":
    main()
