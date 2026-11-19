"""Task 2 – Evaluate dense retrieval with Recall@5 for both chunk sizes.

Metric definition (as required):
    Recall@5 = (# questions where at least one relevant chunk appears in the
                top-5 retrieved) / (total questions)          [hit-rate form]

A chunk counts as "relevant" for a question if ALL of the question's
`relevant_keywords` occur in the chunk text (case-insensitive). This is a
deterministic, reproducible relevance judgement derived from gold passages.

Run:  python evaluate_retrieval.py     -> results/recall_at_5.json
"""
from __future__ import annotations

import json
from pathlib import Path

from retrieval import retrieve

BASE = Path(__file__).resolve().parent
QUESTIONS = BASE / "questions.json"
RESULTS = BASE / "results" / "recall_at_5.json"
TOP_K = 5


def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    low = chunk_text.lower()
    return all(k.lower() in low for k in keywords)


def main() -> None:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = data["questions"]
    out = {"metric": "Recall@5 (hit-rate)", "top_k": TOP_K, "configs": {}}

    for cfg in ("c300", "c600"):
        hits = 0
        details = []
        for q in questions:
            top = retrieve(q["question"], cfg, top_k=TOP_K)
            rel_flags = [is_relevant(h.text, q["relevant_keywords"]) for h in top]
            hit = any(rel_flags)
            hits += int(hit)
            details.append({
                "id": q["id"], "question": q["question"], "recalled": hit,
                "expected_page": q["expected_page"],
                "top5": [{"chunk_id": h.chunk_id, "page": h.page,
                          "score": round(h.score, 4), "relevant": r}
                         for h, r in zip(top, rel_flags)],
            })
        recall = hits / len(questions)
        out["configs"][cfg] = {"recall_at_5": recall, "hits": hits,
                               "n_questions": len(questions), "details": details}
        print(f"{cfg}: Recall@5 = {hits}/{len(questions)} = {recall:.2f}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {RESULTS}")

    # Surface failures for the report (genuine misses, ready to analyse).
    for cfg, res in out["configs"].items():
        for d in res["details"]:
            if not d["recalled"]:
                print(f"MISS [{cfg}] {d['id']}: {d['question']}")


if __name__ == "__main__":
    main()
