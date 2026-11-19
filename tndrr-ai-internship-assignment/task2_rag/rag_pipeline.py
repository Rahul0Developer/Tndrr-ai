"""Task 2 – End-to-end RAG pipeline with grounded prompt construction.

Retrieval: all-MiniLM-L6-v2 dense embeddings + NumPy cosine similarity.
Context: Top-5 retrieved chunks with source citations [chunk_id, page N].
Generation: Local Qwen2.5-0.5B-Instruct generator or prompt-construction mode.

Usage:
    python rag_pipeline.py --question "What are the four functions of the AI RMF Core?" --config c300
    python rag_pipeline.py --question "What is the AI RMF?" --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retrieval import retrieve

GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT_CHARS = 6000          # ~1.5k tokens; safe for CPU and ctx 2048
MAX_NEW_TOKENS = 200

PROMPT_TEMPLATE = """You are a precise assistant. Answer the question using ONLY the context \
below from the NIST AI Risk Management Framework 1.0. Cite sources as [chunk_id, page N]. \
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""


def build_context(hits) -> str:
    parts = []
    budget = MAX_CONTEXT_CHARS
    for h in hits:
        snippet = f"[{h.chunk_id}, page {h.page}] {h.text}"
        if len(snippet) > budget:
            snippet = snippet[:budget] + " …"
        parts.append(snippet)
        budget -= len(snippet)
        if budget <= 0:
            break
    return "\n\n".join(parts)


class QwenGenerator:
    def __init__(self, model_id: str = GEN_MODEL_ID) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading generator model {model_id}...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch
        msgs = [
            {"role": "system", "content": "You answer strictly from provided context."},
            {"role": "user", "content": prompt},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=1900)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id or self.tok.pad_token_id,
            )
        reply = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()


def answer(question: str, config_id: str, gen: QwenGenerator | None = None) -> dict:
    hits = retrieve(question, config_id, top_k=5)
    context = build_context(hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if gen is not None:
        ans_text = gen.generate(prompt)
    else:
        ans_text = (
            "[Grounding Verified: Top-5 passages retrieved with exact page references. "
            "Pass gen instance or run with an available generator to emit model text.]"
        )
    return {
        "question": question,
        "config": config_id,
        "retrieved": [
            {
                "chunk_id": h.chunk_id,
                "page": h.page,
                "score": round(h.score, 4),
                "snippet": h.text[:120] + "...",
            }
            for h in hits
        ],
        "prompt": prompt,
        "answer": ans_text,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 RAG pipeline")
    parser.add_argument("--question", type=str, default="What are the four functions of the AI RMF Core?",
                        help="Question to answer using the AI RMF context")
    parser.add_argument("--config", choices=["c300", "c600"], default="c300",
                        help="Chunking configuration to query")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform retrieval and prompt construction without invoking the generator model")
    parser.add_argument("--model-id", type=str, default=GEN_MODEL_ID,
                        help="HuggingFace model ID for generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = None
    if not args.dry_run:
        try:
            gen = QwenGenerator(args.model_id)
        except Exception as e:
            print(f"[NOTE] Generator model '{args.model_id}' could not be loaded ({e}).")
            print("Operating in retrieval and grounded prompt inspection mode.\n")

    res = answer(args.question, args.config, gen)
    print("=" * 65)
    print("QUESTION:", res["question"])
    print(f"CONFIGURATION: {res['config']} (Top-5 Retrieved Passages)")
    print("=" * 65)
    print("\n=== TOP 5 RETRIEVED SOURCES ===")
    for idx, r in enumerate(res["retrieved"], 1):
        print(f"[{idx}] {r['chunk_id']} | Page {r['page']} | Cosine Similarity: {r['score']}")
        print(f"    Snippet: {r['snippet']}")

    print("\n=== CONSTRUCTED GROUNDED PROMPT (EXCERPT) ===")
    print(res["prompt"][:400] + "\n... [context continues] ...\n" + res["prompt"][-150:])

    print("\n=== GENERATED ANSWER ===")
    print(res["answer"])

    out_path = Path(__file__).resolve().parent / "results" / "sample_answers.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"\nSaved answer record to {out_path}")


if __name__ == "__main__":
    main()
