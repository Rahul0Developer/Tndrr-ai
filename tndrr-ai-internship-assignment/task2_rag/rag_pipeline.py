"""Task 2 – End-to-end RAG pipeline with a small local Qwen generator.

Retrieval: MiniLM + NumPy cosine (top-5 chunks from the chosen config).
Generation: Qwen2.5-0.5B-Instruct via transformers on CPU (int8 optional).
Answer format includes source citations [chunk_id, page N].

Usage:
    python rag_pipeline.py "What are the four functions of the AI RMF Core?" c300
    python rag_pipeline.py            # interactive loop

If you prefer llama.cpp instead of transformers for generation, see
task3_local_inference/ — the prompt template below is identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import retrieve

GEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_CONTEXT_CHARS = 6000          # ~1.5k tokens; safe for 8 GB RAM + ctx 2048
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
    def __init__(self) -> None:
        print(f"Loading {GEN_MODEL_ID} (CPU, bf16)…")
        self.tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map=None)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        msgs = [{"role": "system", "content": "You answer strictly from provided context."},
                {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=1900)
        out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        reply = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.strip()


def answer(question: str, config_id: str, gen: QwenGenerator) -> dict:
    hits = retrieve(question, config_id, top_k=5)
    context = build_context(hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return {
        "question": question,
        "config": config_id,
        "retrieved": [{"chunk_id": h.chunk_id, "page": h.page,
                       "score": round(h.score, 4)} for h in hits],
        "answer": gen.generate(prompt),
    }


def main() -> None:
    if len(sys.argv) >= 2:
        q = sys.argv[1]
        cfg = sys.argv[2] if len(sys.argv) > 2 else "c300"
    else:
        q = input("Question: ").strip()
        cfg = input("Config (c300/c600): ").strip() or "c300"
    gen = QwenGenerator()
    res = answer(q, cfg, gen)
    print("\n=== ANSWER ===\n" + res["answer"])
    print("\n=== SOURCES ===")
    for r in res["retrieved"]:
        print(f"  {r['chunk_id']} p{r['page']} sim={r['score']}")
    out_path = Path(__file__).resolve().parent / "results" / "sample_answers.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
