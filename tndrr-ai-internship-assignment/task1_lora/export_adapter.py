"""Task 1 – Export and Merge LoRA Adapter.

This utility merges the trained low-rank adapter weights into the base
Qwen2.5-0.5B-Instruct model and exports the unified standalone model.
This merged model can be:
1. Directly deployed for zero-overhead inference.
2. Converted to GGUF via llama.cpp for local quantized CPU inference (Task 3).
3. Pushed to Hugging Face Hub.

Usage:
    python export_adapter.py --merge --output-dir ./merged_qwen_0.5b
    python export_adapter.py --package-zip ./task1_adapter.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LORA_OUTPUT_DIR, MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Export and merge LoRA adapter weights")
    parser.add_argument("--adapter-dir", type=Path, default=LORA_OUTPUT_DIR,
                        help="Path to trained LoRA adapter directory")
    parser.add_argument("--base-model", type=str, default=MODEL_ID,
                        help="Base model identifier")
    parser.add_argument("--merge", action="store_true",
                        help="Merge adapter into base model weights and save full standalone model")
    parser.add_argument("--output-dir", type=Path, default=Path("./merged_qwen_0.5b"),
                        help="Directory to save merged model")
    parser.add_argument("--package-zip", type=Path, default=None,
                        help="Package adapter folder into a standalone .zip file for submission")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.adapter_dir.exists():
        print(f"[ERROR] Adapter directory does not exist: {args.adapter_dir}", file=sys.stderr)
        print("Run train_lora.py in Colab or locally first.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("Task 1 LoRA Adapter Exporter")
    print(f"Base model : {args.base_model}")
    print(f"Adapter dir: {args.adapter_dir}")
    print("=" * 60)

    # 1. Package ZIP if requested
    if args.package_zip:
        zip_stem = args.package_zip.with_suffix("")
        archive_path = shutil.make_archive(str(zip_stem), "zip", root_dir=args.adapter_dir)
        print(f"Packaged standalone adapter archive to: {archive_path}")

    # 2. Merge and export standalone model
    if args.merge:
        print("\nLoading base model into CPU memory...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        print(f"Loading LoRA adapter from {args.adapter_dir}...")
        lora_model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))

        print("Merging adapter weights into base model (merge_and_unload)...")
        merged_model = lora_model.merge_and_unload()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged standalone model to {args.output_dir}...")
        merged_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("\nMerged model successfully exported.")
        print(f"You can now convert it to GGUF format for llama.cpp via:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {args.output_dir} --outfile qwen2.5-0.5b-lora-merged.gguf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
