# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
#     "torch>=2.0",
#     "transformers>=4.45",
# ]
# ///
"""
GGUF Conversion Script

Converts trained models to GGUF format for use with llama.cpp, Ollama, LM Studio.

Usage:
    python convert_gguf.py --model ./outputs/lora_adapter --output ./gguf
    python convert_gguf.py --model ./outputs/lora_adapter --quants q4_k_m,q5_k_m,q8_0
"""

import argparse
import os
from pathlib import Path
from unsloth import FastLanguageModel

# ============================================
# QUANTIZATION OPTIONS
# ============================================

QUANT_METHODS = {
    "q2_k": {
        "name": "Q2_K",
        "bits": 2,
        "size_factor": 0.25,
        "quality": "Lowest quality, smallest size",
        "use_case": "Extreme size constraints",
    },
    "q3_k_m": {
        "name": "Q3_K_M",
        "bits": 3,
        "size_factor": 0.35,
        "quality": "Low quality",
        "use_case": "Very limited resources",
    },
    "q4_0": {
        "name": "Q4_0",
        "bits": 4,
        "size_factor": 0.45,
        "quality": "Good quality, legacy format",
        "use_case": "Older llama.cpp versions",
    },
    "q4_k_m": {
        "name": "Q4_K_M",
        "bits": 4,
        "size_factor": 0.48,
        "quality": "Good quality, recommended",
        "use_case": "Best balance of size and quality",
    },
    "q4_k_s": {
        "name": "Q4_K_S",
        "bits": 4,
        "size_factor": 0.46,
        "quality": "Good quality, slightly smaller",
        "use_case": "Slightly smaller than Q4_K_M",
    },
    "q5_0": {
        "name": "Q5_0",
        "bits": 5,
        "size_factor": 0.55,
        "quality": "Better quality, legacy",
        "use_case": "Older llama.cpp versions",
    },
    "q5_k_m": {
        "name": "Q5_K_M",
        "bits": 5,
        "size_factor": 0.58,
        "quality": "Better quality",
        "use_case": "When Q4 isn't quite enough",
    },
    "q5_k_s": {
        "name": "Q5_K_S",
        "bits": 5,
        "size_factor": 0.56,
        "quality": "Better quality, smaller",
        "use_case": "Balanced 5-bit option",
    },
    "q6_k": {
        "name": "Q6_K",
        "bits": 6,
        "size_factor": 0.68,
        "quality": "High quality",
        "use_case": "Quality-focused with space",
    },
    "q8_0": {
        "name": "Q8_0",
        "bits": 8,
        "size_factor": 0.85,
        "quality": "Near-original quality",
        "use_case": "Maximum quality, larger size",
    },
    "f16": {
        "name": "F16",
        "bits": 16,
        "size_factor": 1.0,
        "quality": "Full precision",
        "use_case": "No quantization, maximum quality",
    },
}

DEFAULT_QUANTS = ["q4_k_m", "q5_k_m", "q8_0"]

# ============================================
# CONVERSION
# ============================================

def estimate_size(base_params: int, quant: str) -> float:
    """Estimate output file size in GB."""
    method = QUANT_METHODS[quant]
    # Base model size at fp16 (2 bytes per param)
    fp16_size = base_params * 2 / 1e9
    return fp16_size * method["size_factor"]


def convert_to_gguf(
    model_path: str,
    output_dir: str,
    quant_methods: list[str],
    max_seq_length: int = 2048,
):
    """Convert model to GGUF format."""
    print(f"Loading model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Get base param count for size estimates
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    os.makedirs(output_dir, exist_ok=True)

    for quant in quant_methods:
        if quant not in QUANT_METHODS:
            print(f"WARNING: Unknown quantization '{quant}', skipping")
            continue

        method = QUANT_METHODS[quant]
        est_size = estimate_size(total_params, quant)

        print(f"\n{'='*50}")
        print(f"Converting to {method['name']}")
        print(f"Estimated size: {est_size:.1f} GB")
        print(f"Quality: {method['quality']}")
        print(f"Use case: {method['use_case']}")
        print(f"{'='*50}")

        output_path = os.path.join(output_dir, quant)

        try:
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quant,
            )

            # Find the output file
            gguf_files = list(Path(output_path).glob("*.gguf"))
            if gguf_files:
                actual_size = gguf_files[0].stat().st_size / 1e9
                print(f"Saved: {gguf_files[0]}")
                print(f"Actual size: {actual_size:.2f} GB")
            else:
                print(f"Saved to: {output_path}")

        except Exception as e:
            print(f"ERROR: Failed to convert to {quant}: {e}")

    print(f"\n{'='*50}")
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*50}")


def list_quant_methods():
    """Print all available quantization methods."""
    print("\nAvailable quantization methods:\n")
    print(f"{'Method':<10} {'Bits':<6} {'Quality':<30} {'Use Case'}")
    print("-" * 80)

    for method_id, info in sorted(QUANT_METHODS.items(),
                                   key=lambda x: x[1]['bits']):
        print(f"{info['name']:<10} {info['bits']:<6} {info['quality']:<30} {info['use_case']}")


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model", required=True,
                       help="Path to trained model (LoRA adapter or merged)")
    parser.add_argument("--output", default="./gguf_output",
                       help="Output directory for GGUF files")
    parser.add_argument("--quants", default=",".join(DEFAULT_QUANTS),
                       help="Comma-separated quantization methods")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--list-methods", action="store_true",
                       help="List all available quantization methods")
    args = parser.parse_args()

    if args.list_methods:
        list_quant_methods()
        return

    quant_methods = [q.strip().lower() for q in args.quants.split(",")]

    print("=" * 60)
    print("GGUF Conversion")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Quantizations: {', '.join(quant_methods)}")

    convert_to_gguf(
        model_path=args.model,
        output_dir=args.output,
        quant_methods=quant_methods,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()
