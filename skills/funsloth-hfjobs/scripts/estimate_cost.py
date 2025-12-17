# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Training Cost Estimator

Estimates training time and cost across different platforms and GPUs.

Usage:
    python estimate_cost.py --tokens 15000000 --model llama-3.1-8b --platform hfjobs
    python estimate_cost.py --tokens 15000000 --epochs 3 --compare-all
"""

import argparse
from dataclasses import dataclass

# ============================================
# GPU SPECIFICATIONS
# ============================================

@dataclass
class GPUSpec:
    name: str
    vram_gb: int
    tflops_fp16: float  # Theoretical peak
    tokens_per_sec: int  # Realistic SFT training throughput

GPUS = {
    # Consumer
    "rtx-3090": GPUSpec("RTX 3090", 24, 35.6, 3000),
    "rtx-4090": GPUSpec("RTX 4090", 24, 82.6, 5000),

    # Data center
    "a10g": GPUSpec("A10G", 24, 31.2, 4000),
    "a100-40": GPUSpec("A100 40GB", 40, 77.9, 8000),
    "a100-80": GPUSpec("A100 80GB", 80, 77.9, 10000),
    "h100": GPUSpec("H100 80GB", 80, 267.6, 18000),
}

# ============================================
# PLATFORM PRICING (approximate, Dec 2024)
# ============================================

@dataclass
class PlatformPricing:
    name: str
    gpu_prices: dict[str, float]  # GPU -> $/hour
    min_billing: str  # Billing granularity

PLATFORMS = {
    "hfjobs": PlatformPricing(
        name="Hugging Face Jobs",
        gpu_prices={
            "a10g": 1.50,
            "a100-40": 4.00,
            "a100-80": 6.00,
            "h100": 8.00,
        },
        min_billing="per minute",
    ),
    "runpod": PlatformPricing(
        name="RunPod",
        gpu_prices={
            "rtx-3090": 0.35,
            "rtx-4090": 0.55,
            "a100-40": 1.50,
            "a100-80": 2.00,
            "h100": 3.50,
        },
        min_billing="per second",
    ),
    "lambda": PlatformPricing(
        name="Lambda Labs",
        gpu_prices={
            "a100-40": 1.10,
            "a100-80": 1.29,
            "h100": 2.49,
        },
        min_billing="per second",
    ),
    "vast": PlatformPricing(
        name="Vast.ai",
        gpu_prices={
            "rtx-3090": 0.25,
            "rtx-4090": 0.45,
            "a100-40": 1.20,
            "a100-80": 1.80,
        },
        min_billing="per second",
    ),
}

# ============================================
# MODEL VRAM REQUIREMENTS
# ============================================

def get_vram_requirement(model: str, batch_size: int = 2, lora_rank: int = 16) -> int:
    """Estimate VRAM needed for training."""
    # Base VRAM for 4-bit models
    base_vram = {
        "llama-3.1-8b": 6,
        "llama-3.1-70b": 40,
        "qwen-2.5-7b": 5,
        "qwen-2.5-14b": 10,
        "qwen-2.5-32b": 20,
        "gemma-2-9b": 7,
        "phi-4-14b": 10,
        "mistral-7b": 5,
    }.get(model, 8)

    # Add overhead for activations and optimizer states
    training_overhead = 1.5 + (batch_size * 0.5) + (lora_rank / 16)

    return int(base_vram + training_overhead)


def get_compatible_gpus(model: str, batch_size: int = 2) -> list[str]:
    """Get GPUs that can handle this model."""
    required_vram = get_vram_requirement(model, batch_size)
    return [
        gpu_id for gpu_id, spec in GPUS.items()
        if spec.vram_gb >= required_vram
    ]

# ============================================
# COST CALCULATION
# ============================================

def estimate_training_time(
    total_tokens: int,
    gpu: str,
    epochs: int = 1,
) -> float:
    """Estimate training time in hours."""
    spec = GPUS[gpu]
    total_to_process = total_tokens * epochs
    seconds = total_to_process / spec.tokens_per_sec
    return seconds / 3600


def estimate_cost(
    total_tokens: int,
    gpu: str,
    platform: str,
    epochs: int = 1,
) -> dict:
    """Calculate estimated cost."""
    hours = estimate_training_time(total_tokens, gpu, epochs)

    platform_info = PLATFORMS[platform]
    gpu_price = platform_info.gpu_prices.get(gpu)

    if gpu_price is None:
        return None

    cost = hours * gpu_price

    return {
        "platform": platform_info.name,
        "gpu": GPUS[gpu].name,
        "hours": hours,
        "cost": cost,
        "price_per_hour": gpu_price,
    }

# ============================================
# DISPLAY
# ============================================

def format_time(hours: float) -> str:
    """Format hours as human-readable time."""
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days"


def print_estimate(estimate: dict):
    """Print a single estimate."""
    print(f"  {estimate['platform']:<20} {estimate['gpu']:<15} "
          f"{format_time(estimate['hours']):<15} ${estimate['cost']:.2f}")


def print_comparison(estimates: list[dict]):
    """Print comparison table."""
    print(f"\n{'Platform':<20} {'GPU':<15} {'Time':<15} {'Cost':<10}")
    print("-" * 60)

    for est in sorted(estimates, key=lambda x: x['cost']):
        print_estimate(est)


def recommend_cheapest(estimates: list[dict]) -> dict:
    """Find cheapest option."""
    valid = [e for e in estimates if e is not None]
    if not valid:
        return None
    return min(valid, key=lambda x: x['cost'])


def recommend_fastest(estimates: list[dict]) -> dict:
    """Find fastest option."""
    valid = [e for e in estimates if e is not None]
    if not valid:
        return None
    return min(valid, key=lambda x: x['hours'])

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Estimate training costs")
    parser.add_argument("--tokens", type=int, required=True,
                       help="Total tokens in dataset")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--model", default="llama-3.1-8b",
                       help="Target model")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()),
                       help="Specific platform")
    parser.add_argument("--gpu", choices=list(GPUS.keys()),
                       help="Specific GPU")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all platforms and GPUs")
    args = parser.parse_args()

    print("=" * 60)
    print("Training Cost Estimation")
    print("=" * 60)

    print(f"\nDataset: {args.tokens:,} tokens")
    print(f"Epochs: {args.epochs}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")

    # Get compatible GPUs
    compatible = get_compatible_gpus(args.model, args.batch_size)
    vram_needed = get_vram_requirement(args.model, args.batch_size)
    print(f"VRAM required: ~{vram_needed}GB")
    print(f"Compatible GPUs: {', '.join(GPUS[g].name for g in compatible)}")

    if args.compare_all:
        # Compare all platforms and compatible GPUs
        print("\nComparing all options...")
        estimates = []

        for platform_id in PLATFORMS:
            for gpu_id in compatible:
                est = estimate_cost(
                    args.tokens, gpu_id, platform_id, args.epochs
                )
                if est:
                    estimates.append(est)

        print_comparison(estimates)

        # Recommendations
        print("\n" + "=" * 60)
        print("Recommendations")
        print("=" * 60)

        cheapest = recommend_cheapest(estimates)
        fastest = recommend_fastest(estimates)

        if cheapest:
            print(f"\nCheapest: {cheapest['platform']} + {cheapest['gpu']}")
            print(f"  Time: {format_time(cheapest['hours'])}")
            print(f"  Cost: ${cheapest['cost']:.2f}")

        if fastest and fastest != cheapest:
            print(f"\nFastest: {fastest['platform']} + {fastest['gpu']}")
            print(f"  Time: {format_time(fastest['hours'])}")
            print(f"  Cost: ${fastest['cost']:.2f}")

    elif args.platform and args.gpu:
        # Specific platform and GPU
        if args.gpu not in compatible:
            print(f"\nWARNING: {GPUS[args.gpu].name} may not have enough VRAM")

        est = estimate_cost(args.tokens, args.gpu, args.platform, args.epochs)
        if est:
            print(f"\n{'Metric':<20} {'Value':>15}")
            print("-" * 37)
            print(f"{'Platform':<20} {est['platform']:>15}")
            print(f"{'GPU':<20} {est['gpu']:>15}")
            print(f"{'Price/hour':<20} ${est['price_per_hour']:>14.2f}")
            print(f"{'Estimated time':<20} {format_time(est['hours']):>15}")
            print(f"{'Estimated cost':<20} ${est['cost']:>14.2f}")
        else:
            print(f"\n{args.gpu} not available on {args.platform}")

    elif args.platform:
        # All GPUs on platform
        print(f"\nOptions on {PLATFORMS[args.platform].name}:")
        estimates = []
        for gpu_id in compatible:
            est = estimate_cost(args.tokens, gpu_id, args.platform, args.epochs)
            if est:
                estimates.append(est)
        print_comparison(estimates)

    else:
        # Default: compare best option per platform
        print("\nBest option per platform:")
        estimates = []

        for platform_id in PLATFORMS:
            best = None
            for gpu_id in compatible:
                est = estimate_cost(
                    args.tokens, gpu_id, platform_id, args.epochs
                )
                if est and (best is None or est['cost'] < best['cost']):
                    best = est
            if best:
                estimates.append(best)

        print_comparison(estimates)


if __name__ == "__main__":
    main()
