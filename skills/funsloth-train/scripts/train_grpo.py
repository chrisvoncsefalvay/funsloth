# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
#     "torch>=2.0",
#     "transformers>=4.45",
#     "trl>=0.12",
#     "peft>=0.13",
#     "datasets>=2.18",
#     "accelerate>=0.28",
#     "bitsandbytes>=0.43",
# ]
# ///
"""
Group Relative Policy Optimization (GRPO) Template for Unsloth

GRPO uses reward functions for reinforcement learning with verifiable outputs.
Best for tasks with objective correctness criteria (math, code, factual).

The reward function must return a score for each completion.
Higher scores = better completions.

Usage:
    python train_grpo.py
"""

import re
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# Dataset with prompts for generation
DATASET_NAME = "openai/gsm8k"  # Math problems - good for GRPO
DATASET_SPLIT = "train"

# LoRA
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# GRPO Training
NUM_GENERATIONS = 4  # Completions per prompt for ranking
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
MAX_NEW_TOKENS = 256

OUTPUT_DIR = "outputs_grpo"

# ============================================
# REWARD FUNCTIONS
# ============================================

def extract_answer(text: str) -> str | None:
    """Extract final numerical answer from text."""
    # Look for "#### <number>" format (GSM8K style)
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Look for "The answer is <number>"
    match = re.search(r"[Tt]he answer is[:\s]*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Look for last number in text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def math_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """
    Reward function for math problems.
    Returns 1.0 for correct answer, 0.0 for incorrect.
    """
    rewards = []

    for completion, expected in zip(completions, ground_truth):
        predicted = extract_answer(completion)
        actual = extract_answer(expected)

        if predicted is not None and actual is not None:
            try:
                # Compare as floats to handle formatting differences
                reward = 1.0 if float(predicted) == float(actual) else 0.0
            except ValueError:
                reward = 0.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def length_penalty_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Bonus reward for concise answers.
    Penalizes very long or very short responses.
    """
    rewards = []
    target_length = 200  # tokens

    for completion in completions:
        length = len(completion.split())

        if length < 20:
            reward = -0.5  # Too short
        elif length > 500:
            reward = -0.3  # Too long
        else:
            # Bell curve around target
            reward = 0.2 * max(0, 1 - abs(length - target_length) / target_length)

        rewards.append(reward)

    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Bonus for proper formatting (step-by-step reasoning).
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Bonus for step-by-step
        if "step" in completion.lower() or re.search(r"\d\.", completion):
            reward += 0.1

        # Bonus for clear answer marker
        if "####" in completion or "answer is" in completion.lower():
            reward += 0.1

        rewards.append(reward)

    return rewards


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 50)
    print("Unsloth GRPO Training")
    print("=" * 50)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, "main", split=DATASET_SPLIT)

    # Format prompts
    def format_prompt(example):
        return {
            "prompt": f"Solve this math problem step by step:\n{example['question']}\n\nSolution:",
            "ground_truth": example["answer"],
        }

    dataset = dataset.map(format_prompt)
    dataset = dataset.select(range(min(1000, len(dataset))))  # Limit for demo

    print(f"Using {len(dataset)} examples")

    # Setup GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=[
            math_reward,        # Primary: correctness
            length_penalty_reward,  # Secondary: conciseness
            format_reward,      # Tertiary: formatting
        ],
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            num_generations=NUM_GENERATIONS,
            max_new_tokens=MAX_NEW_TOKENS,
            logging_steps=10,
            save_strategy="epoch",
        ),
    )

    # Train
    print("\nStarting GRPO training...")
    print("This uses RL - expect slower training but better reasoning.")
    stats = trainer.train()
    print(f"Training complete!")

    # Save
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Model saved to {OUTPUT_DIR}/lora_adapter")

    # Test
    print("\nTesting on sample problem...")
    FastLanguageModel.for_inference(model)

    test_prompt = "Solve this math problem step by step:\nIf a train travels 60 miles in 1.5 hours, what is its average speed?\n\nSolution:"
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
