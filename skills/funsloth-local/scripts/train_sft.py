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
#     "wandb",
# ]
# ///
"""
Supervised Fine-Tuning (SFT) Template for Unsloth

This script provides a production-ready template for SFT training.
Modify the CONFIGURATION section to customize for your use case.

Usage:
    python train_sft.py
    # Or with uv:
    uv run train_sft.py
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================
# CONFIGURATION - Modify these values
# ============================================

# Model settings
MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# Dataset settings
DATASET_NAME = "mlabonne/FineTome-100k"
DATASET_SPLIT = "train"
DATASET_FORMAT = "sharegpt"  # Options: alpaca, sharegpt, chatml, raw

# LoRA settings
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training settings
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
WARMUP_STEPS = 5
LR_SCHEDULER = "linear"
OPTIMIZER = "adamw_8bit"

# Output settings
OUTPUT_DIR = "outputs"
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 10

# Weights & Biases (optional)
USE_WANDB = False
WANDB_PROJECT = "funsloth-training"
WANDB_RUN_NAME = None  # Auto-generated if None

# Hub upload (optional)
PUSH_TO_HUB = False
HUB_MODEL_ID = None  # e.g., "username/model-name"

# ============================================
# SETUP
# ============================================

def check_gpu():
    """Verify GPU availability and print info."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU available. Please check your installation.\n"
            "Run: nvidia-smi to verify GPU visibility."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")
    return gpu_name, gpu_mem


def load_model():
    """Load model and tokenizer with Unsloth optimizations."""
    print(f"\nLoading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=LOAD_IN_4BIT,
    )

    print(f"Base parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters to the model."""
    print(f"\nApplying LoRA (rank={LORA_RANK})")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

    return model


def load_and_format_dataset(tokenizer):
    """Load dataset and format for training."""
    print(f"\nLoading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Dataset size: {len(dataset)} examples")

    # Apply chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    if DATASET_FORMAT == "sharegpt":
        dataset = standardize_sharegpt(dataset)

        def format_fn(example):
            return {"text": tokenizer.apply_chat_template(
                example["conversations"], tokenize=False
            )}

    elif DATASET_FORMAT == "alpaca":
        def format_fn(example):
            messages = [{"role": "user", "content": example["instruction"]}]
            if example.get("input"):
                messages[0]["content"] += f"\n\n{example['input']}"
            messages.append({"role": "assistant", "content": example["output"]})
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    elif DATASET_FORMAT == "chatml":
        def format_fn(example):
            return {"text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )}

    else:  # raw
        def format_fn(example):
            return {"text": example["text"]}

    dataset = dataset.map(format_fn)
    return dataset, tokenizer


def setup_wandb():
    """Initialize Weights & Biases tracking."""
    if not USE_WANDB:
        return

    import wandb

    run_name = WANDB_RUN_NAME or f"{MODEL_NAME.split('/')[-1]}-{DATASET_NAME.split('/')[-1]}"

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "lora_rank": LORA_RANK,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION,
        },
    )


def create_trainer(model, tokenizer, dataset):
    """Create SFTTrainer with configured settings."""

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        optim=OPTIMIZER,
        weight_decay=0.01,
        lr_scheduler_type=LR_SCHEDULER,
        seed=42,
        save_strategy=SAVE_STRATEGY,
        report_to="wandb" if USE_WANDB else "none",
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    return trainer


def save_model(model, tokenizer):
    """Save trained model in multiple formats."""
    print("\nSaving model...")

    # Save LoRA adapter
    lora_path = f"{OUTPUT_DIR}/lora_adapter"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA adapter saved to: {lora_path}")

    # Save merged 16-bit
    merged_path = f"{OUTPUT_DIR}/merged_16bit"
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to: {merged_path}")


def test_model(model, tokenizer):
    """Quick inference test."""
    print("\nTesting model...")

    FastLanguageModel.for_inference(model)

    test_messages = [
        {"role": "user", "content": "Hello! Can you introduce yourself?"}
    ]

    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test response:\n{response}")


# ============================================
# MAIN
# ============================================

def main():
    """Run SFT training pipeline."""
    print("=" * 50)
    print("Unsloth SFT Training")
    print("=" * 50)

    # Check GPU
    gpu_name, gpu_mem = check_gpu()

    # Load and setup model
    model, tokenizer = load_model()
    model = apply_lora(model)

    # Load and format dataset
    dataset, tokenizer = load_and_format_dataset(tokenizer)

    # Setup wandb
    setup_wandb()

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset)

    # Train
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)

    trainer_stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"Final loss: {trainer_stats.training_loss:.4f}")

    # Save
    save_model(model, tokenizer)

    # Test
    test_model(model, tokenizer)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
