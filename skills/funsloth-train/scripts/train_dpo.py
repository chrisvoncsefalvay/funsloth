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
Direct Preference Optimization (DPO) Template for Unsloth

DPO trains models using preference pairs (chosen vs rejected responses).
Use this when you have human preference data.

Requires dataset with columns:
- prompt: The input prompt
- chosen: The preferred response
- rejected: The less preferred response

Usage:
    python train_dpo.py
"""

import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# Patch DPOTrainer for 2x speed
PatchDPOTrainer()

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# DPO Dataset (must have prompt, chosen, rejected columns)
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"

# LoRA
LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# DPO Training
BETA = 0.1  # KL penalty coefficient (higher = more conservative)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6  # Lower than SFT typically
NUM_EPOCHS = 1

OUTPUT_DIR = "outputs_dpo"

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 50)
    print("Unsloth DPO Training")
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

    # Load DPO dataset
    print(f"\nLoading DPO dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Verify columns exist
    required_cols = {"prompt", "chosen", "rejected"}
    if not required_cols.issubset(dataset.column_names):
        # Try common alternatives
        col_mapping = {
            "question": "prompt",
            "response": "chosen",
            "rejected_response": "rejected",
        }
        dataset = dataset.rename_columns({
            k: v for k, v in col_mapping.items()
            if k in dataset.column_names
        })

    print(f"Dataset size: {len(dataset)} preference pairs")

    # Setup DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Uses implicit reference (saves memory)
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_EPOCHS,
            beta=BETA,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_SEQ_LENGTH // 2,
            logging_steps=10,
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            save_strategy="epoch",
        ),
    )

    # Train
    print("\nStarting DPO training...")
    stats = trainer.train()
    print(f"Training complete! Final loss: {stats.training_loss:.4f}")

    # Save
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"Model saved to {OUTPUT_DIR}/lora_adapter")


if __name__ == "__main__":
    main()
