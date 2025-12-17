# Training Methods Reference

Comparison of fine-tuning techniques supported by TRL and Unsloth.

## Methods at a Glance

| Method | Data Required | Use Case | Complexity |
|--------|---------------|----------|------------|
| **SFT** | Input → Output pairs | General instruction tuning | Low |
| **DPO** | Prompt + Chosen/Rejected | Preference alignment | Medium |
| **GRPO** | Prompts + Reward function | RL with verifiable tasks | High |
| **ORPO** | Same as DPO | Combined SFT + preference | Medium |
| **KTO** | Input + Good/Bad label | Binary feedback | Medium |

## Supervised Fine-Tuning (SFT)

### When to Use
- Teaching model new skills or knowledge
- Adapting model to specific domain
- Instruction following
- Most common starting point

### Data Format
```json
// Alpaca
{"instruction": "Summarize this:", "input": "Long text...", "output": "Summary..."}

// ShareGPT
{"conversations": [
  {"from": "human", "value": "Question"},
  {"from": "gpt", "value": "Answer"}
]}

// ChatML
{"messages": [
  {"role": "user", "content": "Question"},
  {"role": "assistant", "content": "Answer"}
]}
```

### Key Parameters
| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Learning rate | 1e-5 to 5e-4 | 2e-4 is Unsloth default |
| Batch size | 1-8 | Depends on VRAM |
| Epochs | 1-3 | More for small datasets |
| LoRA rank | 8-64 | 16 is good default |

### Example
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        learning_rate=2e-4,
        num_train_epochs=1,
    ),
)
```

## Direct Preference Optimization (DPO)

### When to Use
- Aligning model with human preferences
- Improving response quality
- After SFT for refinement
- When you have comparison data

### Data Format
```json
{
  "prompt": "Write a poem about AI",
  "chosen": "Silicon dreams in electric streams...",
  "rejected": "AI is good. AI is great. The end."
}
```

### Key Parameters
| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Beta | 0.01 to 0.5 | Higher = more conservative |
| Learning rate | 1e-6 to 5e-5 | Lower than SFT |
| Reference model | None or frozen | None = implicit reference |

### Example
```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Implicit reference
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=DPOConfig(
        beta=0.1,
        learning_rate=5e-6,
    ),
)
```

### Beta Parameter
- `beta = 0.1`: Standard setting, moderate constraint
- `beta < 0.1`: Allows more deviation from reference
- `beta > 0.1`: Stronger constraint, more conservative

## Group Relative Policy Optimization (GRPO)

### When to Use
- Tasks with verifiable correctness (math, code)
- When you can define a reward function
- Online RL training
- Improving reasoning

### Data Format
```json
{
  "prompt": "What is 2 + 2?",
  "ground_truth": "4"  // For reward calculation
}
```

### Reward Function
```python
def reward_fn(completions, ground_truth, **kwargs):
    rewards = []
    for completion, expected in zip(completions, ground_truth):
        # Check if answer is correct
        is_correct = extract_answer(completion) == expected
        rewards.append(1.0 if is_correct else 0.0)
    return rewards
```

### Key Parameters
| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Num generations | 4-8 | Completions per prompt |
| Learning rate | 1e-6 to 1e-5 | Very low for RL |
| Max new tokens | 128-512 | For generation |

### Example
```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_funcs=[correctness_reward, format_reward],
    args=GRPOConfig(
        num_generations=4,
        learning_rate=1e-5,
    ),
)
```

## ORPO (Odds Ratio Preference Optimization)

### When to Use
- Single-stage alignment (combines SFT + preference)
- Simpler than DPO pipeline
- Same data as DPO

### Key Difference
ORPO doesn't need a reference model - it learns SFT and preferences together.

### Example
```python
from trl import ORPOTrainer, ORPOConfig

trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=ORPOConfig(
        beta=0.1,
        learning_rate=8e-6,
    ),
)
```

## KTO (Kahneman-Tversky Optimization)

### When to Use
- Binary feedback (thumbs up/down)
- Simpler preference data
- Based on prospect theory

### Data Format
```json
{
  "prompt": "Question",
  "completion": "Response",
  "label": true  // true = good, false = bad
}
```

### Example
```python
from trl import KTOTrainer, KTOConfig

trainer = KTOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=KTOConfig(
        beta=0.1,
        learning_rate=5e-6,
    ),
)
```

## Method Selection Guide

```
Start
  │
  ├─► Have preference pairs (chosen/rejected)?
  │     ├─► Yes + Want single stage? → ORPO
  │     └─► Yes + Want after SFT? → DPO
  │
  ├─► Have binary feedback (good/bad)?
  │     └─► Yes → KTO
  │
  ├─► Have verifiable tasks (math/code)?
  │     └─► Yes + Can write reward fn? → GRPO
  │
  └─► Default: SFT
```

## Typical Pipeline

### Simple: SFT Only
```
Raw Data → Format → SFT → Deploy
```

### With Alignment: SFT + DPO
```
Raw Data → SFT → Preference Data → DPO → Deploy
```

### With RL: SFT + GRPO
```
Raw Data → SFT → Reward Design → GRPO → Deploy
```

### Single Stage: ORPO
```
Preference Data → ORPO → Deploy
```

## Performance Comparison

| Method | Training Speed | Data Efficiency | Alignment Quality |
|--------|---------------|-----------------|-------------------|
| SFT | Fast | Good | Basic |
| DPO | Medium | Moderate | Good |
| GRPO | Slow | Lower | Excellent for verifiable |
| ORPO | Medium | Good | Good |
| KTO | Medium | Good | Good |

## Common Mistakes

1. **Using DPO without SFT first**
   - DPO works best on already-capable models
   - Do SFT first for foundational skills

2. **Too high learning rate for preference methods**
   - Use 10-100x lower LR than SFT
   - Preference methods are sensitive

3. **Poor reward function design (GRPO)**
   - Reward must clearly distinguish good/bad
   - Test reward fn before training

4. **Preference data quality**
   - Chosen must be clearly better than rejected
   - Bad comparisons hurt more than help
