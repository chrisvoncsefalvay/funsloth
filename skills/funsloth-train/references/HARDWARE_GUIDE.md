# Hardware Requirements Guide

VRAM requirements and GPU recommendations for Unsloth fine-tuning.

## Quick Reference: VRAM Requirements

### 4-bit LoRA Training (Recommended)

| Model Size | Min VRAM | Recommended | Batch Size 2 | Batch Size 4 |
|------------|----------|-------------|--------------|--------------|
| 0.5-1B | 4GB | 6GB | 4GB | 5GB |
| 3B | 6GB | 8GB | 6GB | 8GB |
| 7-8B | 8GB | 12GB | 10GB | 14GB |
| 13-14B | 12GB | 16GB | 14GB | 18GB |
| 32-34B | 20GB | 24GB | 22GB | 28GB |
| 70-72B | 40GB | 48GB | 45GB | 55GB |

### 8-bit LoRA Training

| Model Size | Min VRAM | Batch Size 2 |
|------------|----------|--------------|
| 7-8B | 14GB | 16GB |
| 13-14B | 20GB | 24GB |
| 32-34B | 36GB | 40GB |
| 70-72B | 80GB | 90GB |

### Full Fine-tuning (No LoRA)

| Model Size | Min VRAM |
|------------|----------|
| 0.5-1B | 16GB |
| 3B | 24GB |
| 7-8B | 48GB |
| 13-14B | 80GB |
| 70B+ | Multi-GPU required |

## GPU Recommendations

### Consumer GPUs

| GPU | VRAM | Max Model (4-bit LoRA) | Price Range |
|-----|------|------------------------|-------------|
| RTX 3060 | 12GB | 7B | $300-350 |
| RTX 3080 | 10GB | 7B | $500-600 |
| RTX 3090 | 24GB | 14-32B | $900-1200 |
| RTX 4060 Ti | 16GB | 8-14B | $400-450 |
| RTX 4070 | 12GB | 7B | $500-550 |
| RTX 4080 | 16GB | 8-14B | $1000-1100 |
| RTX 4090 | 24GB | 14-32B | $1600-2000 |

### Data Center GPUs

| GPU | VRAM | Max Model | Cloud Cost/hr |
|-----|------|-----------|---------------|
| A10G | 24GB | 14-32B | $1.00-1.50 |
| A100 40GB | 40GB | 34-70B | $1.50-4.00 |
| A100 80GB | 80GB | 70B+ | $2.00-6.00 |
| H100 | 80GB | 70B+ | $3.00-8.00 |
| L40S | 48GB | 70B | $1.50-2.50 |

## Memory Optimization Techniques

### Reducing VRAM Usage

1. **Use 4-bit quantization** (default with Unsloth)
   ```python
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name=MODEL_NAME,
       load_in_4bit=True,  # 70% VRAM reduction
   )
   ```

2. **Enable gradient checkpointing**
   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       use_gradient_checkpointing="unsloth",  # 30% reduction
   )
   ```

3. **Reduce batch size**
   - Each batch size reduction saves ~0.5-2GB
   - Compensate with gradient accumulation

4. **Reduce sequence length**
   - Halving max_seq_length roughly halves activation memory
   - Only if your data fits in shorter sequences

5. **Lower LoRA rank**
   - rank=8 uses less memory than rank=32
   - Trade-off with model capacity

### VRAM Calculator

Approximate VRAM formula for 4-bit LoRA:
```
VRAM ≈ (model_params / 1e9) * 0.6 + batch_size * 1.5 + base_overhead
```

Where:
- `model_params / 1e9 * 0.6` = 4-bit model weights (~0.6GB per billion params)
- `batch_size * 1.5` = Activations and gradients
- `base_overhead` ≈ 2GB for CUDA, framework, etc.

## Platform-Specific Considerations

### Local Training
- Close other GPU applications
- Browser hardware acceleration can use 1-2GB
- Monitor with `nvidia-smi -l 1`
- Keep temps under 85°C

### Colab
- Free tier: T4 (16GB) - good for 7-8B models
- Pro: A100 (40GB) - good for 34B models
- Session limits apply

### Cloud Providers
- HF Jobs: A10G, A100, H100 available
- RunPod: Consumer + data center GPUs
- Lambda Labs: Best A100/H100 prices
- Vast.ai: Cheapest consumer GPUs

## Multi-GPU Considerations

Unsloth focuses on single-GPU efficiency. For multi-GPU:

1. **LoRA is already efficient** - single GPU usually sufficient
2. **Data parallelism** - possible but often unnecessary
3. **Model parallelism** - only for >70B full fine-tuning

If you need multi-GPU:
```python
from accelerate import Accelerator
accelerator = Accelerator()
```

## Troubleshooting OOM

### Symptoms
```
CUDA out of memory. Tried to allocate X.XX GiB
```

### Solutions (in order)

1. **Reduce batch size**
   ```python
   per_device_train_batch_size=1  # Try 1 if OOM
   gradient_accumulation_steps=8  # Maintain effective batch
   ```

2. **Enable memory optimizations**
   ```python
   use_gradient_checkpointing="unsloth"  # Already default
   ```

3. **Reduce sequence length**
   ```python
   max_seq_length=1024  # Down from 2048
   ```

4. **Clear cache before training**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Use smaller LoRA rank**
   ```python
   r=8  # Down from 16
   ```

6. **Upgrade GPU or use cloud**

## Recommended Configurations

### 8GB VRAM (RTX 3060, etc.)
```python
MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LORA_RANK = 8
```

### 12GB VRAM (RTX 4070, etc.)
```python
MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LORA_RANK = 16
```

### 24GB VRAM (RTX 3090/4090, A10G)
```python
MODEL_NAME = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LORA_RANK = 32
```

### 40-48GB VRAM (A100 40GB, L40S)
```python
MODEL_NAME = "unsloth/llama-3.1-70b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LORA_RANK = 16
```

### 80GB VRAM (A100 80GB, H100)
```python
MODEL_NAME = "unsloth/llama-3.1-70b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 2
LORA_RANK = 32
```
