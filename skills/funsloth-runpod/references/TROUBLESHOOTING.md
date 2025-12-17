# Troubleshooting Guide

Common issues and solutions for Unsloth fine-tuning.

## Installation Issues

### "No module named 'unsloth'"
```bash
# Install unsloth
pip install unsloth

# If issues persist, install from source
pip uninstall unsloth -y
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### "No matching distribution for bitsandbytes"
```bash
# On Linux
pip install bitsandbytes

# On Windows (may need prebuilt)
pip install bitsandbytes-windows
# Or use conda
conda install -c conda-forge bitsandbytes
```

### CUDA version mismatch
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

## GPU Issues

### "CUDA out of memory"
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (try in order):**

1. Reduce batch size
```python
per_device_train_batch_size=1  # Minimum
gradient_accumulation_steps=8  # Maintain effective batch
```

2. Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
```

3. Reduce sequence length
```python
max_seq_length=1024  # Instead of 2048
```

4. Lower LoRA rank
```python
r=8  # Instead of 16 or 32
```

5. Close other GPU applications
```bash
# Check what's using GPU
nvidia-smi
# Kill processes if needed
kill -9 <PID>
```

### "No CUDA GPUs are available"
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
```

**Solutions:**
```bash
# Check NVIDIA driver
nvidia-smi

# If driver not found, install:
# Ubuntu
sudo apt install nvidia-driver-535

# Verify CUDA
nvcc --version
```

### "CUDA error: device-side assert triggered"
Usually indicates data issues (NaN, wrong shapes).

**Solution:** Check your data for:
- Empty strings
- NaN values
- Mismatched conversation formats

## Training Issues

### Loss is NaN or Inf
```
Training Loss: nan
```

**Solutions:**
1. Lower learning rate
```python
learning_rate=1e-5  # Instead of 2e-4
```

2. Enable gradient clipping
```python
max_grad_norm=1.0
```

3. Check data for issues
```python
# Look for empty or problematic examples
for i, ex in enumerate(dataset):
    if not ex.get("text") or len(ex["text"]) < 10:
        print(f"Problem at index {i}")
```

### Loss not decreasing
```
Step 100: Loss 2.5
Step 200: Loss 2.5
Step 300: Loss 2.5
```

**Solutions:**
1. Increase learning rate
```python
learning_rate=5e-4  # Experiment with higher values
```

2. Check data is properly formatted
```python
# Print a sample
print(dataset[0]["text"][:500])
```

3. Verify chat template matches model
```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")  # Must match model
```

4. Increase LoRA rank
```python
r=32  # More capacity
```

### Training too slow
**Solutions:**
1. Enable bf16 (if GPU supports)
```python
bf16=torch.cuda.is_bf16_supported()
```

2. Use packing for short sequences
```python
packing=True  # In SFTTrainer
```

3. Reduce logging frequency
```python
logging_steps=50  # Instead of 1 or 10
```

4. Use faster tokenization
```python
dataset_num_proc=4  # Parallel tokenization
```

## Data Issues

### "KeyError: 'conversations'"
Dataset doesn't have expected columns.

**Solution:** Check format and rename if needed
```python
# Print columns
print(dataset.column_names)

# Rename if needed
dataset = dataset.rename_column("dialogue", "conversations")
```

### Empty dataset after formatting
```python
# Debug
print(f"Before: {len(dataset)}")
dataset = dataset.map(format_fn)
print(f"After: {len(dataset)}")

# If 0, check format_fn
sample = dataset[0]
print(format_fn(sample))
```

### Tokenizer issues
```
Token indices sequence length is longer than the specified maximum
```

**Solution:** Increase max length or truncate
```python
max_seq_length=4096  # Increase

# Or filter long sequences
dataset = dataset.filter(lambda x: len(tokenizer.encode(x["text"])) < 2048)
```

## Saving Issues

### "Permission denied" when saving
```bash
# Check permissions
ls -la outputs/

# Fix permissions
chmod -R 755 outputs/
```

### "No space left on device"
```bash
# Check disk space
df -h

# Clean up old checkpoints
rm -rf outputs/checkpoint-*/
```

### Model not loading after save
Ensure both model and tokenizer are saved:
```python
model.save_pretrained("outputs/model")
tokenizer.save_pretrained("outputs/model")
```

## Hub Upload Issues

### "Repository not found"
```python
from huggingface_hub import HfApi
api = HfApi()

# Create repo first
api.create_repo("username/model-name", exist_ok=True)
```

### "Invalid token"
```bash
# Login with write token
huggingface-cli login

# Verify
huggingface-cli whoami
```

### Large file upload fails
Use Git LFS or chunked upload:
```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./outputs",
    repo_id="username/model",
    commit_message="Add model",
)
```

## Platform-Specific Issues

### Colab: Session crashed
- Usually OOM - reduce batch size
- Enable high-RAM runtime if available
- Use A100 runtime for larger models

### RunPod: SSH connection lost
- Training continues in background
- Reconnect and check: `tmux attach` or `tail -f training.log`

### HF Jobs: Job failed
```bash
# Check logs
huggingface-cli jobs logs <job_id>
```
Common causes:
- Missing dependencies in script header
- OOM - select larger GPU
- Script syntax errors

## Getting Help

### Debug information to collect
```python
import torch
import transformers
import unsloth

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"Unsloth: {unsloth.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Where to ask
- Unsloth GitHub Issues: https://github.com/unslothai/unsloth/issues
- Unsloth Discord: Check their GitHub for link
- Hugging Face Forums: https://discuss.huggingface.co/
