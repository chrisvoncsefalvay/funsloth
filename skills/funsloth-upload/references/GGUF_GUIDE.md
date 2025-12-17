# GGUF Conversion Guide

Converting fine-tuned models to GGUF format for local inference.

## What is GGUF?

GGUF (GPT-Generated Unified Format) is the format used by:
- **llama.cpp** - CPU/GPU inference engine
- **Ollama** - Easy local model serving
- **LM Studio** - GUI for local models
- **GPT4All** - Desktop LLM application

## Quantization Options

### Quick Reference

| Method | Bits | Size (7B) | Quality | Use Case |
|--------|------|-----------|---------|----------|
| Q2_K | 2 | ~2.5GB | Lowest | Extreme constraints |
| Q3_K_M | 3 | ~3.5GB | Low | Very limited resources |
| **Q4_K_M** | 4 | ~4.5GB | **Good** | **Recommended default** |
| Q5_K_M | 5 | ~5.5GB | Better | Quality-focused |
| Q6_K | 6 | ~6.5GB | High | Near-original |
| Q8_0 | 8 | ~8GB | Excellent | Maximum quality |
| F16 | 16 | ~15GB | Original | No quantization |

### Detailed Comparison

```
Quality:  Q2_K < Q3_K_M < Q4_K_M < Q5_K_M < Q6_K < Q8_0 < F16
Size:     Q2_K < Q3_K_M < Q4_K_M < Q5_K_M < Q6_K < Q8_0 < F16
Speed:    Q2_K > Q3_K_M > Q4_K_M > Q5_K_M > Q6_K > Q8_0 > F16
```

### K-Variants Explained

- **Q4_0, Q5_0, Q8_0**: Original quantization (legacy)
- **Q4_K_M, Q5_K_M**: K-quants with mixed precision (better quality)
- **Q4_K_S, Q5_K_S**: K-quants, slightly smaller (good balance)

The `_M` (medium) variants are generally recommended over `_S` (small) for better quality.

## Converting with Unsloth

### Basic Conversion
```python
from unsloth import FastLanguageModel

# Load your trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/lora_adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Convert to GGUF
model.save_pretrained_gguf(
    "./gguf_output",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Multiple Quantizations
```python
for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.save_pretrained_gguf(
        f"./gguf_{quant}",
        tokenizer,
        quantization_method=quant,
    )
```

### Custom Filename
```python
model.save_pretrained_gguf(
    "./gguf_output",
    tokenizer,
    quantization_method="q4_k_m",
    # Output will be: ./gguf_output/model-unsloth.Q4_K_M.gguf
)
```

## Using with Ollama

### 1. Create Modelfile
```dockerfile
# Modelfile
FROM ./your-model.Q4_K_M.gguf

# Set the chat template (match your base model)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
```

### 2. Create and Run
```bash
# Create the model
ollama create my-model -f Modelfile

# Run it
ollama run my-model

# Or via API
curl http://localhost:11434/api/generate -d '{
  "model": "my-model",
  "prompt": "Hello!"
}'
```

### Chat Templates for Ollama

**Llama 3.x:**
```dockerfile
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
```

**Qwen 2.5:**
```dockerfile
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER stop "<|im_end|>"
```

**Mistral:**
```dockerfile
TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]{{ .Response }}"""
```

## Using with llama.cpp

### Basic Inference
```bash
./main -m your-model.Q4_K_M.gguf \
    -p "User: Hello!\nAssistant:" \
    -n 256 \
    --temp 0.7 \
    --top-p 0.9
```

### Server Mode
```bash
./server -m your-model.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080

# Then query:
curl http://localhost:8080/completion \
    -d '{"prompt": "Hello!", "n_predict": 128}'
```

### GPU Acceleration
```bash
# Use GPU layers (-ngl)
./main -m model.gguf -ngl 35  # Load 35 layers to GPU
./main -m model.gguf -ngl 99  # Load all layers to GPU
```

## Using with LM Studio

1. Download GGUF file
2. Open LM Studio → My Models → Import
3. Select the `.gguf` file
4. Configure chat template to match base model
5. Start chatting

## File Size Estimates

| Base Model | Q4_K_M | Q5_K_M | Q8_0 | F16 |
|------------|--------|--------|------|-----|
| 7B | 4.1GB | 4.9GB | 7.2GB | 14GB |
| 8B | 4.7GB | 5.6GB | 8.2GB | 16GB |
| 13B | 7.4GB | 8.9GB | 13GB | 26GB |
| 14B | 8.0GB | 9.5GB | 14GB | 28GB |
| 32B | 18GB | 22GB | 32GB | 64GB |
| 70B | 40GB | 48GB | 70GB | 140GB |

## RAM Requirements for Inference

| Model | Q4_K_M | Q8_0 |
|-------|--------|------|
| 7B | 6GB | 10GB |
| 13B | 10GB | 16GB |
| 32B | 22GB | 38GB |
| 70B | 45GB | 80GB |

Add 2-4GB overhead for context and system.

## Troubleshooting

### "Model not found" in Ollama
- Check file path in Modelfile
- Use absolute path if relative doesn't work

### Wrong output format
- Verify chat template matches base model
- Check stop tokens are correct

### Slow inference
- Enable GPU layers: `-ngl 35` or higher
- Use smaller quantization (Q4 instead of Q8)
- Reduce context length

### OOM during conversion
- Convert from merged model, not during training
- Ensure enough RAM (2x model size)
- Close other applications

## Best Practices

1. **Choose Q4_K_M as default** - best quality/size balance
2. **Include multiple quants** when uploading to HF Hub
3. **Test chat template** before distributing
4. **Document the base model** so users know the correct template
5. **Verify outputs** after conversion match original model
