# Model Selection Guide

Quick reference for choosing the right base model for your fine-tuning task.

## Model Families at a Glance

| Family | Best For | Strengths | Considerations |
|--------|----------|-----------|----------------|
| **Llama 3.x** | General-purpose | Balanced, well-documented | Requires license acceptance |
| **Qwen 2.5/3** | Multilingual, coding | Excellent Chinese, strong code | Larger context options |
| **Gemma 2/3** | Instruction following | Efficient, safety-focused | Google's approach to safety |
| **Phi-4** | Reasoning | Compact but powerful | Microsoft license |
| **Mistral** | Speed, chat | Fast inference | European origins |
| **DeepSeek** | Complex reasoning | R1-style thinking | Newer, less documentation |

## Recommended Models by Use Case

### General Instruction Following
```
Primary:   unsloth/llama-3.1-8b-unsloth-bnb-4bit
Larger:    unsloth/llama-3.1-70b-unsloth-bnb-4bit
Budget:    unsloth/Qwen2.5-7B-Instruct-bnb-4bit
```

### Code Generation
```
Primary:   unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit
Larger:    unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit
Alt:       unsloth/DeepSeek-Coder-V2-Lite-Instruct-bnb-4bit
```

### Multilingual (Chinese)
```
Primary:   unsloth/Qwen2.5-7B-Instruct-bnb-4bit
Larger:    unsloth/Qwen2.5-72B-Instruct-bnb-4bit
```

### Chat & Conversation
```
Primary:   unsloth/mistral-7b-instruct-v0.3-bnb-4bit
Alt:       unsloth/llama-3.1-8b-unsloth-bnb-4bit
Fast:      unsloth/Phi-3.5-mini-instruct-bnb-4bit
```

### Complex Reasoning / Math
```
Primary:   unsloth/Phi-4-bnb-4bit
Alt:       unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit
Budget:    unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit
```

### Safety-Focused Applications
```
Primary:   unsloth/gemma-2-9b-it-bnb-4bit
Larger:    unsloth/gemma-2-27b-it-bnb-4bit
```

### Research / Continued Pretraining
```
Primary:   unsloth/llama-3.1-8b-unsloth-bnb-4bit
Large:     unsloth/llama-3.1-70b-unsloth-bnb-4bit
```

## Size vs. Capability Tradeoffs

| Size | VRAM (4-bit) | Capability | Training Speed | Best For |
|------|--------------|------------|----------------|----------|
| 0.5-3B | 4-6GB | Limited | Very fast | Edge, prototyping |
| 7-8B | 8-12GB | Good | Fast | Most use cases |
| 13-14B | 12-16GB | Better | Moderate | Quality-focused |
| 32-34B | 20-24GB | Strong | Slower | Complex tasks |
| 70-72B | 40-48GB | Excellent | Slow | Maximum quality |

## Model-Specific Chat Templates

When using `get_chat_template()`, use the correct template:

| Model | Chat Template |
|-------|---------------|
| Llama 3.x | `llama-3.1` |
| Qwen 2.5 | `qwen-2.5-chat` |
| Gemma 2 | `gemma` |
| Phi-4 | `phi-4` |
| Mistral | `mistral` |
| DeepSeek | `deepseek` |

```python
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
```

## Unsloth 4-bit Model IDs

### Llama 3.x
- `unsloth/llama-3.1-8b-unsloth-bnb-4bit`
- `unsloth/llama-3.1-70b-unsloth-bnb-4bit`
- `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`

### Qwen 2.5
- `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-14B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-72B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit`

### Gemma 2
- `unsloth/gemma-2-2b-it-bnb-4bit`
- `unsloth/gemma-2-9b-it-bnb-4bit`
- `unsloth/gemma-2-27b-it-bnb-4bit`

### Phi
- `unsloth/Phi-3.5-mini-instruct-bnb-4bit`
- `unsloth/Phi-4-bnb-4bit`

### Mistral
- `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`

### DeepSeek
- `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit`
- `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit`

## Decision Flowchart

```
Start
  │
  ├─► Need multilingual (esp. Chinese)?
  │     └─► Yes: Qwen 2.5
  │
  ├─► Need code generation?
  │     └─► Yes: Qwen 2.5 Coder
  │
  ├─► Need complex reasoning/math?
  │     └─► Yes: Phi-4 or DeepSeek R1
  │
  ├─► Need safety/content filtering?
  │     └─► Yes: Gemma 2
  │
  ├─► Need fast chat?
  │     └─► Yes: Mistral
  │
  └─► Default: Llama 3.1 8B
```

## License Considerations

| Model | License | Commercial Use | Notes |
|-------|---------|----------------|-------|
| Llama 3.x | Llama 3 Community | Yes, with conditions | Accept on HF |
| Qwen 2.5 | Apache 2.0 / Qwen | Yes | Check specific model |
| Gemma 2 | Gemma Terms | Yes, with conditions | Google ToS |
| Phi-4 | MIT | Yes | Microsoft |
| Mistral | Apache 2.0 | Yes | Open source |
| DeepSeek | MIT | Yes | Open weights |

Always verify the current license on the model's Hugging Face page before deployment.
