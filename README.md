# funsloth

A Claude Code skills marketplace for fine-tuning language models with [Unsloth](https://github.com/unslothai/unsloth). It's premised on the [dorkestration](https://chrisvoncsefalvay.com/posts/dorkestration/) paradigm, enabling seamless orchestration of multiple skills to achieve complex workflows using Claude Code.

## Overview

funsloth provides six connected skills that guide you through the complete fine-tuning workflow:

```
funsloth-check → funsloth-train → [hfjobs|runpod|local] → funsloth-upload
```

| Skill | Description |
|-------|-------------|
| `funsloth-check` | Validate datasets, analyze token counts, calculate Chinchilla optimality |
| `funsloth-train` | Generate Unsloth training notebooks with sensible defaults or custom config |
| `funsloth-hfjobs` | Train on Hugging Face Jobs cloud GPUs |
| `funsloth-runpod` | Train on RunPod GPU instances |
| `funsloth-local` | Train on your local GPU |
| `funsloth-upload` | Generate model cards and upload to Hugging Face Hub |

## Installation

```bash
claude plugin install funsloth
```

Or install from source:
```bash
git clone https://github.com/chrisvoncsefalvay/funsloth
cd funsloth
claude plugin install .
```

## Usage

### Quick start

Just tell Claude what you want to do:

```
> I want to fine-tune Llama 3.1 8B on my custom dataset
```

Claude will automatically invoke the appropriate skills.

### Manual skill invocation

You can also invoke skills directly:

```
> /funsloth-check mlabonne/FineTome-100k
> /funsloth-train
> /funsloth-local
> /funsloth-upload
```

## Supported models

| Family | Sizes | Recommended 4-bit |
|--------|-------|-------------------|
| Llama 3.x | 1B, 3B, 8B, 70B | `unsloth/llama-3.1-8b-unsloth-bnb-4bit` |
| Qwen 2.5/3 | 0.5B-72B | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Gemma 2/3 | 2B, 9B, 27B | `unsloth/gemma-2-9b-it-bnb-4bit` |
| Phi-4 | 14B | `unsloth/Phi-4-bnb-4bit` |
| Mistral | 7B, 8x7B | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` |
| DeepSeek | 7B+ | `unsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit` |

## Supported training techniques

- **SFT** - Supervised Fine-Tuning
- **[DPO](https://arxiv.org/abs/2305.18290)** - Direct Preference Optimization (Rafailov et al., NeurIPS 2023)
- **[GRPO](https://arxiv.org/abs/2402.03300)** - Group Relative Policy Optimization (DeepSeekMath, 2024)
- **[ORPO](https://arxiv.org/abs/2403.07691)** - Odds Ratio Preference Optimization (Hong et al., EMNLP 2024)
- **[KTO](https://arxiv.org/abs/2402.01306)** - Kahneman-Tversky Optimization (Ethayarajh et al., ICML 2024)

## Data formats



| Format | Structure | Use Case |
|--------|-----------|----------|
| Raw Corpus | `{"text": "..."}` | Continued pretraining |
| Alpaca | `{"instruction", "input", "output"}` | Instruction tuning |
| ShareGPT | `[{"from": "human", "value": "..."}]` | Conversations |
| ChatML | `[{"role": "user", "content": "..."}]` | Native chat |

## Requirements

- Claude Code CLI
- Python 3.10+
- CUDA-capable GPU (for local training)
- Hugging Face account (for dataset/model hosting)

## License

MIT

## Acknowledgments

- [Ben Burtenshaw](https://huggingface.co/blog/hf-skills-training), the OG of Claude Code fine-tuning skills
- [Unsloth](https://github.com/unslothai/unsloth) - fast LLM fine-tuning
- [Hugging Face](https://huggingface.co) - model and dataset hosting
- [TRL](https://github.com/huggingface/trl) - RL for transformers!
