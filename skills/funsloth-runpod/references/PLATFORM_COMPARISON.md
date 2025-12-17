# Training Platform Comparison

Comparison of cloud GPU platforms for Unsloth fine-tuning.

## Platform Overview

| Platform | Best For | Pricing | Min Billing |
|----------|----------|---------|-------------|
| **Local** | Iteration, privacy | Hardware cost | N/A |
| **HF Jobs** | Integration, simplicity | Premium | Per minute |
| **RunPod** | Flexibility, price | Budget | Per second |
| **Lambda Labs** | H100 availability | Competitive | Per second |
| **Vast.ai** | Cheapest GPUs | Very cheap | Per second |

## Detailed Comparison

### Local Training

**Pros:**
- No ongoing costs after hardware
- Full control and privacy
- No upload/download time
- Best for iteration

**Cons:**
- High upfront hardware cost
- Limited by your GPU
- No easy scaling
- Power and cooling costs

**Best For:** Regular fine-tuning, sensitive data, development

### Hugging Face Jobs

**Pros:**
- Seamless HF ecosystem integration
- Push to Hub directly from job
- PEP 723 script format
- Good documentation

**Cons:**
- Higher prices than competitors
- Requires PRO subscription
- Less GPU variety
- Queue times during peak

**Pricing (approx):**
| GPU | $/hour |
|-----|--------|
| A10G | $1.50 |
| A100 40GB | $4.00 |
| A100 80GB | $6.00 |
| H100 | $8.00 |

**Best For:** HF-centric workflows, pushing directly to Hub

### RunPod

**Pros:**
- Wide GPU selection (consumer + datacenter)
- Per-second billing
- Network volumes for persistence
- Good price/performance
- SSH access

**Cons:**
- More setup than HF Jobs
- Consumer GPUs can be preempted
- Variable availability

**Pricing (approx):**
| GPU | $/hour |
|-----|--------|
| RTX 3090 | $0.35 |
| RTX 4090 | $0.55 |
| A100 40GB | $1.50 |
| A100 80GB | $2.00 |
| H100 | $3.50 |

**Best For:** Budget training, flexibility, long runs

### Lambda Labs

**Pros:**
- Best H100 availability
- Simple pricing
- Good reliability
- Reserved instances option

**Cons:**
- Limited GPU variety
- Often sold out
- Minimum rental periods

**Pricing (approx):**
| GPU | $/hour |
|-----|--------|
| A100 40GB | $1.10 |
| A100 80GB | $1.29 |
| H100 | $2.49 |

**Best For:** H100 access, reliable enterprise use

### Vast.ai

**Pros:**
- Cheapest prices (marketplace)
- Huge GPU variety
- P2P marketplace model
- Good for experimentation

**Cons:**
- Variable reliability
- Host quality varies
- Less support
- Security concerns (shared hosts)

**Pricing (approx):**
| GPU | $/hour |
|-----|--------|
| RTX 3090 | $0.20-0.30 |
| RTX 4090 | $0.40-0.50 |
| A100 40GB | $1.00-1.30 |

**Best For:** Maximum savings, experimentation

## Cost Comparison Example

**Training 7B model on 10M tokens (~2 hours):**

| Platform | GPU | Time | Cost |
|----------|-----|------|------|
| Local (RTX 4090) | RTX 4090 | 2h | $0 (owned) |
| Vast.ai | RTX 4090 | 2h | $1.00 |
| RunPod | RTX 4090 | 2h | $1.10 |
| Lambda | A100 40GB | 1.5h | $1.65 |
| HF Jobs | A10G | 2.5h | $3.75 |

*Times and costs are approximate*

## Feature Comparison

| Feature | Local | HF Jobs | RunPod | Lambda | Vast.ai |
|---------|-------|---------|--------|--------|---------|
| GPU Variety | Your HW | Limited | Excellent | Good | Excellent |
| Pricing | Fixed | Premium | Budget | Competitive | Cheapest |
| SSH Access | N/A | No | Yes | Yes | Yes |
| Persistence | Yes | Job-based | Volumes | Instances | Varies |
| Auto-scaling | No | No | Manual | Manual | Manual |
| HF Integration | Manual | Native | Manual | Manual | Manual |
| WandB Support | Yes | Yes | Yes | Yes | Yes |
| Security | Your control | Good | Good | Good | Variable |

## Recommendations by Scenario

### Just Starting Out
1. **Colab Free** - T4 for small experiments
2. **Local RTX 3060** - If you have one
3. **Vast.ai** - Cheap experimentation

### Regular Fine-tuning (Weekly)
1. **Local RTX 4090** - Best long-term value
2. **RunPod** - When local isn't enough
3. **HF Jobs** - If deeply in HF ecosystem

### Production/Enterprise
1. **Lambda Labs** - Reliability + H100s
2. **HF Jobs** - Integration + support
3. **AWS/GCP/Azure** - If already using

### Training 70B+ Models
1. **Lambda H100** - Best availability
2. **RunPod A100 80GB** - Budget option
3. **HF Jobs H100** - Premium but integrated

### Sensitive Data
1. **Local** - Full control
2. **Lambda Reserved** - Dedicated instances
3. **Enterprise cloud** - Compliance options

## Quick Start by Platform

### Local
```bash
# Just run your script
python train_sft.py
```

### HF Jobs
```bash
# Create script with PEP 723 header
# Submit via CLI
huggingface-cli jobs create --config job.yaml
```

### RunPod
```bash
# Create pod via web or API
# SSH in, upload script, run
runpod ssh <pod_id>
python train_sft.py
```

### Lambda Labs
```bash
# Reserve instance via web
# SSH in
ssh ubuntu@<instance_ip>
python train_sft.py
```

### Vast.ai
```bash
# Rent via web interface
# SSH in (varies by host)
ssh -p <port> root@<ip>
python train_sft.py
```

## Monitoring Across Platforms

All platforms support:
- **WandB** - Add `report_to="wandb"` to TrainingArguments
- **TensorBoard** - Local logging
- **Trackio** - TRL-specific monitoring

Platform-specific:
- **HF Jobs** - Built-in logs via CLI
- **RunPod** - Dashboard + SSH
- **Lambda** - Dashboard
- **Vast.ai** - Basic dashboard
