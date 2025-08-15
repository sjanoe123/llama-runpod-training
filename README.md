# Llama 3.2 3B RunPod Training

This repository contains everything needed to fine-tune Llama 3.2 3B on RunPod.

## Quick Start on RunPod

1. Create a RunPod GPU pod with this configuration:
   - GPU: NVIDIA L4 (for testing) or A100 (for production)
   - Container Image: `runpod/pytorch:2.0.1-py3.10-cuda11.8-devel`
   - Disk: 50GB minimum

2. In the pod's startup command or terminal:
```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/llama-runpod-training.git /workspace/repo
cd /workspace/repo

# Run setup
bash setup.sh

# Start training
python scripts/train.py
```

## Environment Variables

- `MODEL_PATH`: Path to model (default: tries HuggingFace)
- `DATA_PATH`: Path to training data (default: `/workspace/data/train.jsonl`)

## Directory Structure

```
/workspace/
├── repo/           # This repository
├── model/          # Model files
├── data/           # Training data
├── outputs/        # Training outputs
└── logs/           # Training logs
```

## Using with RunPod

### Option 1: Direct Clone in Pod
```python
# In RunPod pod creation:
dockerCommand: "git clone https://github.com/YOUR_USERNAME/llama-runpod-training.git /workspace/repo && cd /workspace/repo && bash setup.sh"
```

### Option 2: As Docker Environment
```dockerfile
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8-devel
RUN git clone https://github.com/YOUR_USERNAME/llama-runpod-training.git /workspace/repo
WORKDIR /workspace/repo
RUN bash setup.sh
CMD ["python", "scripts/train.py"]
```
