# 🚀 RunPod Quick Start Guide

## Your Active Pod
- **Pod ID**: `nppwrq5xwxh7cy`
- **Jupyter URL**: https://nppwrq5xwxh7cy-8888.proxy.runpod.net
- **Password**: `llama2024`
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Cost**: $0.39/hour

## Step-by-Step Instructions

### 1. Access Your Pod
Open Jupyter: https://nppwrq5xwxh7cy-8888.proxy.runpod.net

### 2. Open Terminal in Jupyter
Click: New → Terminal

### 3. Clone Your Updated Repository
```bash
cd /workspace
git clone https://github.com/sjanoe123/llama-runpod-training.git repo
cd repo
```

### 4. Run Setup Script
```bash
./setup.sh
```

### 5. Upload Your Model (6.72GB)

#### Option A: Via Google Drive (Recommended)
```bash
# Install gdown
pip install gdown

# Download from your Google Drive link
gdown 'YOUR_GOOGLE_DRIVE_SHARE_LINK' -O /workspace/model.tar.gz

# Extract
tar -xzf /workspace/model.tar.gz -C /workspace/
mv /workspace/Llama3.2-3B /workspace/model
```

#### Option B: Direct Upload via Jupyter
1. In Jupyter file browser, navigate to `/workspace/`
2. Create folder named `model`
3. Upload all files from your local `/Volumes/PS2000W/.llama/checkpoints/Llama3.2-3B/`

### 6. Prepare Training Data

#### Quick Test (3 samples):
```bash
python run_training.py --mode test
```
This will create sample data automatically.

#### Full Training Data:
```bash
# Upload your CLA-510 dataset, then:
python scripts/prepare_finetuning_data.py \
  --input /workspace/data/CLA-510-Fine-Tuning-Curation \
  --output /workspace/training_data/train.jsonl \
  --samples 1000000
```

### 7. Run Training

#### Test Run (Quick validation):
```bash
python run_training.py --mode test
```
- Uses 1,000 samples
- Takes ~30 minutes
- Validates setup

#### Full Training:
```bash
python run_training.py --mode full
```
- Uses all available samples
- Takes 2-3 hours on L4
- Produces fine-tuned model

### 8. Monitor Training
Training will show:
- GPU utilization
- Loss curves
- Estimated time remaining
- Memory usage

### 9. Get Results
After training:
```bash
ls -lh /workspace/outputs/
```

Your fine-tuned model will be in:
- `/workspace/outputs/final_model/` (fine-tuned)
- `/workspace/outputs/quantized_model/` (if AWQ applied)

## Troubleshooting

### Out of Memory
- Reduce batch size in `configs/finetuning_config.json`
- Increase gradient accumulation steps

### Model Not Found
```bash
ls -la /workspace/model/
# Should show config.json, pytorch_model.bin, tokenizer files
```

### Training Fails
Check logs:
```bash
tail -n 100 /workspace/outputs/training.log
```

## Next Steps

### For Production (A100):
1. Create new A100 pod
2. Clone this repo
3. Upload full 1M sample dataset
4. Run with `--mode full`
5. Apply AWQ quantization:
   ```bash
   python scripts/quantize_awq.py
   ```

### Download Trained Model:
```bash
# Compress
cd /workspace/outputs
tar -czf trained_model.tar.gz final_model/

# Download via Jupyter
# Navigate to /workspace/outputs/ and download trained_model.tar.gz
```

## Costs
- **L4 (Current)**: $0.39/hour × 3 hours = ~$1.17
- **A100 (Production)**: $1.89/hour × 14 hours = ~$26.46

## Support
- GitHub Issues: https://github.com/sjanoe123/llama-runpod-training/issues
- RunPod Console: https://www.runpod.io/console/pods