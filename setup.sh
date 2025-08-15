#!/bin/bash
set -e

echo "🚀 RunPod Llama Training Setup Script"
echo "======================================"

# Install Python packages
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0 accelerate==0.34.2 peft==0.12.0
pip install bitsandbytes==0.43.3 datasets==2.20.0 autoawq==0.2.6
pip install safetensors rich tqdm jupyterlab wandb

# Create workspace directories
echo "📁 Creating directories..."
mkdir -p /workspace/{model,data,outputs,logs}

# Test GPU
echo "🖥️ Testing GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Download model if not present
if [ ! -d "/workspace/model/llama-3.2-3b" ]; then
    echo "📥 Model not found locally. Please upload your model to /workspace/model/"
    echo "   Or set MODEL_PATH environment variable to HuggingFace model ID"
fi

echo "✅ Setup complete!"
echo ""
echo "To start training, run:"
echo "  python /workspace/repo/scripts/train.py"
echo ""
echo "To start Jupyter Lab, run:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
