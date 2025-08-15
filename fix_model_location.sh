#!/bin/bash

echo "🔍 Finding and setting up Llama 3.2 3B model in RunPod"
echo "================================================="

# Find the tar.gz file
echo -e "\n📂 Searching for model tar.gz file..."
MODEL_TAR=$(find /workspace -name "llama3.2-3b-hf.tar.gz" 2>/dev/null | head -1)

if [ -z "$MODEL_TAR" ]; then
    echo "❌ Model tar.gz not found!"
    echo ""
    echo "Looking for any tar.gz files in /workspace:"
    find /workspace -name "*.tar.gz" -type f 2>/dev/null
    exit 1
fi

echo "✅ Found model at: $MODEL_TAR"

# Check if model directory already exists
if [ -d "/workspace/model" ]; then
    echo -e "\n⚠️  /workspace/model already exists"
    echo "Contents:"
    ls -lh /workspace/model/
    read -p "Delete and re-extract? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf /workspace/model
    else
        echo "Keeping existing model directory"
        exit 0
    fi
fi

# Extract the model
echo -e "\n📦 Extracting model..."
cd /workspace

# Try to extract and see what's inside
echo "Checking tar contents first..."
tar -tzf "$MODEL_TAR" | head -5

# Extract the tar file
tar -xzf "$MODEL_TAR"

# Find what was extracted
echo -e "\n🔍 Looking for extracted model files..."

# Check if it extracted to a named directory
if [ -d "/workspace/llama3.2-3b-hf" ]; then
    echo "✅ Found extracted directory: llama3.2-3b-hf"
    mv /workspace/llama3.2-3b-hf /workspace/model
elif [ -d "/workspace/Llama3.2-3B" ]; then
    echo "✅ Found extracted directory: Llama3.2-3B"
    mv /workspace/Llama3.2-3B /workspace/model
elif [ -d "/workspace/llama-3.2-3b-hf" ]; then
    echo "✅ Found extracted directory: llama-3.2-3b-hf"
    mv /workspace/llama-3.2-3b-hf /workspace/model
else
    # Check if files were extracted directly to /workspace
    if [ -f "/workspace/config.json" ]; then
        echo "⚠️  Files extracted directly to /workspace"
        echo "Moving to /workspace/model..."
        mkdir -p /workspace/model
        mv /workspace/*.json /workspace/model/ 2>/dev/null
        mv /workspace/*.bin /workspace/model/ 2>/dev/null
        mv /workspace/*.safetensors /workspace/model/ 2>/dev/null
        mv /workspace/*.model /workspace/model/ 2>/dev/null
        mv /workspace/tokenizer* /workspace/model/ 2>/dev/null
        mv /workspace/special_tokens* /workspace/model/ 2>/dev/null
    else
        echo "❌ Could not find extracted model files!"
        echo "Current /workspace contents:"
        ls -la /workspace/
        exit 1
    fi
fi

# Verify model files
echo -e "\n✅ Verifying model files in /workspace/model..."
ls -lh /workspace/model/

# Check for required files
REQUIRED_FILES=("config.json")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "/workspace/model/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

# Check for model weights (either .bin or .safetensors)
if [ ! -f "/workspace/model/pytorch_model.bin" ] && [ ! -f "/workspace/model/model.safetensors" ]; then
    # Check for sharded model
    if ! ls /workspace/model/*.safetensors >/dev/null 2>&1 && ! ls /workspace/model/*.bin >/dev/null 2>&1; then
        MISSING_FILES+=("model weights (.bin or .safetensors)")
    fi
fi

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "\n✅ All required model files found!"
    echo ""
    echo "Model setup complete! You can now run:"
    echo "  cd /workspace/repo"
    echo "  python run_training.py --mode test"
else
    echo -e "\n⚠️  Missing files: ${MISSING_FILES[*]}"
    echo "But continuing anyway..."
fi

echo -e "\n📊 Model statistics:"
du -sh /workspace/model/
echo "Number of files: $(find /workspace/model -type f | wc -l)"

echo -e "\n✅ Model setup complete!"