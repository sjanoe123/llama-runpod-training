#!/usr/bin/env python3
"""
RunPod Training Launcher for Llama 3.2 3B
Automatically configures based on available GPU and handles all setup
"""

import os
import sys
import torch
import json
import subprocess
from pathlib import Path
import argparse
import time

def detect_gpu():
    """Detect GPU type and available memory"""
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        return None, 0
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️ GPU Detected: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    return gpu_name, gpu_memory

def adjust_config_for_gpu(gpu_name, gpu_memory):
    """Adjust training config based on GPU"""
    
    config = {
        "batch_size": 4,
        "gradient_accumulation": 8,
        "max_memory": f"{int(gpu_memory * 0.9)}GB",
        "fp16": True,
        "load_in_4bit": True
    }
    
    if "L4" in gpu_name:
        print("📊 Configuring for L4 GPU (Test mode)")
        config["batch_size"] = 2
        config["gradient_accumulation"] = 16
        config["samples"] = 10000
        
    elif "A100" in gpu_name:
        print("🚀 Configuring for A100 GPU (Production mode)")
        config["batch_size"] = 8
        config["gradient_accumulation"] = 4
        config["samples"] = 1000000
        config["fp16"] = False
        config["bf16"] = True
        
    elif "A6000" in gpu_name or "A40" in gpu_name:
        print("⚡ Configuring for A6000/A40 GPU")
        config["batch_size"] = 6
        config["gradient_accumulation"] = 6
        config["samples"] = 100000
        
    else:
        print(f"⚠️ Unknown GPU: {gpu_name}, using conservative settings")
        config["batch_size"] = 1
        config["gradient_accumulation"] = 32
        config["samples"] = 1000
    
    return config

def check_model_exists():
    """Check if model files exist"""
    model_path = Path("/workspace/model")
    
    if not model_path.exists():
        print("\n❌ Model not found at /workspace/model/")
        print("\n📥 To upload your model:")
        print("1. In Jupyter, use the upload button")
        print("2. Or download from cloud storage:")
        print("   wget YOUR_MODEL_URL -O model.tar.gz")
        print("   tar -xzf model.tar.gz -C /workspace/model/")
        return False
    
    # Check for key files
    required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    missing = []
    
    for file in required_files:
        if not (model_path / file).exists():
            # Check for safetensors alternative
            if file == "pytorch_model.bin" and (model_path / "model.safetensors").exists():
                continue
            missing.append(file)
    
    if missing:
        print(f"⚠️ Missing model files: {missing}")
        return False
    
    print("✅ Model files found")
    return True

def check_data_exists():
    """Check if training data exists"""
    data_path = Path("/workspace/training_data")
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print("📁 Created training_data directory")
    
    train_file = data_path / "train.jsonl"
    
    if not train_file.exists():
        print("\n❌ Training data not found")
        print("\n📥 To prepare training data:")
        print("1. Upload your data to /workspace/training_data/")
        print("2. Or run: python scripts/prepare_finetuning_data.py")
        
        # Create sample data for testing
        print("\n📝 Creating sample data for testing...")
        sample_data = [
            {"text": "Patient presents with chest pain. Diagnosis: Angina pectoris (I20.9)"},
            {"text": "Claim for MRI brain with contrast. CPT: 70553. Medical necessity confirmed."},
            {"text": "Prior authorization required for specialty medication. Review per CMS guidelines."}
        ]
        
        with open(train_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        print("✅ Created sample training data (3 examples)")
        return True
    
    # Count lines in training data
    with open(train_file, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"✅ Training data found: {line_count:,} examples")
    return True

def run_training(config):
    """Run the actual training"""
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    # Update config file with GPU-specific settings
    config_path = Path("configs/finetuning_config.json")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        
        # Update with GPU-specific settings
        full_config["training_config"]["per_device_train_batch_size"] = config["batch_size"]
        full_config["training_config"]["gradient_accumulation_steps"] = config["gradient_accumulation"]
        full_config["model_config"]["max_memory"]["0"] = config["max_memory"]
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
    
    # Run training script
    cmd = [
        "python", "scripts/finetune_medical.py",
        "--config", "configs/finetuning_config.json",
        "--output_dir", "/workspace/outputs",
        "--num_samples", str(config["samples"])
    ]
    
    print(f"\n📊 Training Configuration:")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Gradient Accumulation: {config['gradient_accumulation']}")
    print(f"   Samples: {config['samples']:,}")
    print(f"   Memory Limit: {config['max_memory']}")
    
    estimated_time = config["samples"] / 10000  # rough estimate: 1 hour per 10K samples
    print(f"\n⏱️ Estimated Time: {estimated_time:.1f} hours")
    
    print("\n🔄 Starting training process...")
    print("   Command:", " ".join(cmd))
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            return True
        else:
            print(f"\n❌ Training failed with code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='RunPod Training Launcher')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                       help='Training mode: test (small) or full (complete)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    print("🦙 Llama 3.2 3B Healthcare Training Launcher")
    print("="*60)
    
    # Detect GPU
    gpu_name, gpu_memory = detect_gpu()
    if not gpu_name:
        sys.exit(1)
    
    # Adjust config
    config = adjust_config_for_gpu(gpu_name, gpu_memory)
    
    # Override samples if in test mode
    if args.mode == 'test':
        config["samples"] = min(config["samples"], 1000)
        print(f"📋 Test mode: Limited to {config['samples']} samples")
    
    # Check prerequisites
    if not args.skip_checks:
        print("\n🔍 Checking prerequisites...")
        
        model_ok = check_model_exists()
        data_ok = check_data_exists()
        
        if not model_ok:
            print("\n⚠️ Please upload model files before training")
            print("See instructions above")
            sys.exit(1)
        
        if not data_ok:
            print("\n⚠️ Using sample data for testing")
            print("Upload real training data for better results")
    
    # Confirm before starting
    print("\n" + "="*60)
    print("Ready to start training with:")
    print(f"  GPU: {gpu_name}")
    print(f"  Samples: {config['samples']:,}")
    print(f"  Mode: {args.mode}")
    print("="*60)
    
    input("\nPress Enter to start training (Ctrl+C to cancel)...")
    
    # Run training
    success = run_training(config)
    
    if success:
        print("\n🎉 Training pipeline completed!")
        print("\n📁 Output location: /workspace/outputs/")
        
        # If AWQ quantization is needed
        if args.mode == 'full':
            print("\n🔧 To run AWQ quantization:")
            print("   python scripts/quantize_awq.py")
    else:
        print("\n❌ Training pipeline failed")
        print("Check logs for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())