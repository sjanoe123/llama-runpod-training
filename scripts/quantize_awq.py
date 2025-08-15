#!/usr/bin/env python3
"""
Apply AWQ quantization to Llama 3.2 3B model
Uses medical calibration data for healthcare-optimized quantization
"""

import os
import sys
import argparse
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from typing import List
import json
import time

def load_calibration_data(calibration_file: str, max_samples: int = 512) -> List[str]:
    """Load calibration data from file"""
    
    print(f"📂 Loading calibration data from: {calibration_file}")
    
    calibration_data = []
    
    with open(calibration_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            calibration_data.append(line.strip())
    
    print(f"✅ Loaded {len(calibration_data)} calibration samples")
    
    return calibration_data

def quantize_model(model_path: str, calibration_file: str, output_path: str, 
                  bits: int = 4, group_size: int = 128):
    """Apply AWQ quantization to model"""
    
    print("\n🔧 Starting AWQ Quantization Process")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Calibration: {calibration_file}")
    print(f"Output: {output_path}")
    print(f"Settings: {bits}-bit, group_size={group_size}")
    print("=" * 50)
    
    # Load calibration data
    calibration_data = load_calibration_data(calibration_file)
    
    if not calibration_data:
        print("❌ No calibration data found!")
        sys.exit(1)
    
    # Load model
    print("\n📥 Loading original model...")
    print("⏳ This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        model = AutoAWQForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # Load on CPU first
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get original model size
    original_size = sum(p.numel() * p.element_size() for p in model.model.parameters()) / (1024**3)
    print(f"📊 Original model size: {original_size:.2f} GB")
    
    # Configure quantization
    print("\n⚙️  Configuring AWQ quantization...")
    
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",  # Optimized for GPU inference
        "modules_to_not_convert": None  # Quantize all layers
    }
    
    print(f"Configuration: {json.dumps(quant_config, indent=2)}")
    
    # Apply quantization
    print("\n🔨 Applying quantization with medical calibration...")
    print("⏳ This will take 30-45 minutes...")
    print("\nProgress:")
    
    quant_start = time.time()
    
    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data,
            batch_size=1,  # Process one sample at a time for stability
            use_triton=False,  # Disable triton for compatibility
            warmup=10,  # Warmup iterations
            auto_scale=True,  # Enable automatic scaling
            mse_range=True  # Use MSE for range estimation
        )
        
        quant_time = time.time() - quant_start
        print(f"\n✅ Quantization completed in {quant_time/60:.1f} minutes")
        
    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough RAM (16GB+ recommended)")
        print("2. Try reducing calibration samples")
        print("3. Check CUDA compatibility if using GPU")
        sys.exit(1)
    
    # Save quantized model
    print("\n💾 Saving quantized model...")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_start = time.time()
    
    try:
        model.save_quantized(output_path, safetensors=True)
        tokenizer.save_pretrained(output_path)
        
        save_time = time.time() - save_start
        print(f"✅ Model saved in {save_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        sys.exit(1)
    
    # Calculate quantized size
    quantized_files = list(Path(output_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in quantized_files) / (1024**3)
    
    # Save quantization info
    quant_info = {
        "original_model": model_path,
        "quantization_config": quant_config,
        "calibration_samples": len(calibration_data),
        "original_size_gb": original_size,
        "quantized_size_gb": quantized_size,
        "compression_ratio": original_size / quantized_size,
        "size_reduction_percent": (1 - quantized_size/original_size) * 100,
        "quantization_time_minutes": quant_time / 60
    }
    
    with open(Path(output_path) / "quantization_info.json", 'w') as f:
        json.dump(quant_info, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("✅ QUANTIZATION COMPLETE!")
    print("=" * 50)
    print(f"📊 Results:")
    print(f"   Original size: {original_size:.2f} GB")
    print(f"   Quantized size: {quantized_size:.2f} GB")
    print(f"   Compression: {quant_info['compression_ratio']:.1f}x")
    print(f"   Size reduction: {quant_info['size_reduction_percent']:.1f}%")
    print(f"   Time taken: {quant_time/60:.1f} minutes")
    print(f"\n📂 Quantized model saved to: {output_path}")
    
    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return True

def validate_quantized_model(model_path: str):
    """Quick validation of quantized model"""
    
    print("\n🧪 Validating quantized model...")
    
    try:
        # Check files exist
        required_files = [
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'quantization_info.json'
        ]
        
        model_dir = Path(model_path)
        for file in required_files:
            if not (model_dir / file).exists():
                print(f"❌ Missing file: {file}")
                return False
        
        # Check for quantized weights
        safetensor_files = list(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            print("❌ No quantized weight files found")
            return False
        
        print(f"✅ Found {len(safetensor_files)} quantized weight files")
        
        # Load quantization info
        with open(model_dir / "quantization_info.json", 'r') as f:
            info = json.load(f)
        
        print(f"✅ Quantization validated:")
        print(f"   - Method: AWQ {info['quantization_config']['w_bit']}-bit")
        print(f"   - Size: {info['quantized_size_gb']:.2f} GB")
        print(f"   - Compression: {info['compression_ratio']:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Quantize Llama model with AWQ')
    parser.add_argument('--model_path', type=str,
                       default='./models/llama-3.2-3b-original',
                       help='Path to original model')
    parser.add_argument('--calibration_file', type=str,
                       default='./calibration/medical_calibration.txt',
                       help='Path to calibration data')
    parser.add_argument('--output_path', type=str,
                       default='./models/llama-3.2-3b-awq',
                       help='Output path for quantized model')
    parser.add_argument('--bits', type=int, default=4,
                       help='Quantization bits (4 or 8)')
    parser.add_argument('--group_size', type=int, default=128,
                       help='Group size for quantization')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip post-quantization validation')
    
    args = parser.parse_args()
    
    print("🦙 Llama 3.2 3B AWQ Quantization")
    print("=" * 50)
    
    # Check inputs exist
    if not Path(args.model_path).exists():
        print(f"❌ Model not found: {args.model_path}")
        print("Please run: python scripts/2_download_model.py first")
        sys.exit(1)
    
    if not Path(args.calibration_file).exists():
        print(f"❌ Calibration file not found: {args.calibration_file}")
        print("Please run: python scripts/1_prepare_calibration.py first")
        sys.exit(1)
    
    # Check if output already exists
    if Path(args.output_path).exists():
        print(f"\n⚠️  Output directory already exists: {args.output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
        import shutil
        shutil.rmtree(args.output_path)
    
    # Run quantization
    success = quantize_model(
        args.model_path,
        args.calibration_file,
        args.output_path,
        args.bits,
        args.group_size
    )
    
    if not success:
        print("\n❌ Quantization failed!")
        sys.exit(1)
    
    # Validate
    if not args.skip_validation:
        if not validate_quantized_model(args.output_path):
            print("\n⚠️  Validation failed, but model may still work")
    
    print("\n" + "=" * 50)
    print("✅ AWQ quantization complete!")
    print(f"📂 Quantized model: {args.output_path}")
    print("\nNext step: Validate the model with:")
    print(f"python scripts/4_validate_model.py --model_path {args.output_path}")

if __name__ == "__main__":
    main()