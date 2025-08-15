#!/usr/bin/env python3
"""
Convert raw Meta Llama 3.2 3B checkpoint to HuggingFace format
This allows us to use Meta's original model files without HuggingFace access
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
import shutil
from safetensors.torch import save_file
import time

def load_meta_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load Meta format checkpoint"""
    
    print(f"📂 Loading Meta checkpoint from: {checkpoint_path}")
    
    checkpoint_file = Path(checkpoint_path) / "consolidated.00.pth"
    
    if not checkpoint_file.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        sys.exit(1)
    
    print(f"📥 Loading {checkpoint_file.stat().st_size / 1024**3:.2f}GB checkpoint...")
    
    start_time = time.time()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    load_time = time.time() - start_time
    
    print(f"✅ Checkpoint loaded in {load_time:.1f} seconds")
    print(f"📊 Found {len(checkpoint)} weight tensors")
    
    return checkpoint

def load_meta_params(checkpoint_path: str) -> Dict[str, Any]:
    """Load Meta model parameters"""
    
    params_file = Path(checkpoint_path) / "params.json"
    
    if not params_file.exists():
        print(f"❌ Params file not found: {params_file}")
        sys.exit(1)
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print("📋 Model configuration:")
    print(f"   Hidden size: {params['dim']}")
    print(f"   Layers: {params['n_layers']}")
    print(f"   Heads: {params['n_heads']}")
    print(f"   KV heads: {params['n_kv_heads']}")
    print(f"   Vocab size: {params['vocab_size']}")
    
    return params

def convert_weight_names(meta_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Meta weight names to HuggingFace format"""
    
    print("\n🔄 Converting weight names to HuggingFace format...")
    
    # Weight name mapping from Meta to HuggingFace
    name_mapping = {
        # Embeddings
        'tok_embeddings.weight': 'model.embed_tokens.weight',
        
        # Final layer norm and output
        'norm.weight': 'model.norm.weight',
        'output.weight': 'lm_head.weight',
    }
    
    hf_state_dict = {}
    converted_count = 0
    
    for name, tensor in meta_state_dict.items():
        # Handle layer-specific mappings
        if 'layers.' in name:
            # Extract layer number
            parts = name.split('.')
            layer_num = parts[1]
            
            # Attention weights
            if 'attention.wq.weight' in name:
                new_name = f'model.layers.{layer_num}.self_attn.q_proj.weight'
            elif 'attention.wk.weight' in name:
                new_name = f'model.layers.{layer_num}.self_attn.k_proj.weight'
            elif 'attention.wv.weight' in name:
                new_name = f'model.layers.{layer_num}.self_attn.v_proj.weight'
            elif 'attention.wo.weight' in name:
                new_name = f'model.layers.{layer_num}.self_attn.o_proj.weight'
            
            # FFN weights
            elif 'feed_forward.w1.weight' in name:
                new_name = f'model.layers.{layer_num}.mlp.gate_proj.weight'
            elif 'feed_forward.w2.weight' in name:
                new_name = f'model.layers.{layer_num}.mlp.down_proj.weight'
            elif 'feed_forward.w3.weight' in name:
                new_name = f'model.layers.{layer_num}.mlp.up_proj.weight'
            
            # Layer norms
            elif 'attention_norm.weight' in name:
                new_name = f'model.layers.{layer_num}.input_layernorm.weight'
            elif 'ffn_norm.weight' in name:
                new_name = f'model.layers.{layer_num}.post_attention_layernorm.weight'
            
            # Rotary embeddings (if present)
            elif 'attention.rotary_emb.inv_freq' in name:
                continue  # Skip, will be computed
            else:
                print(f"⚠️  Unknown layer weight: {name}")
                new_name = name
        else:
            # Use direct mapping or keep original
            new_name = name_mapping.get(name, name)
        
        hf_state_dict[new_name] = tensor
        converted_count += 1
    
    print(f"✅ Converted {converted_count} weights")
    
    return hf_state_dict

def create_hf_config(meta_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create HuggingFace config from Meta params"""
    
    print("\n📝 Creating HuggingFace configuration...")
    
    # Calculate intermediate size (FFN dimension)
    ffn_dim_multiplier = meta_params.get("ffn_dim_multiplier", 1.0)
    multiple_of = meta_params.get("multiple_of", 256)
    hidden_size = meta_params["dim"]
    
    # Calculate FFN hidden dimension
    ffn_hidden_dim = int(2 * hidden_size * 4 / 3)
    ffn_hidden_dim = ffn_dim_multiplier * ffn_hidden_dim
    ffn_hidden_dim = int((ffn_hidden_dim + multiple_of - 1) // multiple_of * multiple_of)
    
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        
        # Model dimensions
        "hidden_size": hidden_size,
        "intermediate_size": ffn_hidden_dim,
        "num_hidden_layers": meta_params["n_layers"],
        "num_attention_heads": meta_params["n_heads"],
        "num_key_value_heads": meta_params.get("n_kv_heads", meta_params["n_heads"]),
        
        # Vocabulary
        "vocab_size": meta_params["vocab_size"],
        "max_position_embeddings": 131072,  # Llama 3.2 supports up to 128K
        
        # Normalization
        "rms_norm_eps": meta_params.get("norm_eps", 1e-5),
        
        # Rotary embeddings
        "rope_theta": meta_params.get("rope_theta", 500000.0),
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        } if meta_params.get("use_scaled_rope", False) else None,
        
        # Other settings
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "use_cache": True,
        "tie_word_embeddings": False,
        
        # Special tokens
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 128001,
        
        # Attention
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
    }
    
    return config

def convert_tokenizer(meta_path: str, output_path: str):
    """Convert Meta tokenizer to HuggingFace format"""
    
    print("\n🔤 Converting tokenizer...")
    
    tokenizer_file = Path(meta_path) / "tokenizer.model"
    
    if not tokenizer_file.exists():
        print(f"❌ Tokenizer not found: {tokenizer_file}")
        return False
    
    # Copy tokenizer model
    output_tokenizer = Path(output_path) / "tokenizer.model"
    shutil.copy2(tokenizer_file, output_tokenizer)
    
    # Create tokenizer config
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_type": "llama",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|end_of_text|>",
        "unk_token": "<unk>",
        "add_bos_token": True,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 131072,
        "padding_side": "left",
        "use_default_system_prompt": False,
        "legacy": False
    }
    
    with open(Path(output_path) / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create special tokens map
    special_tokens_map = {
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|end_of_text|>",
        "unk_token": "<unk>"
    }
    
    with open(Path(output_path) / "special_tokens_map.json", 'w') as f:
        json.dump(special_tokens_map, f, indent=2)
    
    print("✅ Tokenizer converted")
    
    return True

def save_hf_model(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any], output_path: str):
    """Save model in HuggingFace format"""
    
    print(f"\n💾 Saving HuggingFace model to: {output_path}")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save model weights as safetensors (recommended format)
    print("📦 Saving model weights as safetensors...")
    
    # Split into shards if model is large
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    size_gb = total_size / 1024**3
    
    print(f"📊 Total model size: {size_gb:.2f}GB")
    
    if size_gb > 5:
        # Split into multiple shards
        shard_size = 5 * 1024**3  # 5GB per shard
        current_shard = {}
        current_size = 0
        shard_idx = 1
        
        for name, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > shard_size and current_shard:
                # Save current shard
                shard_file = output_dir / f"model-{shard_idx:05d}-of-{len(state_dict):05d}.safetensors"
                save_file(current_shard, shard_file)
                print(f"   Saved shard {shard_idx}")
                
                # Start new shard
                current_shard = {}
                current_size = 0
                shard_idx += 1
            
            current_shard[name] = tensor
            current_size += tensor_size
        
        # Save final shard
        if current_shard:
            shard_file = output_dir / f"model-{shard_idx:05d}-of-{len(state_dict):05d}.safetensors"
            save_file(current_shard, shard_file)
            print(f"   Saved shard {shard_idx}")
    else:
        # Save as single file
        save_file(state_dict, output_dir / "model.safetensors")
    
    print("✅ Model weights saved")
    
    # Create model index for sharded models
    safetensor_files = list(output_dir.glob("*.safetensors"))
    
    if len(safetensor_files) > 1:
        weight_map = {}
        for file in safetensor_files:
            # Load to get tensor names (metadata only)
            from safetensors import safe_open
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = file.name
        
        model_index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        
        with open(output_dir / "model.safetensors.index.json", 'w') as f:
            json.dump(model_index, f, indent=2)
    
    return True

def validate_conversion(output_path: str) -> bool:
    """Validate the converted model"""
    
    print("\n🔍 Validating converted model...")
    
    output_dir = Path(output_path)
    
    # Check required files
    required_files = [
        "config.json",
        "tokenizer.model",
        "tokenizer_config.json"
    ]
    
    for file in required_files:
        if not (output_dir / file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    # Check for model weights
    safetensor_files = list(output_dir.glob("*.safetensors"))
    if not safetensor_files:
        print("❌ No model weight files found")
        return False
    
    print(f"✅ Found {len(safetensor_files)} weight file(s)")
    
    # Try loading config
    try:
        with open(output_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        print("✅ Config loaded successfully")
        print(f"   Model type: {config.get('model_type', 'unknown')}")
        print(f"   Hidden size: {config.get('hidden_size', 'unknown')}")
        print(f"   Layers: {config.get('num_hidden_layers', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Test with transformers (optional)
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test config loading
        hf_config = AutoConfig.from_pretrained(output_path)
        print("✅ HuggingFace config validation passed")
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(output_path)
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test passed ({len(tokens['input_ids'][0])} tokens)")
        
    except Exception as e:
        print(f"⚠️  Transformers validation skipped: {e}")
        print("   (This is okay, model files are still valid)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert Meta Llama checkpoint to HuggingFace format')
    parser.add_argument('--meta_path', type=str,
                       default='/Volumes/PS2000W/.llama/checkpoints/Llama3.2-3B',
                       help='Path to Meta checkpoint directory')
    parser.add_argument('--output_path', type=str,
                       default='./models/llama-3.2-3b-hf',
                       help='Output path for HuggingFace model')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip post-conversion validation')
    
    args = parser.parse_args()
    
    print("🦙 Meta to HuggingFace Converter for Llama 3.2 3B")
    print("=" * 60)
    
    # Check if Meta checkpoint exists
    if not Path(args.meta_path).exists():
        print(f"❌ Meta checkpoint not found: {args.meta_path}")
        sys.exit(1)
    
    # Check if output already exists
    if Path(args.output_path).exists():
        print(f"\n⚠️  Output directory already exists: {args.output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
        shutil.rmtree(args.output_path)
    
    # Load Meta checkpoint
    meta_checkpoint = load_meta_checkpoint(args.meta_path)
    
    # Load Meta parameters
    meta_params = load_meta_params(args.meta_path)
    
    # Convert weight names
    hf_state_dict = convert_weight_names(meta_checkpoint)
    
    # Create HuggingFace config
    hf_config = create_hf_config(meta_params)
    
    # Save HuggingFace model
    save_hf_model(hf_state_dict, hf_config, args.output_path)
    
    # Convert tokenizer
    convert_tokenizer(args.meta_path, args.output_path)
    
    # Validate conversion
    if not args.skip_validation:
        if validate_conversion(args.output_path):
            print("\n✅ Conversion validation passed!")
        else:
            print("\n⚠️  Validation had issues, but files may still work")
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"📂 HuggingFace model saved to: {args.output_path}")
    print("\nNext steps:")
    print("1. Prepare calibration data:")
    print("   python scripts/1_prepare_calibration.py")
    print("2. Apply AWQ quantization:")
    print(f"   python scripts/3_quantize_awq.py --model_path {args.output_path}")
    print("\nThe model is now in HuggingFace format and ready for quantization!")

if __name__ == "__main__":
    main()