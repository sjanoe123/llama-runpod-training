#!/usr/bin/env python3
"""
Prepare fine-tuning dataset by merging CLA-510 and Enhanced Llama Corpus
Intelligently samples 1M high-quality examples for optimal training
"""

import json
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import hashlib
from collections import Counter
import sys

class FineTuningDataPreparator:
    """Prepare and merge training datasets for fine-tuning"""
    
    def __init__(self, target_samples: int = 1000000):
        self.target_samples = target_samples
        self.all_samples = []
        self.category_distribution = Counter()
        self.quality_threshold = 0.7
        
    def load_cla510_dataset(self, path: str, max_samples: int = 5000000) -> int:
        """Load CLA-510 consolidated training data"""
        
        print(f"\n📂 Loading CLA-510 dataset from: {path}")
        
        loaded = 0
        file_path = Path(path)
        
        if not file_path.exists():
            print(f"❌ File not found: {path}")
            return 0
        
        # Get file size for progress tracking
        file_size = file_path.stat().st_size / (1024**3)  # GB
        print(f"   File size: {file_size:.2f}GB")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading CLA-510", unit=" samples"):
                if loaded >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Format for fine-tuning
                    sample = {
                        "instruction": data.get("instruction", ""),
                        "input": data.get("input", ""),
                        "output": data.get("output", ""),
                        "category": data.get("category", "medical"),
                        "source": "CLA-510",
                        "quality_score": self._calculate_quality_score(data)
                    }
                    
                    if sample["quality_score"] >= self.quality_threshold:
                        self.all_samples.append(sample)
                        self.category_distribution[sample["category"]] += 1
                        loaded += 1
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        print(f"✅ Loaded {loaded:,} samples from CLA-510")
        return loaded
    
    def load_enhanced_corpus(self, directory: str) -> int:
        """Load Enhanced Llama Training Corpus"""
        
        print(f"\n📂 Loading Enhanced Llama Corpus from: {directory}")
        
        loaded = 0
        corpus_dir = Path(directory)
        
        if not corpus_dir.exists():
            print(f"❌ Directory not found: {directory}")
            return 0
        
        # Find all enhanced corpus files
        corpus_files = sorted(corpus_dir.glob("enhanced_llama_training_chunk_*.json"))
        print(f"   Found {len(corpus_files)} corpus files")
        
        for file_path in tqdm(corpus_files, desc="Loading corpus files"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                for item in data:
                    sample = {
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                        "category": item.get("example_type", "medical"),
                        "source": "Enhanced_Corpus",
                        "regulatory_focus": item.get("regulatory_focus", None),
                        "quality_score": self._calculate_quality_score(item)
                    }
                    
                    if sample["quality_score"] >= self.quality_threshold:
                        self.all_samples.append(sample)
                        self.category_distribution[sample["category"]] += 1
                        loaded += 1
                        
            except Exception as e:
                print(f"⚠️  Error loading {file_path.name}: {e}")
                continue
        
        print(f"✅ Loaded {loaded:,} samples from Enhanced Corpus")
        return loaded
    
    def _calculate_quality_score(self, data: Dict) -> float:
        """Calculate quality score for a training sample"""
        
        score = 0.0
        
        # Check instruction quality
        instruction = data.get("instruction", "")
        if len(instruction) > 20:
            score += 0.2
        if any(keyword in instruction.lower() for keyword in 
               ["medical", "clinical", "diagnosis", "treatment", "hipaa", "billing"]):
            score += 0.2
        
        # Check input quality
        input_text = data.get("input", "")
        if len(input_text) > 50:
            score += 0.2
        if "ICD" in input_text or "CPT" in input_text or "diagnosis" in input_text.lower():
            score += 0.1
        
        # Check output quality
        output = data.get("output", "")
        if len(output) > 30:
            score += 0.2
        if not output.startswith("I'm sorry") and not output.startswith("I cannot"):
            score += 0.1
        
        return min(score, 1.0)
    
    def intelligent_sampling(self) -> List[Dict]:
        """Intelligently sample training data for balanced distribution"""
        
        print(f"\n🎯 Intelligent sampling of {self.target_samples:,} examples")
        print(f"   Total available: {len(self.all_samples):,}")
        
        if len(self.all_samples) <= self.target_samples:
            print("   Using all available samples")
            return self.all_samples
        
        # Sort by quality score
        self.all_samples.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Take top quality samples
        sampled = []
        
        # First, ensure diversity by taking from each category
        categories = list(self.category_distribution.keys())
        samples_per_category = self.target_samples // len(categories)
        
        for category in categories:
            category_samples = [s for s in self.all_samples if s["category"] == category]
            sampled.extend(category_samples[:samples_per_category])
        
        # Fill remaining with highest quality samples
        remaining_needed = self.target_samples - len(sampled)
        already_sampled_ids = {id(s) for s in sampled}
        
        for sample in self.all_samples:
            if id(sample) not in already_sampled_ids:
                sampled.append(sample)
                if len(sampled) >= self.target_samples:
                    break
        
        # Shuffle to mix categories
        random.shuffle(sampled)
        
        print(f"✅ Sampled {len(sampled):,} high-quality examples")
        
        return sampled[:self.target_samples]
    
    def create_train_validation_split(self, samples: List[Dict], 
                                     validation_ratio: float = 0.1) -> tuple:
        """Split data into training and validation sets"""
        
        print(f"\n📊 Creating train/validation split ({int((1-validation_ratio)*100)}/{int(validation_ratio*100)})")
        
        # Shuffle before splitting
        random.shuffle(samples)
        
        split_idx = int(len(samples) * (1 - validation_ratio))
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        print(f"   Training samples: {len(train_samples):,}")
        print(f"   Validation samples: {len(val_samples):,}")
        
        return train_samples, val_samples
    
    def format_for_training(self, sample: Dict) -> Dict:
        """Format sample for HuggingFace training"""
        
        # Create conversational format
        formatted = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a healthcare AI assistant specializing in medical coding, billing compliance, and clinical documentation."
                },
                {
                    "role": "user",
                    "content": f"{sample['instruction']}\n\n{sample['input']}"
                },
                {
                    "role": "assistant",
                    "content": sample['output']
                }
            ],
            "metadata": {
                "category": sample["category"],
                "source": sample["source"],
                "quality_score": sample["quality_score"]
            }
        }
        
        # Add text format for compatibility
        formatted["text"] = f"<|system|>\nYou are a healthcare AI assistant specializing in medical coding, billing compliance, and clinical documentation.\n<|user|>\n{sample['instruction']}\n\n{sample['input']}\n<|assistant|>\n{sample['output']}"
        
        return formatted
    
    def save_datasets(self, train_samples: List[Dict], val_samples: List[Dict], 
                      output_dir: str):
        """Save training and validation datasets"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training set
        train_file = output_path / "train_1M.jsonl"
        print(f"\n💾 Saving training set to: {train_file}")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(train_samples, desc="Writing training data"):
                formatted = self.format_for_training(sample)
                f.write(json.dumps(formatted) + '\n')
        
        # Save validation set
        val_file = output_path / "validation_100K.jsonl"
        print(f"💾 Saving validation set to: {val_file}")
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(val_samples, desc="Writing validation data"):
                formatted = self.format_for_training(sample)
                f.write(json.dumps(formatted) + '\n')
        
        # Save metadata
        metadata = {
            "total_samples": len(train_samples) + len(val_samples),
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "categories": dict(self.category_distribution),
            "sources": {
                "CLA-510": sum(1 for s in train_samples + val_samples if s["source"] == "CLA-510"),
                "Enhanced_Corpus": sum(1 for s in train_samples + val_samples if s["source"] == "Enhanced_Corpus")
            },
            "average_quality_score": sum(s["quality_score"] for s in train_samples + val_samples) / len(train_samples + val_samples)
        }
        
        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata to: {metadata_file}")
        
        # Calculate file sizes
        train_size = train_file.stat().st_size / (1024**3)
        val_size = val_file.stat().st_size / (1024**3)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Training file size: {train_size:.2f}GB")
        print(f"   Validation file size: {val_size:.2f}GB")
        print(f"   Total size: {train_size + val_size:.2f}GB")

def main():
    parser = argparse.ArgumentParser(description='Prepare fine-tuning dataset')
    parser.add_argument('--cla510_path', type=str,
                       default='/Volumes/PS2000W/CLA-510-Fine-Tuning-Curation/05-Finalized-Dataset/splits/train/consolidated_train.jsonl',
                       help='Path to CLA-510 training data')
    parser.add_argument('--corpus_dir', type=str,
                       default='/Volumes/PS2000W/healthcare-ai/datasets/llama-training-corpus',
                       help='Directory containing enhanced corpus')
    parser.add_argument('--output_dir', type=str,
                       default='./training_data',
                       help='Output directory for processed datasets')
    parser.add_argument('--target_samples', type=int,
                       default=1000000,
                       help='Target number of training samples')
    parser.add_argument('--max_cla510', type=int,
                       default=5000000,
                       help='Maximum samples to load from CLA-510')
    parser.add_argument('--validation_ratio', type=float,
                       default=0.1,
                       help='Ratio of data for validation')
    parser.add_argument('--seed', type=int,
                       default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("🏥 Healthcare Fine-Tuning Data Preparation")
    print("=" * 60)
    
    # Initialize preparator
    preparator = FineTuningDataPreparator(target_samples=args.target_samples)
    
    # Load datasets
    cla510_count = preparator.load_cla510_dataset(args.cla510_path, args.max_cla510)
    corpus_count = preparator.load_enhanced_corpus(args.corpus_dir)
    
    total_loaded = cla510_count + corpus_count
    
    if total_loaded == 0:
        print("❌ No data loaded! Check paths and try again.")
        sys.exit(1)
    
    print(f"\n📊 Total samples loaded: {total_loaded:,}")
    print(f"   Category distribution:")
    for category, count in preparator.category_distribution.most_common(10):
        print(f"      {category}: {count:,}")
    
    # Intelligent sampling
    sampled = preparator.intelligent_sampling()
    
    # Create train/validation split
    train_samples, val_samples = preparator.create_train_validation_split(
        sampled, args.validation_ratio
    )
    
    # Save datasets
    preparator.save_datasets(train_samples, val_samples, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\n📂 Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review dataset metadata")
    print("2. Upload to RunPod for training")
    print("3. Start fine-tuning with:")
    print(f"   python scripts/8_finetune_medical.py --data_dir {args.output_dir}")

if __name__ == "__main__":
    main()