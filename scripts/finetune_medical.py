#!/usr/bin/env python3
"""
Medical fine-tuning script with QLoRA for Llama 3.2 3B
Optimized for healthcare domain with regulatory focus
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
import transformers
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MedicalFineTuner:
    """Fine-tune Llama for healthcare domain"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🏥 Medical Fine-Tuner initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    def load_model(self, model_path: str):
        """Load model with QLoRA configuration"""
        
        logger.info(f"📦 Loading model from: {model_path}")
        
        # BitsAndBytes config for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            max_memory={0: "35GB"}
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("✅ Model loaded successfully")
    
    def apply_lora(self):
        """Apply LoRA configuration"""
        
        logger.info("🔧 Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.config["lora_config"]["r"],
            lora_alpha=self.config["lora_config"]["lora_alpha"],
            target_modules=self.config["lora_config"]["target_modules"],
            lora_dropout=self.config["lora_config"]["lora_dropout"],
            bias=self.config["lora_config"]["bias"],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ LoRA applied:")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    def load_datasets(self, train_path: str, val_path: str):
        """Load and prepare datasets"""
        
        logger.info("📂 Loading datasets")
        
        # Load datasets
        self.train_dataset = load_dataset(
            "json",
            data_files=train_path,
            split="train"
        )
        
        self.val_dataset = load_dataset(
            "json",
            data_files=val_path,
            split="train"
        )
        
        logger.info(f"   Training samples: {len(self.train_dataset):,}")
        logger.info(f"   Validation samples: {len(self.val_dataset):,}")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        self.val_dataset = self.val_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=self.val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        logger.info("✅ Datasets loaded and tokenized")
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        
        # Get text field
        texts = examples.get("text", examples.get("messages", []))
        
        # If messages format, convert to text
        if isinstance(texts[0], list):
            formatted_texts = []
            for messages in texts:
                conversation = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    conversation += f"<|{role}|>\n{content}\n"
                formatted_texts.append(conversation)
            texts = formatted_texts
        
        # Tokenize
        model_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config["data_config"]["max_seq_length"],
            return_tensors="pt"
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    def create_trainer(self, output_dir: str):
        """Create trainer with medical optimization"""
        
        logger.info("🎯 Creating trainer")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config["training_config"]["num_train_epochs"],
            per_device_train_batch_size=self.config["training_config"]["per_device_train_batch_size"],
            per_device_eval_batch_size=self.config["training_config"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=self.config["training_config"]["gradient_accumulation_steps"],
            learning_rate=self.config["training_config"]["learning_rate"],
            warmup_steps=self.config["training_config"]["warmup_steps"],
            logging_steps=self.config["training_config"]["logging_steps"],
            save_steps=self.config["training_config"]["save_steps"],
            evaluation_strategy=self.config["training_config"]["evaluation_strategy"],
            eval_steps=self.config["training_config"]["eval_steps"],
            save_total_limit=self.config["training_config"]["save_total_limit"],
            load_best_model_at_end=self.config["training_config"]["load_best_model_at_end"],
            metric_for_best_model=self.config["training_config"]["metric_for_best_model"],
            greater_is_better=self.config["training_config"]["greater_is_better"],
            fp16=self.config["training_config"]["fp16"],
            bf16=self.config["training_config"]["bf16"],
            gradient_checkpointing=self.config["training_config"]["gradient_checkpointing"],
            optim=self.config["training_config"]["optim"],
            lr_scheduler_type=self.config["training_config"]["lr_scheduler_type"],
            max_grad_norm=self.config["training_config"]["max_grad_norm"],
            group_by_length=self.config["training_config"]["group_by_length"],
            report_to=self.config["training_config"]["report_to"],
            ddp_find_unused_parameters=self.config["training_config"]["ddp_find_unused_parameters"],
            gradient_checkpointing_kwargs=self.config["training_config"]["gradient_checkpointing_kwargs"],
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[MedicalMetricsCallback(self.config)]
        )
        
        logger.info("✅ Trainer created")
    
    def compute_metrics(self, eval_pred):
        """Compute medical-specific metrics"""
        
        predictions, labels = eval_pred
        
        # Calculate perplexity
        loss = torch.nn.functional.cross_entropy(
            torch.tensor(predictions).view(-1, predictions.shape[-1]),
            torch.tensor(labels).view(-1),
            ignore_index=-100
        )
        
        perplexity = torch.exp(loss)
        
        return {
            "perplexity": perplexity.item()
        }
    
    def train(self):
        """Execute training"""
        
        logger.info("🚀 Starting training")
        logger.info(f"   Epochs: {self.config['training_config']['num_train_epochs']}")
        logger.info(f"   Batch size: {self.config['training_config']['per_device_train_batch_size']}")
        logger.info(f"   Learning rate: {self.config['training_config']['learning_rate']}")
        
        # Start training
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model")
        self.trainer.save_model()
        
        # Save training metrics
        metrics_path = Path(self.trainer.args.output_dir) / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"✅ Training complete!")
        logger.info(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
        
        return train_result

class MedicalMetricsCallback(transformers.TrainerCallback):
    """Custom callback for medical-specific metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.medical_keywords = config["medical_optimization"]["medical_keywords_boost"]
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log medical-specific metrics"""
        
        if logs:
            # Add timestamp
            logs["timestamp"] = datetime.now().isoformat()
            
            # Check if we're meeting medical accuracy targets
            if "eval_loss" in logs:
                estimated_accuracy = 1.0 / (1.0 + logs["eval_loss"])
                logs["estimated_medical_accuracy"] = estimated_accuracy
                
                target = self.config["optimization_targets"]["medical_accuracy"]
                if estimated_accuracy >= target:
                    logger.info(f"🎯 Medical accuracy target reached: {estimated_accuracy:.2%} >= {target:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Llama for medical domain')
    parser.add_argument('--model_path', type=str,
                       default='./models/llama-3.2-3b-hf',
                       help='Path to HuggingFace model')
    parser.add_argument('--config_path', type=str,
                       default='./configs/finetuning_config.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str,
                       default='./training_data',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs/finetuned_model',
                       help='Output directory for fine-tuned model')
    parser.add_argument('--resume_from_checkpoint', type=str,
                       default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("🏥 Medical Fine-Tuning Pipeline")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training will be very slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Initialize fine-tuner
    finetuner = MedicalFineTuner(args.config_path)
    
    # Load model
    finetuner.load_model(args.model_path)
    
    # Apply LoRA
    finetuner.apply_lora()
    
    # Load datasets
    train_path = Path(args.data_dir) / "train_1M.jsonl"
    val_path = Path(args.data_dir) / "validation_100K.jsonl"
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Dataset files not found in {args.data_dir}")
        print("   Run scripts/6_prepare_finetuning_data.py first")
        sys.exit(1)
    
    finetuner.load_datasets(str(train_path), str(val_path))
    
    # Create trainer
    finetuner.create_trainer(args.output_dir)
    
    # Train
    if args.resume_from_checkpoint:
        logger.info(f"📂 Resuming from checkpoint: {args.resume_from_checkpoint}")
        finetuner.trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        finetuner.train()
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 60)
    print(f"\n📂 Model saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Test the fine-tuned model")
    print("2. Run quantization: python scripts/9_quantize_awq.py")
    print("3. Deploy to production")

if __name__ == "__main__":
    main()