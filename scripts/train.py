#!/usr/bin/env python3
"""
Llama 3.2 3B Fine-tuning Script
Automatically downloads model from HuggingFace or uses local files
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def main():
    print("🚀 Starting Llama 3.2 3B Fine-tuning")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model path - can be HuggingFace ID or local path
    model_path = os.getenv("MODEL_PATH", "meta-llama/Llama-3.2-3B")
    
    print(f"Loading model from: {model_path}")
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    data_path = os.getenv("DATA_PATH", "/workspace/data/train.jsonl")
    if os.path.exists(data_path):
        dataset = load_dataset("json", data_files=data_path)
    else:
        print("Using sample dataset for testing")
        dataset = load_dataset("imdb", split="train[:100]")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/workspace/outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-4,
        fp16=True,
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model("/workspace/outputs/final_model")
    
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
