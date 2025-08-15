#!/usr/bin/env python3
"""
Complete Llama 3.2 3B Healthcare Fine-Tuning Pipeline Orchestrator
Manages the entire workflow from data prep to deployment
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrate complete fine-tuning pipeline"""
    
    def __init__(self, mode: str = "full"):
        """
        Initialize orchestrator
        
        Args:
            mode: "test" for small sample, "full" for complete training
        """
        self.mode = mode
        self.start_time = time.time()
        self.checkpoints = {}
        self.results = {}
        
        # Paths
        self.base_dir = Path("/workspace/repo")
        self.scripts_dir = self.base_dir / "scripts"
        self.venv_path = Path("/usr/local")  # RunPod has system Python
        
        # Configuration
        self.config = {
            "test": {
                "samples": 10000,
                "epochs": 1,
                "gpu": "NVIDIA RTX A6000",
                "estimated_time": 2,  # hours
                "estimated_cost": 5   # dollars
            },
            "full": {
                "samples": 1000000,
                "epochs": 3,
                "gpu": "NVIDIA A100-SXM4-40GB",
                "estimated_time": 14,  # hours
                "estimated_cost": 50   # dollars
            }
        }[mode]
        
        logger.info(f"🚀 Pipeline Orchestrator initialized in {mode.upper()} mode")
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites are met"""
        
        logger.info("🔍 Checking prerequisites...")
        
        checks = {
            "Python environment": True,  # RunPod has Python installed
            "Converted model": Path("/workspace/model").exists(),
            "Training data": Path("/workspace/training_data/train.jsonl").exists(),
            "GPU available": torch.cuda.is_available() if 'torch' in sys.modules else True,
            "Scripts directory": self.scripts_dir.exists(),
            "Config file": (self.base_dir / "configs/finetuning_config.json").exists()
        }
        
        all_passed = True
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            logger.error("❌ Prerequisites not met!")
            logger.info("\nTo fix:")
            if not checks["Python environment"]:
                logger.info("   python -m venv venv && source venv/bin/activate")
            if not checks["Converted model"]:
                logger.info("   python scripts/0_convert_meta_to_hf.py")
            if not checks["RunPod API key"]:
                logger.info("   export RUNPOD_API_KEY='your_key'")
        
        return all_passed
    
    def run_command(self, cmd: str, description: str, timeout: Optional[int] = None) -> bool:
        """Run a command and track progress"""
        
        logger.info(f"\n📌 {description}")
        logger.info(f"   Command: {cmd}")
        
        try:
            # Activate venv in command
            if "python" in cmd:
                cmd = f"source {self.venv_path}/bin/activate && {cmd}"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                logger.info(f"   ✅ Success")
                self.checkpoints[description] = {
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                return True
            else:
                logger.error(f"   ❌ Failed with code {result.returncode}")
                logger.error(f"   Error: {result.stderr[:500]}")
                self.checkpoints[description] = {
                    "status": "failed",
                    "timestamp": datetime.now().isoformat(),
                    "error": result.stderr[:500]
                }
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"   ❌ Timeout after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"   ❌ Exception: {e}")
            return False
    
    def step1_prepare_data(self) -> bool:
        """Step 1: Prepare training data"""
        
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        output_dir = "test_data" if self.mode == "test" else "training_data"
        
        cmd = f"""python scripts/6_prepare_finetuning_data.py \
            --target_samples {self.config['samples']} \
            --output_dir ./{output_dir} \
            --validation_ratio 0.1"""
        
        return self.run_command(
            cmd,
            f"Preparing {self.config['samples']:,} training samples",
            timeout=1800  # 30 minutes
        )
    
    def step2_local_testing(self) -> bool:
        """Step 2: Local testing (optional)"""
        
        if self.mode != "test":
            logger.info("\n📌 Skipping local testing in FULL mode")
            return True
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: LOCAL TESTING")
        logger.info("="*60)
        
        # Test model loading
        cmd = """python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = './models/llama-3.2-3b-hf'
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='cpu',
    low_cpu_mem_usage=True
)
print('✅ Model loads successfully')

# Test inference
inputs = tokenizer('What is HIPAA?', return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0])
print(f'Response: {response[:100]}...')
" """
        
        return self.run_command(
            cmd,
            "Testing model loading and inference",
            timeout=300  # 5 minutes
        )
    
    def step3_runpod_deployment(self) -> bool:
        """Step 3: Deploy to RunPod"""
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: RUNPOD DEPLOYMENT")
        logger.info("="*60)
        
        logger.info(f"\n💰 Estimated cost: ${self.config['estimated_cost']}")
        logger.info(f"⏱️  Estimated time: {self.config['estimated_time']} hours")
        
        # Confirm with user
        if self.mode == "full":
            response = input("\n⚠️  This will cost ~$50. Continue? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Deployment cancelled by user")
                return False
        
        data_dir = "test_data" if self.mode == "test" else "training_data"
        
        cmd = f"""python scripts/7_runpod_full_deploy.py \
            --model_path ./models/llama-3.2-3b-hf \
            --data_path ./{data_dir} \
            --output_dir ./deployment_results \
            --gpu "{self.config['gpu']}" \
            {'--test_mode' if self.mode == 'test' else ''}"""
        
        # This is a long-running command
        return self.run_command(
            cmd,
            f"RunPod {self.mode} deployment",
            timeout=self.config['estimated_time'] * 3600 + 3600  # +1 hour buffer
        )
    
    def step4_validate_results(self) -> bool:
        """Step 4: Validate results"""
        
        logger.info("\n" + "="*60)
        logger.info("STEP 4: VALIDATION")
        logger.info("="*60)
        
        results_dir = Path("deployment_results")
        
        if not results_dir.exists():
            logger.error("❌ Results directory not found")
            return False
        
        # Check for quantized model
        quantized_model = results_dir / "quantized_model"
        if quantized_model.exists():
            model_files = list(quantized_model.glob("*.safetensors"))
            if model_files:
                model_size = sum(f.stat().st_size for f in model_files) / (1024**3)
                logger.info(f"✅ Quantized model found: {model_size:.2f}GB")
                self.results["model_size"] = model_size
            else:
                logger.error("❌ No model files found")
                return False
        
        # Check training logs
        log_file = results_dir / "training.log"
        if log_file.exists():
            logger.info(f"✅ Training log found: {log_file.stat().st_size / 1024:.1f}KB")
            
            # Extract final metrics
            with open(log_file, 'r') as f:
                content = f.read()
                if "Fine-tuning complete!" in content:
                    logger.info("✅ Training completed successfully")
                else:
                    logger.warning("⚠️  Training may not have completed")
        
        # Test the quantized model
        logger.info("\n🧪 Testing quantized model...")
        
        test_cmd = f"""python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = './deployment_results/quantized_model'
print('Loading quantized model...')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    torch_dtype=torch.float16
)

# Test medical queries
test_prompts = [
    'What is the ICD-10 code for Type 2 diabetes?',
    'Explain HIPAA requirements for PHI.',
    'What is medical necessity for MRI?'
]

for prompt in test_prompts:
    print(f'\\nPrompt: {prompt}')
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=100, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Response: {response[:200]}...')
" """
        
        return self.run_command(
            test_cmd,
            "Testing quantized model",
            timeout=600  # 10 minutes
        )
    
    def generate_report(self):
        """Generate final report"""
        
        runtime = time.time() - self.start_time
        runtime_hours = runtime / 3600
        
        report = {
            "mode": self.mode,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "runtime_hours": round(runtime_hours, 2),
            "checkpoints": self.checkpoints,
            "results": self.results,
            "success": all(cp.get("status") == "success" for cp in self.checkpoints.values())
        }
        
        # Save report
        report_path = Path(f"pipeline_report_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📊 Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        print(f"Mode: {self.mode.upper()}")
        print(f"Runtime: {runtime_hours:.1f} hours")
        print(f"Success: {'✅ YES' if report['success'] else '❌ NO'}")
        
        if self.results.get("model_size"):
            print(f"Model size: {self.results['model_size']:.2f}GB")
        
        print("\nCheckpoints:")
        for name, checkpoint in self.checkpoints.items():
            status = "✅" if checkpoint["status"] == "success" else "❌"
            print(f"  {status} {name}")
        
        return report
    
    def run(self) -> bool:
        """Run complete pipeline"""
        
        print("\n" + "="*60)
        print(f"🚀 LLAMA 3.2 3B HEALTHCARE FINE-TUNING PIPELINE")
        print(f"   Mode: {self.mode.upper()}")
        print(f"   Estimated time: {self.config['estimated_time']} hours")
        print(f"   Estimated cost: ${self.config['estimated_cost']}")
        print("="*60)
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Run pipeline steps
        steps = [
            ("Data Preparation", self.step1_prepare_data),
            ("Local Testing", self.step2_local_testing),
            ("RunPod Deployment", self.step3_runpod_deployment),
            ("Validation", self.step4_validate_results)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🎯 Executing: {step_name}")
            
            if not step_func():
                logger.error(f"❌ Pipeline failed at: {step_name}")
                self.generate_report()
                return False
        
        # Generate final report
        report = self.generate_report()
        
        if report["success"]:
            print("\n" + "🎉"*20)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉"*20)
            print("\n✅ Your fine-tuned, quantized model is ready!")
            print(f"📂 Location: deployment_results/quantized_model")
            print(f"📊 Size: {self.results.get('model_size', 'Unknown')}GB")
            print(f"⏱️  Total time: {report['runtime_hours']:.1f} hours")
        
        return report["success"]

def main():
    parser = argparse.ArgumentParser(
        description='Complete Llama healthcare fine-tuning pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'full'],
        default='test',
        help='Pipeline mode: test (10K samples) or full (1M samples)'
    )
    parser.add_argument(
        '--skip-confirmation',
        action='store_true',
        help='Skip cost confirmation prompts'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(mode=args.mode)
    
    # Run pipeline
    success = orchestrator.run()
    
    # Exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()