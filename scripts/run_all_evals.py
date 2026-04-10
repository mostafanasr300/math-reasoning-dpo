import subprocess
import os
import sys

# Set encoding to utf-8 for Windows console
os.environ["PYTHONIOENCODING"] = "utf-8"

def run_eval(model_path, output_file, script="scripts/evaluate_smollm.py", base_model="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
    print(f"\n" + "="*50)
    print(f"Starting evaluation: {model_path}")
    print("="*50)
    
    # Check if model exists
    if "/" not in model_path and not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist. Skipping.")
        return

    # Delete existing log if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Use the python executable from the virtual environment
    venv_python = os.path.abspath(os.path.join("super_math", "Scripts", "python.exe"))
    
    cmd = [
        venv_python, 
        script, 
        "--model_path", model_path, 
        "--base_model", base_model,
        "--num_examples", "100" # Testing on 100 for balance
    ]
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding="utf-8"
            )
            process.wait()
        print(f"✓ Success! Results saved to {output_file}")
    except Exception as e:
        print(f"✗ Failed during {model_path} evaluation: {e}")

def main():
    print("Starting Comprehensive Evaluation Suite (SmolLM2)...")
    print("Device: CPU")
    
    # 1. Base Model
    run_eval("HuggingFaceTB/SmolLM2-1.7B-Instruct", "eval_base_smollm.txt")
    
    # 2. SFT Model
    run_eval("output/sft_model_smollm", "eval_sft_smollm.txt")
    
    # 3. DPO Model
    run_eval("output/dpo_model_smollm", "eval_dpo_smollm.txt")
    
    print("\n" + "="*50)
    print("ALL EVALUATIONS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
