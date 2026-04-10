import subprocess
import os
import sys

# Set encoding to utf-8 for Windows console
os.environ["PYTHONIOENCODING"] = "utf-8"

def run_eval(model_path, output_file, num_examples=100, base_model="Qwen/Qwen2.5-0.5B-Instruct", script="scripts/evaluate_final.py", start_idx=0, start_correct=0, start_total=0):
    print(f"\n" + "="*50)
    print(f"Starting evaluation: {model_path} (resuming from {start_idx})")
    print("="*50)
    
    # Check if model exists (skip check for HF models if no slash in local directory)
    if "/" in model_path and not model_path.startswith("Qwen") and not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist. Skipping.")
        return

    # Use the python executable from the virtual environment
    venv_python = os.path.abspath(os.path.join("super_math", "Scripts", "python.exe"))
    
    cmd = [
        venv_python, 
        script, 
        "--model_path", model_path, 
        "--base_model", base_model,
        "--num_examples", str(num_examples),
        "--start_idx", str(start_idx),
        "--start_correct", str(start_correct),
        "--start_total", str(start_total)
    ]
    
    try:
        # Open in append mode 'a' instead of 'w' to resume
        with open(output_file, "a", encoding="utf-8") as f:
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
    print("Starting Comprehensive Evaluation Suite (Qwen2.5-0.5B-Instruct)...")
    print("Device: CPU")
    
    # 1. Base Model (Resuming from 969)
    run_eval("Qwen/Qwen2.5-0.5B-Instruct", "eval_base_full.txt", num_examples=2000, start_idx=969, start_correct=428, start_total=969)
    
    # 2. SFT Model (Starting Fresh)
    run_eval("output/sft_model_merged", "eval_sft_full.txt", num_examples=2000)
    
    # 3. DPO Model (Adapter over SFT Model, Starting Fresh)
    run_eval("output/dpo_model_final", "eval_dpo_full.txt", base_model="output/sft_model_merged", num_examples=2000)
    
    print("\n" + "="*50)
    print("ALL EVALUATIONS COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
