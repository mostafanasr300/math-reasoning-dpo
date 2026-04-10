from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse
from datasets import load_dataset
import os

print("="*50)
print("QWEN EVALUATION - GSM8K Benchmark")
print("="*50)

def eval_gsm8k(model, tokenizer, device, num_examples=3, start_idx=0, start_correct=0, start_total=0):
    print(f"\nLoading GSM8K test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    model.eval()
    model.to(device)
    
    correct = start_correct
    total = start_total
    
    num_examples = min(num_examples, len(dataset))
    print(f"\n--- Evaluating on {num_examples} examples (resuming from {start_idx}) ---\n")
    
    for i in range(start_idx, num_examples):
        data = dataset[i]
        question = data["question"]
        answer = data["answer"]
        
        # Extract ground truth number
        ground_truth = answer.split("####")[-1].strip()
        
        # Format input for Qwen
        prompt = f"<|im_start|>system\nYou are a helpful assistant specialized in math reasoning.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate
        print(f"[{i+1}/{num_examples}] Generating...", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response
        response = prediction.split("assistant")[-1].strip()
        
        # Check if answer is correct
        if ground_truth in response:
            correct += 1
            print(" CORRECT", flush=True)
        else:
            print(" INCORRECT", flush=True)
        
        total += 1
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*50}")
    print(f"RESULTS: {correct}/{total} correct ({accuracy:.1f}%)")
    print(f"{'='*50}\n")
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or model name")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--start_correct", type=int, default=0)
    parser.add_argument("--start_total", type=int, default=0)
    args = parser.parse_args()
    
    device = "cpu"
    print(f"Device: {device}")
    
    # Check if local adapter or base model
    if os.path.exists(args.model_path):
        print(f"Loading local model/adapter from: {args.model_path}")
        # If it's a directory with adapter_model.safetensors, use PeftModel
        if os.path.exists(os.path.join(args.model_path, "adapter_model.safetensors")):
            print(f"Detected adapter, loading base model: {args.base_model}")
            base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            model = PeftModel.from_pretrained(base, args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        print(f"Loading model from HuggingFace: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_gsm8k(model, tokenizer, device, args.num_examples, args.start_idx, args.start_correct, args.start_total)

if __name__ == "__main__":
    main()
