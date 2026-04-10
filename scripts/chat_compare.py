import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def load_model(base_name, adapter_path=None):
    print(f"Loading {adapter_path if adapter_path else base_name}...")
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    else:
        model = base
        tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt):
    # SmolLM2 Prompt Style
    chat_prompt = f"<|im_start|>system\nYou are a helpful assistant specialized in math reasoning.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract response
    return text.split("assistant")[-1].strip()

def main():
    base_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    sft_path = "output/sft_model_smollm"
    dpo_path = "output/dpo_model_smollm"
    
    print("\n" + "="*50)
    print("MATH MODEL COMPARISON CHAT (SmolLM2)")
    print("="*50)
    
    # Load models
    try:
        m_base, t_base = load_model(base_name)
        m_sft, t_sft = None, None
        m_dpo, t_dpo = None, None
        
        if os.path.exists(sft_path):
            m_sft, t_sft = load_model(base_name, sft_path)
        if os.path.exists(dpo_path):
            m_dpo, t_dpo = load_model(base_name, dpo_path)
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    while True:
        print("\n" + "-"*50)
        query = input("Enter a math question (or 'exit'): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
            
        print("\nGenerating responses...\n")
        
        # Base
        print(f"[BASE MODEL]:\n{generate(m_base, t_base, query)}")
        print("-" * 20)
        
        # SFT
        if m_sft:
            print(f"[SFT MODEL]:\n{generate(m_sft, t_sft, query)}")
            print("-" * 20)
        else:
            print(f"[SFT MODEL]: Not found at {sft_path}")
            
        # DPO
        if m_dpo:
            print(f"[DPO MODEL]:\n{generate(m_dpo, t_dpo, query)}")
        else:
            print(f"[DPO MODEL]: Not found at {dpo_path}")

if __name__ == "__main__":
    main()
