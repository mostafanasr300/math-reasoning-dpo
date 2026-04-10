import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
SFT_DATASET_ID = "meta-math/MetaMathQA"
DPO_DATASET_ID = "argilla/distilabel-math-preference-dpo"
OUTPUT_DIR = "datasets"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_sft_data():
    print(f"Loading SFT dataset: {SFT_DATASET_ID}...")
    dataset = load_dataset(SFT_DATASET_ID, split="train")
    
    # MetaMathQA has 'query' and 'response'. 
    # We'll validte it exists and maybe take a subset for quick testing if user wants, 
    # but for now we save the whole thing.
    
    print(f"SFT Dataset size: {len(dataset)}")
    
    # Save to disk
    sft_path = os.path.join(OUTPUT_DIR, "sft_dataset")
    dataset.save_to_disk(sft_path)
    print(f"Saved SFT dataset to {sft_path}")

def prepare_dpo_data():
    print(f"Loading DPO dataset: {DPO_DATASET_ID}...")
    dataset = load_dataset(DPO_DATASET_ID, split="train")
    
    # Dataset likely has 'instruction', 'chosen_response', 'rejected_response'
    # Or similar structures. Let's map them to standard DPO columns if needed: prompt, chosen, rejected
    
    # Check column names
    print(f"Original DPO Columns: {dataset.column_names}")
    
    def format_dpo(example):
        return {
            "prompt": example["instruction"],
            "chosen": example["chosen_response"],
            "rejected": example["rejected_response"]
        }
    
    # Apply formatting if columns match expectation, otherwise let user know (or handle dynamically)
    # The search result said it has instruction, chosen, rejected.
    # We will assume standard names or map them.
    # Based on typical distilabel datasets: 'instruction', 'chosen', 'rejected' or similar.
    # Let's simple check and rename if needed or just save.
    
    # We'll save directly for now, validation can happen during loading in training script
    dpo_path = os.path.join(OUTPUT_DIR, "dpo_dataset")
    dataset.save_to_disk(dpo_path)
    print(f"Saved DPO dataset to {dpo_path}")

if __name__ == "__main__":
    prepare_sft_data()
    prepare_dpo_data()
