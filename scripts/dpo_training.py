import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("="*50)
print("DPO ALIGNMENT - Fine-Tuning (LoRA)")
print("="*50)

# --- Configuration ---
BASE_MODEL_PATH = os.path.join("..", "output", "sft_model_merged")
MAX_SEQ_LENGTH = 256  # Reduced to save memory on CPU
DATASET_PATH = os.path.join("..", "datasets", "dpo_dataset")
OUTPUT_DIR = os.path.join("..", "output", "dpo_model_full")


# --- Model & Tokenizer ---
print(f"Loading merged SFT model from: {BASE_MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
if tokenizer.unk_token is None:
    tokenizer.unk_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Enable gradient checkpointing to drastically reduce memory usage during full fine-tuning
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# --- Data Preparation ---
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)

def format_dpo_fn(example):
    prompt_text = f"<|im_start|>system\nYou are a helpful assistant specialized in math reasoning.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    chosen_text = f"{example['chosen_response']}<|im_end|>"
    rejected_text = f"{example['rejected_response']}<|im_end|>"
    
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

print("Formatting dataset for DPO...")
formatted_ds = dataset.map(format_dpo_fn, remove_columns=dataset.column_names)

print(f"Dataset ready. Total samples: {len(formatted_ds)}")

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Increased batch size for TPU 
    gradient_accumulation_steps=2,  # Scaled down to match memory
    learning_rate=5e-6,
    num_train_epochs=1,
    # --- Critical for Visibility ---
    logging_strategy="steps",    
    logging_steps=10,            # Adjusted logging freq since it's faster
    log_level="info",            
    report_to="none",            
    # ------------------------------
    save_strategy="steps",
    save_steps=50,               # Save a checkpoint every 50 steps
    save_total_limit=2,          
    gradient_checkpointing=False,  
    use_cpu=False,               # Let accelerate detect the Kaggle TPU!
    bf16=True,                   # TPUs utilize bfloat16 incredibly well
    fp16=False,
    max_length=256,
    beta=0.1,
    remove_unused_columns=False,
)

# 1. Define the LoRA Configuration
peft_config = LoraConfig(
    r=8,                    # Rank: higher = more parameters, 16 is a good balance
    lora_alpha=16,           # Scaling factor (usually 2x Rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Targets the attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2. Wrap your existing model
model = get_peft_model(model, peft_config)

# 3. Print the new parameter count to verify the 10% goal
model.print_trainable_parameters()


dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_ds,
    tokenizer=tokenizer,
)

last_checkpoint = None
if os.path.exists(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        print(f"Found checkpoint: {last_checkpoint}")

print("Starting Full DPO training...")
# Start training!
if last_checkpoint is not None:
    print(f"Resuming training from {last_checkpoint}...")
dpo_trainer.train(resume_from_checkpoint=last_checkpoint)


# --- Save ---
print("Saving DPO model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Full DPO Model saved to {OUTPUT_DIR}")
