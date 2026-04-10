from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_from_disk
import torch
import os

print("="*50)
print("SFT TRAINING - Qwen2.5-1.5B (The Smart Upgrade)")
print("="*50)

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
MAX_SEQ_LENGTH = 384  # Optimized for CPU RAM
DATASET_PATH = os.path.join("datasets", "sft_dataset")
OUTPUT_DIR = os.path.join("output", "sft_model_qwen")

# Detect device
DEVICE = "cpu" 
print(f"\nUsing device: {DEVICE}")

# --- Model Loading ---
print(f"\nLoading model: {MODEL_NAME}")
# Trust remote code is needed for some Qwen layers
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Add LoRA Adapters ---
print("\nAdding LoRA adapters...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    # Targeting all linear layers for maximum "brain upgrade"
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
print("\nTrainable parameters:")
model.print_trainable_parameters()

# --- Data Loading ---
print(f"\nLoading dataset from: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)

# Use larger subset as requested
SUBSET_SIZE = 2000
dataset = dataset.select(range(min(SUBSET_SIZE, len(dataset))))
print(f"Using {len(dataset)} examples for training")

# Format for Qwen (ChatML style)
def formatting_prompts_func(examples):
    instructions = examples["query"]
    outputs = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # Qwen-style instruct template
        text = f"<|im_start|>system\nYou are a helpful assistant specialized in math reasoning.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

# --- Training ---
print("\nSetting up training...")
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # Higher accumulation since 1.5B is heavier
    warmup_steps=100,
    max_steps=500, # Lower steps initially to see progress, can increase later
    learning_rate=1e-4, # Slightly lower for larger models
    fp16=False,
    logging_steps=5,
    optim="adamw_torch",
    output_dir=OUTPUT_DIR,
    save_steps=250,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

print("\n" + "="*50)
print("Starting SFT training (Qwen)...")
print("="*50 + "\n")

trainer.train()

# --- Save Model ---
print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Qwen SFT Model saved to {OUTPUT_DIR}")
