from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

SFT_MODEL_PATH = os.path.join("output", "sft_model_final")
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MERGED_MODEL_PATH = os.path.join("output", "sft_model_merged")

print(f"Loading base model: {BASE_MODEL_NAME}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"Loading adapter from: {SFT_MODEL_PATH}")
model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)

print("Merging adapters...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {MERGED_MODEL_PATH}")
merged_model.save_pretrained(MERGED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print("Done!")
