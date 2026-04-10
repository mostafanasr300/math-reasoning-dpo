"""
Flask API for Math Reasoning Models (Base, SFT, DPO)
Wraps three Qwen2.5-0.5B model variants and exposes them via REST endpoints.
"""

import os
import sys
import time
import torch
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HUGGINGFACE_SFT_MODEL = "mostafa-nasr14/Qwen-Math-SFT"
HUGGINGFACE_DPO_ADAPTER = "mostafa-nasr14/Qwen-Math-DPO-Adapter"
MODEL_CONFIGS = {
    "base": {
        "name": "Base (Qwen2.5-0.5B-Instruct)",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": None,
        "description": "Original pre-trained Qwen 2.5 0.5B Instruct model without any fine-tuning.",
        "accuracy": 44.0,
        "eval_samples": 1319,
        "eval_correct": 580,
    },
    "sft": {
        "name": "SFT (Supervised Fine-Tuned)",
        "base_model": HUGGINGFACE_SFT_MODEL,     #os.path.join(BASE_DIR, "output", "sft_model_merged"),
        "adapter_path": None,
        "description": "Qwen 2.5 0.5B fine-tuned with supervised learning on math reasoning data.",
        "accuracy": 55.6,
        "eval_samples": 1319,
        "eval_correct": 733,
    },
    "dpo": {
        "name": "DPO (Direct Preference Optimization)",
        "base_model": HUGGINGFACE_SFT_MODEL, #os.path.join(BASE_DIR, "output", "sft_model_merged"),
        "adapter_path": os.path.join(BASE_DIR, "output", "dpo_model_final"),
        "description": "SFT model further aligned with DPO to prefer high-quality math reasoning.",
        "accuracy": 56.0,
        "eval_samples": 1319,
        "eval_correct": 739,
    },
}

# Using a Few-Shot CoT system prompt to maximize mathematical logic and accuracy
SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning assistant. You must carefully think step-by-step before calculating. "
    "Break down complex problems into manageable logic.Focus on the math problem steps and make sure to compute each step correctly. "
    "Be aware of the problem order and the steps you need to take to solve it. "
    "Always conclude your final mathematical answer with '#### [answer]'.\n\n"
    "Here are two examples of the expected reasoning structure:\n"
    "---\n"
    "Example 1:\n"
    "Problem: In a class of 30 students: 18 students like Math. 16 students like Science. 5 students like neither Math nor Science. How many students like both Math and Science?\n"
    "Reasoning: First, calculate the total number of students who like at least one subject. Exclude the 'Neither' group: 30 - 5 = 25 students like at least one of the two subjects. "
    "Sum the individual groups: 18 + 16 = 34. Since we only have 25 students who like subjects, the 'extra' count must be the students we counted twice (the ones who like both). "
    "34 - 25 = 9.\n"
    "#### 9\n"
    "---\n"
    "Example 2:\n"
    "Problem: You have two coins in a hat. One is a fair coin (Heads/Tails), and the other is a double-headed coin (Heads/Heads). You pull one coin out at random and flip it; it lands on Heads. What is the probability that the coin you flipped is the double-headed coin?\n"
    "Reasoning: This requires Bayes' Theorem. We calculate the likelihood of seeing 'Heads' from each coin. "
    "Probability of picking Fair coin (P(F)) = 1/2. Probability of picking Double-headed coin (P(D)) = 1/2. "
    "P(Heads | F) = 1/2. P(Heads | D) = 1. "
    "Total probability of Heads: (1/2 * 1/2) + (1/2 * 1) = 1/4 + 1/2 = 3/4. "
    "Probability it's the Double-headed coin: (P(Heads | D) * P(D)) / P(Total Heads) = (1/2) / (3/4) = 2/3.\n"
    "#### 2/3\n"
    "---"
)

# ---------------------------------------------------------------------------
# Global model storage
# ---------------------------------------------------------------------------
models = {}          # model_key -> (model, tokenizer)
loading_status = {}  # model_key -> "loading" | "ready" | "error: ..."
load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

generation_lock = threading.Lock()

def load_all_models():
    """Load models with dynamic memory sharing to prevent RAM exhaustion."""
    # 1. Load the pristine 'base' model normally
    loading_status["base"] = "loading"
    try:
        print("  [LOADING]  Base (Qwen2.5-0.5B-Instruct) ...", flush=True)
        t0 = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIGS["base"]["base_model"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        base_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["base"]["base_model"])
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.eval()
        models["base"] = (base_model, base_tokenizer)
        loading_status["base"] = "ready"
        print(f"  [READY]    Base loaded in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        loading_status["base"] = f"error: {exc}"
        print(f"  [ERROR]    Failed to load Base: {exc}", flush=True)

    # 2. Load the SFT base model ONCE, and wrap it with the DPO adapter ONCE
    loading_status["sft"] = "loading"
    loading_status["dpo"] = "loading"
    try:
        print("  [LOADING]  Shared SFT/DPO Architecture (Dynamic Adapter Swapping) ...", flush=True)
        t0 = time.time()
        shared_base = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIGS["sft"]["base_model"],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        shared_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS["sft"]["base_model"])
        if shared_tokenizer.pad_token is None:
            shared_tokenizer.pad_token = shared_tokenizer.eos_token
            
        shared_model = PeftModel.from_pretrained(shared_base, MODEL_CONFIGS["dpo"]["adapter_path"])
        shared_model.eval()

        models["sft"] = (shared_model, shared_tokenizer)
        models["dpo"] = (shared_model, shared_tokenizer)
        
        loading_status["sft"] = "ready"
        loading_status["dpo"] = "ready"
        print(f"  [READY]    SFT + DPO dynamically sharing RAM. Loaded in {time.time() - t0:.1f}s", flush=True)
    except Exception as exc:
        loading_status["sft"] = f"error: {exc}"
        loading_status["dpo"] = f"error: {exc}"
        print(f"  [ERROR]    Failed to load Shared SFT/DPO: {exc}", flush=True)


def generate_response(model_key: str, user_message: str, history: list = None, max_tokens: int = 512) -> dict:
    """Generate a response from a specific model."""
    if model_key not in models:
        status = loading_status.get(model_key, "unknown")
        return {"error": f"Model '{model_key}' not available (status: {status})"}

    model, tokenizer = models[model_key]

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    if history:
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()

    with generation_lock:
        # Dynamically switch the PEFT adapter state based on the requested model
        if model_key == "sft" and hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()
        elif model_key == "dpo" and hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    elapsed = time.time() - t0
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant response
    response = full_text.split("assistant")[-1].strip()

    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

    return {
        "model": model_key,
        "model_name": MODEL_CONFIGS[model_key]["name"],
        "response": response,
        "generation_time_s": round(elapsed, 2),
        "tokens_generated": int(num_tokens),
    }


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow Streamlit to call the API


@app.route("/", methods=["GET"])
def index():
    """Welcome page to verify server is alive."""
    return jsonify({
        "message": "Math Reasoning Model API is ONLINE",
        "status": "awake",
        "loading_progress": loading_status,
        "endpoints": ["/", "/health", "/models", "/chat", "/chat/compare"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "server": "alive",
        "models_status": loading_status
    })


@app.route("/models", methods=["GET"])
def list_models():
    """Return info about available models."""
    info = []
    for key, cfg in MODEL_CONFIGS.items():
        info.append({
            "key": key,
            "name": cfg["name"],
            "description": cfg["description"],
            "accuracy_gsm8k": cfg["accuracy"],
            "eval_samples": cfg["eval_samples"],
            "eval_correct": cfg["eval_correct"],
            "status": loading_status.get(key, "unknown"),
        })
    return jsonify(info)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Generate a response from a specific model.
    Body: { "model": "dpo"|"sft"|"base", "message": "...", "max_tokens": 512 }
    """
    data = request.get_json(force=True)
    model_key = data.get("model", "dpo")
    message = data.get("message", "")
    history = data.get("history", [])
    max_tokens = data.get("max_tokens", 512)

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400
    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model '{model_key}'. Choose from: {list(MODEL_CONFIGS.keys())}"}), 400

    result = generate_response(model_key, message, history, max_tokens)
    if "error" in result:
        return jsonify(result), 503

    return jsonify(result)


@app.route("/chat/compare", methods=["POST"])
def chat_compare():
    """
    Generate responses from ALL loaded models for side-by-side comparison.
    Body: { "message": "...", "max_tokens": 512 }
    """
    data = request.get_json(force=True)
    message = data.get("message", "")
    history = data.get("history", [])
    max_tokens = data.get("max_tokens", 512)

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    results = {}
    for key in MODEL_CONFIGS:
        results[key] = generate_response(key, message, history, max_tokens)

    return jsonify({"question": message, "responses": results})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  MATH REASONING MODEL API")
    print("  Loading 3 models: Base / SFT / DPO")
    print("=" * 60)
    load_all_models()
    print("\n  API ready at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
