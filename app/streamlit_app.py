"""
Streamlit Chat UI for Math Reasoning Models
Connects to the Flask API to query Base, SFT, and DPO models.
"""

import streamlit as st
import requests
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "http://localhost:5000"

MODEL_META = {
    "base": {
        "name": "Base",
        "full_name": "Qwen2.5-0.5B-Instruct",
        "icon": "\U0001F9E0",
        "color": "#6366f1",
        "accuracy": 44.0,
        "eval_correct": 580,
        "eval_total": 1319,
        "tag": "Baseline",
        "desc": "Original pre-trained model. No fine-tuning applied.",
    },
    "sft": {
        "name": "SFT",
        "full_name": "Supervised Fine-Tuned",
        "icon": "\U0001F4DA",
        "color": "#f59e0b",
        "accuracy": 55.6,
        "eval_correct": 733,
        "eval_total": 1319,
        "tag": "Fine-Tuned",
        "desc": "Fine-tuned with supervised learning on math reasoning data.",
    },
    "dpo": {
        "name": "DPO",
        "full_name": "Direct Preference Optimization",
        "icon": "\U0001F3AF",
        "color": "#10b981",
        "accuracy": 56.0,
        "eval_correct": 739,
        "eval_total": 1319,
        "tag": "Best \u2b50",
        "desc": "SFT model aligned via DPO for preferred answers.",
    },
}

SAMPLE_PROBLEMS = [
    "If a train travels 120 miles in 2 hours, what is its speed in miles per hour?",
    "A store sells apples for $1.50 each. If you buy 4 apples and pay with a $10 bill, how much change do you get?",
    "There are 24 students in a class. If 3/8 of them are girls, how many boys are there?",
    "A bat and a ball cost $1.10$ in total. The bat costs $1.00$ more than the ball. How much does the ball cost?",
    "If you save $15 per week, how many weeks will it take to save $180?",
    "A car's gas tank holds 16 gallons. If the car gets 32 miles per gallon, how far can it travel on a full tank?",
    """Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. 
    She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?""",
]


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Math Reasoning Arena",
    page_icon="\U0001F9EE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #10b981, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Model cards */
    .model-card {
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        margin-bottom: 0.75rem;
    }
    .model-card:hover {
        border-color: rgba(255,255,255,0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    .model-card .card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .model-card .card-header .icon {
        font-size: 1.5rem;
    }
    .model-card .card-header .name {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .model-card .card-header .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: auto;
    }
    .model-card .accuracy-bar-bg {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 0.4rem 0;
    }
    .model-card .accuracy-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease;
    }
    .model-card .accuracy-text {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
    }
    .model-card .accuracy-label {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .model-card .description {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    /* Chat bubbles */
    .chat-msg-user {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }
    .chat-msg-model {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .chat-msg-model .model-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Compare columns */
    .compare-col-header {
        text-align: center;
        padding: 0.6rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Sidebar styles */
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1rem 0 0.5rem 0;
    }

    /* Metric highlight */
    .metric-highlight {
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .metric-highlight .value {
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-highlight .label {
        font-size: 0.75rem;
        color: #94a3b8;
    }

    /* Sidebar / Button Text Wrapping */
    div.stButton > button {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-align: left !important;
        height: auto !important;
        padding: 0.8rem !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }

    /* Hide streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health():
    """Check if Flask API is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json()
    except Exception:
        return None


def call_chat(model_key: str, message: str, history: list, max_tokens: int = 820):
    """Call the /chat endpoint."""
    try:
        r = requests.post(
            f"{API_URL}/chat",
            json={"model": model_key, "message": message, "history": history, "max_tokens": max_tokens},
            timeout=200,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the Flask server running?"}
    except Exception as e:
        return {"error": str(e)}


def call_compare(message: str, history: list, max_tokens: int = 820):
    """Call the /chat/compare endpoint."""
    try:
        r = requests.post(
            f"{API_URL}/chat/compare",
            json={"message": message, "history": history, "max_tokens": max_tokens},
            timeout=300,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the Flask server running?"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "model", ...}
if "mode" not in st.session_state:
    st.session_state.mode = "Single Model"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "dpo"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">API Status</div>', unsafe_allow_html=True)
    health = check_api_health()
    ready_count = 0
    total_count = len(MODEL_META)
    
    if health:
        model_statuses = health.get("models_status", {})
        ready_count = sum(1 for s in model_statuses.values() if s == "ready")


        if ready_count == total_count:
            st.success(f"All {ready_count} models loaded")
        elif ready_count > 0:
            st.warning(f"{ready_count}/{total_count} models loaded")
        else:
            st.info("Models are loading...")

        for mkey, mstatus in model_statuses.items():
            meta = MODEL_META.get(mkey, {})
            icon = meta.get("icon", "")
            name = meta.get("name", mkey)
            if mstatus == "ready":
                st.markdown(f"<span style='color:#10b981'>●</span> {icon} **{name}** — Ready", unsafe_allow_html=True)
            elif mstatus == "loading":
                st.markdown(f"<span style='color:#f59e0b'>●</span> {icon} **{name}** — Loading...", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#ef4444'>●</span> {icon} **{name}** — Error", unsafe_allow_html=True)
    else:
        st.error("API offline — start Flask server first")
        st.code("python api/flask_api.py", language="bash")

    st.divider()

    st.markdown('<div class="sidebar-section-title">Chat Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Select mode",
        ["Single Model", "Compare All"],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    if mode == "Single Model":
        st.markdown('<div class="sidebar-section-title">Select Model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Model",
            options=["dpo", "sft", "base"],
            format_func=lambda k: f"{MODEL_META[k]['icon']} {MODEL_META[k]['name']} ({MODEL_META[k]['accuracy']}%)",
            label_visibility="collapsed",
        )
        st.session_state.selected_model = model_choice

    st.divider()
    st.markdown('<div class="sidebar-section-title">Max Tokens</div>', unsafe_allow_html=True)
    max_tokens = st.slider("Max tokens", 64, 1024, 512, step=64, label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sidebar-section-title">Sample Problems</div>', unsafe_allow_html=True)
    for i, problem in enumerate(SAMPLE_PROBLEMS):
        if st.button(f"\U0001F4DD {problem}", key=f"sample_{i}", use_container_width=True):
            st.session_state.sample_input = problem

    st.divider()
    if st.button("\U0001F5D1\uFE0F  Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>\U0001F9EE Math Reasoning Arena</h1>
    <p>Compare Base \u2192 SFT \u2192 DPO models on math problems \u2022 GSM8K Benchmark</p>
</div>
""", unsafe_allow_html=True)

# Model overview cards
col1, col2, col3 = st.columns(3)
for col, mkey in zip([col1, col2, col3], ["base", "sft", "dpo"]):
    meta = MODEL_META[mkey]
    with col:
        tag_bg = meta["color"] + "22"
        tag_color = meta["color"]
        bar_width = meta["accuracy"]

        st.markdown(f"""
        <div class="model-card">
            <div class="card-header">
                <span class="icon">{meta['icon']}</span>
                <span class="name">{meta['name']}</span>
                <span class="tag" style="background:{tag_bg}; color:{tag_color}">{meta['tag']}</span>
            </div>
            <div class="accuracy-text" style="color:{meta['color']}">{meta['accuracy']}%</div>
            <div class="accuracy-label">GSM8K Accuracy ({meta['eval_correct']}/{meta['eval_total']})</div>
            <div class="accuracy-bar-bg">
                <div class="accuracy-bar-fill" style="width:{bar_width}%; background: linear-gradient(90deg, {meta['color']}88, {meta['color']})"></div>
            </div>
            <div class="description">{meta['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

# Check for sample problem injection
prefill = ""
if "sample_input" in st.session_state:
    prefill = st.session_state.pop("sample_input")

user_input = st.chat_input("Type a math problem...", key="chat_input")

# Use sample problem if clicked
if prefill and not user_input:
    user_input = prefill

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })

    if st.session_state.mode == "Single Model":
        model_key = st.session_state.selected_model
        meta = MODEL_META[model_key]

        with st.spinner(f"{meta['icon']} {meta['name']} is thinking..."):
            history_copy = st.session_state.chat_history[:-1] # Remove just-appended user message from context buffer
            result = call_chat(model_key, user_input, history_copy, max_tokens)

        if "error" in result:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"\u26A0\uFE0F {result['error']}",
                "model": model_key,
                "time": 0,
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "model": model_key,
                "time": result.get("generation_time_s", 0),
                "tokens": result.get("tokens_generated", 0),
            })

    else:  # Compare All
        with st.spinner("Generating from all three models..."):
            history_copy = st.session_state.chat_history[:-1]
            result = call_compare(user_input, history_copy, max_tokens)

        if "error" in result:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"\u26A0\uFE0F {result['error']}",
                "model": "compare",
                "time": 0,
            })
        else:
            st.session_state.chat_history.append({
                "role": "compare",
                "responses": result["responses"],
            })

    st.rerun()

# ---------------------------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-msg-user">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    elif msg["role"] == "assistant":
        model_key = msg.get("model", "dpo")
        meta = MODEL_META.get(model_key, MODEL_META["dpo"])
        gen_time = msg.get("time", 0)
        tokens = msg.get("tokens", 0)

        badge_html = f'<span class="model-badge" style="background:{meta["color"]}22; color:{meta["color"]}">{meta["icon"]} {meta["name"]}</span>'
        time_html = f'<span style="float:right; font-size:0.7rem; color:#64748b">{gen_time}s \u2022 {tokens} tokens</span>' if gen_time else ""

        st.markdown(
            f'<div class="chat-msg-model">{badge_html}{time_html}<br/>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )

    elif msg["role"] == "compare":
        responses = msg.get("responses", {})
        cols = st.columns(3)
        for col, mkey in zip(cols, ["base", "sft", "dpo"]):
            meta = MODEL_META[mkey]
            resp = responses.get(mkey, {})
            with col:
                st.markdown(
                    f'<div class="compare-col-header" style="background:{meta["color"]}22; color:{meta["color"]}">'
                    f'{meta["icon"]} {meta["name"]}</div>',
                    unsafe_allow_html=True,
                )
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    gen_time = resp.get("generation_time_s", 0)
                    tokens = resp.get("tokens_generated", 0)
                    st.markdown(
                        f'<div class="chat-msg-model">'
                        f'<span style="font-size:0.7rem; color:#64748b">{gen_time}s \u2022 {tokens} tok</span><br/>'
                        f'{resp.get("response", "")}</div>',
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Background Auto-Polling logic
# ---------------------------------------------------------------------------
# If the API is entirely offline/booting (health is None)
# OR if the API is online but still loading models (ready_count < total_count)
# Wait briefly and trigger a page rerun, without blocking the initial UI draw.
if (health is None) or (ready_count < total_count):
    time.sleep(2.5)
    st.rerun()
