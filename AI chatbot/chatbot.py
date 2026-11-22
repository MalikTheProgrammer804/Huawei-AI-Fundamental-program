"""
Professional AI Chatbot Application
Powered by Streamlit and the Official Groq API.

Features:
- Modern, responsive, dark-themed UI.
- Secure API key management and live connection validation.
- Model selection including flagship open-weights and custom model IDs.
- Role templates: General Assistant, Python Tutor, Machine Learning Expert,
  AI & Robotics Trainer, Data Analyst, and Custom System Prompt.
- Hyperparameter tuning: Temperature, Max Tokens, Top-P.
- Response streaming and multi-turn conversation history toggles.
- Session controls: New Chat, Clear History, Reset Settings.
- Export conversation to formatted Plain Text (.txt) or JSON (.json).
- Robust error handling for authentication, rate limits, and network issues.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Generator, Tuple

import streamlit as st
from dotenv import load_dotenv
from groq import (
    Groq,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
    APIConnectionError,
    APIError,
)

# Load environment variables if available
load_dotenv()

# ==============================================================================
# Page Configuration & Constants
# ==============================================================================
st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Custom Model ID",
]

ROLE_TEMPLATES: Dict[str, str] = {
    "General Assistant": (
        "You are a helpful, versatile, and knowledgeable AI assistant. "
        "Answer questions clearly, accurately, and politely. Structure long responses "
        "with headings, bullet points, and code blocks where appropriate."
    ),
    "Python Tutor": (
        "You are an expert Python programming instructor and mentor. Provide clear "
        "explanations, PEP 8-compliant code examples, debugging guidance, and beginner-friendly "
        "analogies. Always explain how and why the code works, and suggest idiomatic best practices."
    ),
    "Machine Learning Expert": (
        "You are a senior Machine Learning & Deep Learning research engineer. You specialize "
        "in model architectures, loss functions, optimization, scikit-learn, PyTorch, TensorFlow, "
        "feature engineering, and model evaluation. Provide mathematically sound and production-ready recommendations."
    ),
    "AI & Robotics Trainer": (
        "You are an AI & Robotics engineering specialist. You guide users through ROS/ROS2, "
        "kinematics, control systems (PID, MPC), sensor fusion (Kalman Filters, LiDAR/IMU), "
        "computer vision, reinforcement learning for robotics, and autonomous agent design."
    ),
    "Data Analyst": (
        "You are an expert Data Analyst and Business Intelligence specialist. You excel in SQL, "
        "pandas, statistical hypothesis testing, KPI definition, data storytelling, visualization strategies, "
        "and translating raw numbers into actionable business insights."
    ),
    "Custom Role": (
        "You are a specialized AI assistant. Follow user instructions precisely and maintain "
        "a professional, context-aware demeanor at all times."
    ),
}

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
DEFAULT_ROLE = "General Assistant"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# ==============================================================================
# Modern Dark-Themed Styling
# ==============================================================================
DARK_THEME_CSS = """
<style>
    /* Global styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top header container */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(33, 38, 45, 0.85) 100%);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .main-header-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .main-header-badge {
        font-size: 0.78rem;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        background: rgba(245, 80, 54, 0.15);
        color: #f55036;
        border: 1px solid rgba(245, 80, 54, 0.35);
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-secondary {
        background: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.35);
    }
    .badge-meta {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.35);
    }

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f0f6fc;
    }

    .sidebar-section-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 1rem;
    }

    /* Chat bubble enhancements */
    div[data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #1a2234;
        border-color: rgba(56, 139, 253, 0.3);
    }

    /* Timestamp metadata */
    .chat-meta {
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: 0.4rem;
        text-align: right;
    }

    /* Starter prompt cards */
    .quick-start-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0 2.5rem 0;
    }
    .quick-start-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.1rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }
    .quick-start-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.15);
    }
    .quick-start-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f0f6fc;
        margin-bottom: 0.35rem;
    }
    .quick-start-desc {
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.4;
    }

    /* Custom buttons */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    div.stDownloadButton > button {
        border-radius: 8px;
        font-weight: 500;
        width: 100%;
    }

    /* Subtle scrollbar */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
def init_session_state() -> None:
    """Initialize session state variables with default values."""
    defaults = {
        "messages": [],
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "validation_status": None,  # Tuple: (bool is_valid, str message)
        "role_template": DEFAULT_ROLE,
        "system_prompt": ROLE_TEMPLATES[DEFAULT_ROLE],
        "model_selection": DEFAULT_MODEL,
        "custom_model_id": "",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "stream_responses": True,
        "enable_history": True,
        "pending_prompt": None,  # Used for quick-start cards
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# Helper Functions: Groq Client & Validation
# ==============================================================================
def get_groq_client(api_key: str) -> Groq:
    """Initialize and return a Groq client instance."""
    if not api_key.strip():
        raise ValueError("Groq API Key is empty or missing.")
    return Groq(api_key=api_key.strip())


def validate_connection(api_key: str) -> Tuple[bool, str]:
    """
    Validate the Groq API key by querying the list of available models.
    Does NOT log or leak the API key.
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty. Please enter your Groq API key."
    try:
        client = get_groq_client(api_key)
        models = client.models.list()
        count = len(models.data) if hasattr(models, "data") else "active"
        return True, f"Connection verified successfully ({count} models available)."
    except AuthenticationError:
        return False, "Authentication failed: The provided Groq API key is invalid."
    except APIConnectionError:
        return False, "Network error: Unable to reach Groq API servers. Please check your internet connection."
    except RateLimitError:
        return False, "Rate limit warning: Groq API rate limit reached. Key is valid, but quota has been exceeded."
    except APIError as e:
        return False, f"Groq API error: {e.message}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_active_model() -> str:
    """Return the currently selected model ID."""
    if st.session_state.model_selection == "Custom Model ID":
        return st.session_state.custom_model_id.strip() or DEFAULT_MODEL
    return st.session_state.model_selection


def export_conversation_txt() -> str:
    """Export conversation as human-readable Plain Text."""
    lines = [
        "=" * 70,
        "GROQ AI CHATBOT - CONVERSATION EXPORT",
        f"Exported At : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model       : {get_active_model()}",
        f"Role        : {st.session_state.role_template}",
        f"Temperature : {st.session_state.temperature}",
        f"Max Tokens  : {st.session_state.max_tokens}",
        f"Top-P       : {st.session_state.top_p}",
        "=" * 70,
        f"\n[SYSTEM PROMPT]\n{st.session_state.system_prompt}\n",
        "-" * 70,
    ]
    for msg in st.session_state.messages:
        role = msg.get("role", "unknown").upper()
        ts = msg.get("timestamp", "")
        content = msg.get("content", "")
        lines.append(f"\n[{role}] ({ts})\n{content}\n")
    lines.append("=" * 70)
    return "\n".join(lines)


def export_conversation_json() -> str:
    """Export conversation as structured JSON with configuration metadata."""
    payload = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "model": get_active_model(),
            "role_template": st.session_state.role_template,
            "system_prompt": st.session_state.system_prompt,
            "parameters": {
                "temperature": st.session_state.temperature,
                "max_tokens": st.session_state.max_tokens,
                "top_p": st.session_state.top_p,
                "stream_responses": st.session_state.stream_responses,
                "enable_history": st.session_state.enable_history,
            },
            "total_messages": len(st.session_state.messages),
        },
        "messages": st.session_state.messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def reset_to_defaults() -> None:
    """Reset configuration parameters back to factory defaults."""
    st.session_state.role_template = DEFAULT_ROLE
    st.session_state.system_prompt = ROLE_TEMPLATES[DEFAULT_ROLE]
    st.session_state.model_selection = DEFAULT_MODEL
    st.session_state.custom_model_id = ""
    st.session_state.temperature = DEFAULT_TEMPERATURE
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.top_p = DEFAULT_TOP_P
    st.session_state.stream_responses = True
    st.session_state.enable_history = True
    st.session_state.validation_status = None


def clear_conversation() -> None:
    """Clear conversation messages while maintaining configuration."""
    st.session_state.messages = []


def on_role_template_change() -> None:
    """Synchronize system prompt when predefined role dropdown changes."""
    selected = st.session_state.role_template_selector
    st.session_state.role_template = selected
    st.session_state.system_prompt = ROLE_TEMPLATES.get(selected, "")


# ==============================================================================
# Sidebar Configuration Panel
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚡ Groq Configuration")
    st.caption("Manage authentication, AI models, personas, and runtime settings.")

    # 1. API Key Authentication
    st.markdown("---")
    st.markdown("#### 🔐 API Credentials")
    api_key_input = st.text_input(
        "Groq API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="gsk_...",
        help="Enter your Groq API key securely. Get one from console.groq.com/keys.",
    )
    st.session_state.api_key = api_key_input.strip()

    col_val, col_stat = st.columns([1, 1])
    with col_val:
        if st.button("🔍 Validate Key", use_container_width=True):
            is_valid, msg = validate_connection(st.session_state.api_key)
            st.session_state.validation_status = (is_valid, msg)

    with col_stat:
        if st.session_state.validation_status:
            is_valid, msg = st.session_state.validation_status
            if is_valid:
                st.success("Connected", icon="✅")
            else:
                st.error("Invalid", icon="❌")

    if st.session_state.validation_status:
        is_valid, msg = st.session_state.validation_status
        if is_valid:
            st.caption(f"✨ {msg}")
        else:
            st.caption(f"⚠️ {msg}")

    # 2. Model Selection
    st.markdown("---")
    st.markdown("#### 🧠 Model Selection")
    selected_model_option = st.selectbox(
        "Groq Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.model_selection)
        if st.session_state.model_selection in MODEL_OPTIONS
        else 0,
        help="Select a Groq-hosted open model or enter a custom model ID.",
    )
    st.session_state.model_selection = selected_model_option

    if selected_model_option == "Custom Model ID":
        custom_id = st.text_input(
            "Custom Model ID",
            value=st.session_state.custom_model_id,
            placeholder="e.g. deepseek-r1-distill-llama-70b",
            help="Type any valid Groq model name.",
        )
        st.session_state.custom_model_id = custom_id

    # 3. System Role & Prompt Persona
    st.markdown("---")
    st.markdown("#### 🎭 Persona & System Role")
    current_role_index = (
        list(ROLE_TEMPLATES.keys()).index(st.session_state.role_template)
        if st.session_state.role_template in ROLE_TEMPLATES
        else 0
    )
    st.selectbox(
        "Role Template",
        options=list(ROLE_TEMPLATES.keys()),
        index=current_role_index,
        key="role_template_selector",
        on_change=on_role_template_change,
        help="Select a persona preset or customize the prompt below.",
    )

    custom_system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=130,
        help="Instruction sent at the start of each chat session to steer behavior.",
    )
    st.session_state.system_prompt = custom_system_prompt

    # 4. Model Hyperparameters
    st.markdown("---")
    st.markdown("#### ⚙️ Hyperparameters")
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.temperature),
        step=0.05,
        help="Higher values yield more creative outputs; lower values are more deterministic.",
    )

    st.session_state.max_tokens = st.slider(
        "Max Response Tokens",
        min_value=128,
        max_value=8192,
        value=int(st.session_state.max_tokens),
        step=128,
        help="Maximum number of tokens to generate in a single response.",
    )

    st.session_state.top_p = st.slider(
        "Top-P (Nucleus Sampling)",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.top_p),
        step=0.05,
        help="Tokens comprising the top-p probability mass are considered.",
    )

    # 5. Generation & History Controls
    st.markdown("---")
    st.markdown("#### 🎛️ Runtime Options")
    st.session_state.stream_responses = st.toggle(
        "Stream Responses",
        value=st.session_state.stream_responses,
        help="Render text chunks live as they are received from Groq.",
    )

    st.session_state.enable_history = st.toggle(
        "Include Conversation History",
        value=st.session_state.enable_history,
        help="Pass past conversation turns to maintain multi-turn context.",
    )

    # 6. Session Actions
    st.markdown("---")
    st.markdown("#### 🗂️ Session Management")
    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("➕ New Chat", use_container_width=True, help="Start a clean conversation"):
            clear_conversation()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
            clear_conversation()
            st.rerun()

    if st.button("🔄 Reset Settings", use_container_width=True, help="Restore default settings"):
        reset_to_defaults()
        st.rerun()

    # 7. Export Controls
    st.markdown("---")
    st.markdown("#### 💾 Export Conversation")
    has_history = len(st.session_state.messages) > 0

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_data = export_conversation_txt() if has_history else ""
    json_data = export_conversation_json() if has_history else ""

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            label="📄 Plain TXT",
            data=txt_data,
            file_name=f"groq_chat_{timestamp_str}.txt",
            mime="text/plain",
            disabled=not has_history,
            use_container_width=True,
        )
    with col_exp2:
        st.download_button(
            label="📊 JSON Data",
            data=json_data,
            file_name=f"groq_chat_{timestamp_str}.json",
            mime="application/json",
            disabled=not has_history,
            use_container_width=True,
        )


# ==============================================================================
# Main Interface: Header & Metadata Badges
# ==============================================================================
active_model_name = get_active_model()
message_count = len(st.session_state.messages)
stream_badge = "Streaming: ON" if st.session_state.stream_responses else "Streaming: OFF"
history_badge = "Context: Full History" if st.session_state.enable_history else "Context: Single Turn"

st.markdown(
    f"""
    <div class="main-header">
        <div class="main-header-title">
            <span>⚡</span>
            <span>Groq AI Studio</span>
            <span class="main-header-badge">{active_model_name}</span>
            <span class="main-header-badge badge-secondary">{st.session_state.role_template}</span>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
            <span class="main-header-badge badge-meta">{message_count} messages</span>
            <span class="main-header-badge badge-meta">{stream_badge}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Quick Start Prompt Suggestions (When conversation is empty)
# ==============================================================================
if len(st.session_state.messages) == 0:
    st.markdown("### 👋 Welcome to Groq AI Studio")
    st.markdown(
        "Powered by Groq's high-throughput LPU Inference Engine. Select a quick starter prompt "
        "or type your custom request in the input bar below."
    )

    quick_prompts = [
        {
            "title": "🐍 Python Performance Optimization",
            "desc": "Explain concurrency vs parallelism in Python and when to use asyncio vs multiprocessing.",
            "prompt": "Explain concurrency vs parallelism in Python. When should I choose asyncio over multiprocessing? Provide clear code examples.",
        },
        {
            "title": "🤖 Robotics & Kinematics",
            "desc": "How does Forward Kinematics differ from Inverse Kinematics in a 6-DOF robotic arm?",
            "prompt": "How does Forward Kinematics differ from Inverse Kinematics in a 6-DOF robotic arm? Explain DH parameters and Jacobian matrices simply.",
        },
        {
            "title": "🧠 ML Loss Functions & Metrics",
            "desc": "Compare Focal Loss vs Cross-Entropy for severe class imbalance in computer vision.",
            "prompt": "Compare Focal Loss against standard Cross-Entropy for object detection with severe class imbalance. Explain the gamma focusing parameter.",
        },
        {
            "title": "📊 Data Analytics & SQL Window Functions",
            "desc": "Write SQL queries calculating 7-day rolling revenue and customer churn cohort analysis.",
            "prompt": "Write standard SQL queries demonstrating: 1) A 7-day moving average of revenue, and 2) Monthly retention cohort analysis using window functions.",
        },
    ]

    col1, col2 = st.columns(2)
    for idx, item in enumerate(quick_prompts):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(
                f"**{item['title']}**\n\n{item['desc']}",
                key=f"quick_prompt_{idx}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = item["prompt"]
                st.rerun()

# ==============================================================================
# Conversation History Rendering
# ==============================================================================
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", "")

    with st.chat_message(role):
        st.markdown(content)
        if timestamp:
            st.markdown(f'<div class="chat-meta">{timestamp}</div>', unsafe_allow_html=True)

# ==============================================================================
# Helper to Build Request Messages
# ==============================================================================
def build_request_messages(user_text: str) -> list:
    """Build the list of message dicts according to system prompt and history settings."""
    messages = []
    # Always insert system prompt if defined
    if st.session_state.system_prompt.strip():
        messages.append({"role": "system", "content": st.session_state.system_prompt.strip()})

    # Include conversation history if enabled
    if st.session_state.enable_history:
        for msg in st.session_state.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
    else:
        # Only include the latest user prompt
        messages.append({"role": "user", "content": user_text})

    return messages


# ==============================================================================
# Chat Input & Response Handling
# ==============================================================================
# Check for pending prompt triggered by quick-start buttons
prompt_from_input = st.chat_input("Ask a question, request code, or explore an idea...")
active_prompt = prompt_from_input or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if active_prompt:
    # 1. Ensure API Key is available
    if not st.session_state.api_key.strip():
        st.error(
            "⚠️ Groq API key is required. Please enter your API key in the sidebar panel to continue.",
            icon="🔑",
        )
        st.stop()

    now_ts = datetime.now().strftime("%I:%M %p")

    # 2. Add User Message to History & Display
    st.session_state.messages.append(
        {"role": "user", "content": active_prompt, "timestamp": now_ts}
    )
    with st.chat_message("user"):
        st.markdown(active_prompt)
        st.markdown(f'<div class="chat-meta">{now_ts}</div>', unsafe_allow_html=True)

    # 3. Prepare Payload
    request_messages = build_request_messages(active_prompt)
    target_model = get_active_model()

    # 4. Generate AI Response
    with st.chat_message("assistant"):
        try:
            client = get_groq_client(st.session_state.api_key)

            if st.session_state.stream_responses:
                # Streaming mode
                def stream_generator() -> Generator[str, None, None]:
                    response_stream = client.chat.completions.create(
                        model=target_model,
                        messages=request_messages,
                        temperature=float(st.session_state.temperature),
                        max_tokens=int(st.session_state.max_tokens),
                        top_p=float(st.session_state.top_p),
                        stream=True,
                    )
                    for chunk in response_stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta

                full_response = st.write_stream(stream_generator())
            else:
                # Non-streaming mode
                with st.spinner(f"Generating response with {target_model}..."):
                    completion = client.chat.completions.create(
                        model=target_model,
                        messages=request_messages,
                        temperature=float(st.session_state.temperature),
                        max_tokens=int(st.session_state.max_tokens),
                        top_p=float(st.session_state.top_p),
                        stream=False,
                    )
                    full_response = completion.choices[0].message.content or ""
                    st.markdown(full_response)

            assistant_ts = datetime.now().strftime("%I:%M %p")
            st.markdown(f'<div class="chat-meta">{assistant_ts}</div>', unsafe_allow_html=True)

            # Record assistant response in session state
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": assistant_ts,
                }
            )

        except AuthenticationError:
            st.error(
                "❌ **Authentication Failed**: Invalid Groq API key. Please check the API key entered in the sidebar.",
                icon="🔒",
            )
        except RateLimitError as e:
            st.error(
                f"⏳ **Rate Limit Exceeded**: Groq API rate limit or quota reached. Please wait a moment.\n\n*Details: {e.message}*",
                icon="⚠️",
            )
        except BadRequestError as e:
            st.error(
                f"⚠️ **Bad Request Error**: {e.message}\n\nPlease verify model parameters, token limits, or model ID compatibility.",
                icon="🚫",
            )
        except APIConnectionError:
            st.error(
                "🌐 **Connection Error**: Failed to communicate with Groq servers. Please check your network connection.",
                icon="📡",
            )
        except APIError as e:
            st.error(
                f"💥 **Groq API Error**: {e.message} (Status Code: {getattr(e, 'status_code', 'N/A')})",
                icon="🚨",
            )
        except Exception as e:
            st.error(
                f"❌ **Unexpected Error**: An error occurred while generating response: {str(e)}",
                icon="💥",
            )
