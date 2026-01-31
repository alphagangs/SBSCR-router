import streamlit as st
import time
import pandas as pd
import sys
import os
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbscr.routers.sbscr import SBSCRRouter

# Page Config
st.set_page_config(
    page_title="SBSCR Inference Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Glass-Box" aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-box {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f3f4f6;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .stage-success {
        color: #10b981;
        font-weight: bold;
    }
    .stage-skip {
        color: #6b7280;
    }
    .router-log {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        background-color: #111827;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Router (Cached)
@st.cache_resource(show_spinner="Initializing Neural Router...")
def get_router():
    return SBSCRRouter()

try:
    router = get_router()
except Exception as e:
    st.error(f"Failed to initialize router: {e}")
    st.stop()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug_info" not in st.session_state:
    st.session_state.last_debug_info = None

# --- Layout ---
col_left, col_right = st.columns([1.2, 0.8])

# --- LEFT PANE: Chat Interface ---
with col_left:
    st.title("💬 SBSCR Inference Chat")
    st.caption("Sub-Millisecond Semantic Routing Engine")
    
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask something complex (e.g., 'Write a binary search in Python')..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Router
        with st.spinner("Routing..."):
            debug_info = router.route_with_debug(prompt)
            st.session_state.last_debug_info = debug_info
            
            # Simulate bot response (mocking the content generation for the demo)
            model_name = debug_info["final_model"]
            response_text = f"✅ **Routed to {model_name}**\n\n*(In a real app, {model_name} would generate the answer here. This demo focuses on the routing decision.)*"
            
        # Add bot message
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
        # Force rerun to update right pane immediately
        st.rerun()

# --- RIGHT PANE: The "Glass Box" (Debug) ---
with col_right:
    st.markdown("### 🔍 Observability Panel")
    
    info = st.session_state.last_debug_info
    
    if info:
        # 1. Top Level Metrics
        cols = st.columns(3)
        
        # Calculate Costs (Mock)
        # GPT-4: $30/1M, Llama-70B: $0.9/1M, Phi-3: $0.1/1M
        gpt4_cost = 30.0
        model_name = info["final_model"]
        actual_cost = 30.0 if "gpt-4" in model_name else (0.9 if "llama" in model_name or "claude" in model_name else 0.1)
        savings_per_1m = gpt4_cost - actual_cost
        
        with cols[0]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Latency (ms)</div>
                <div class="metric-value" style="color: #60a5fa">{info['total_latency_ms']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            score = info['stages'].get('stage3_complexity', {}).get('score', 0)
            color = "#ef4444" if score > 0.7 else ("#fbbf24" if score > 0.3 else "#10b981")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Complexity</div>
                <div class="metric-value" style="color: {color}">{score:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[2]:
             st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Savings / 1M</div>
                <div class="metric-value" style="color: #34d399">${savings_per_1m:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # 2. Pipeline Visualization
        st.markdown("#### ⚡ Routing Pipeline Trace")
        
        stages = info['stages']
        
        # Stage 0: Trivial
        s0 = stages.get("stage0_trivial", {})
        is_trivial = s0.get("is_trivial", False)
        st.markdown(f"""
        <div class="router-log">
            <span class="{'stage-success' if is_trivial else 'stage-skip'}">● Stage 0: Trivial Filter</span>
            <br>&nbsp;&nbsp;Is Trivial: {is_trivial}
            <br>&nbsp;&nbsp;Latency: {s0.get('latency_ms', 0):.4f}ms
        </div>
        """, unsafe_allow_html=True)
        
        if not is_trivial:
            # Stage 1: Fast Path
            s1 = stages.get("stage1_fast_path", {})
            fast_conf = s1.get("confidence", 0)
            st.markdown(f"""
            <div class="router-log" style="margin-top: 5px">
                <span class="{'stage-success' if fast_conf > 0.5 else 'stage-skip'}">● Stage 1: Keyword Fast Path</span>
                <br>&nbsp;&nbsp;Intent: {s1.get('intent', 'none')}
                <br>&nbsp;&nbsp;Confidence: {fast_conf:.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Stage 2: LSH
            s2 = stages.get("stage2_lsh", {})
            st.markdown(f"""
            <div class="router-log" style="margin-top: 5px">
                <span class="stage-success">● Stage 2: LSH Semantic Bucket</span>
                <br>&nbsp;&nbsp;Bucket ID: {s2.get('bucket_id', 'N/A')}
                <br>&nbsp;&nbsp;Intent: {s2.get('intent', 'unknown')}
                <br>&nbsp;&nbsp;Confidence: {s2.get('confidence', 0):.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Stage 3: Complexity
            s3 = stages.get("stage3_complexity", {})
            st.markdown(f"""
            <div class="router-log" style="margin-top: 5px">
                <span class="stage-success">● Stage 3: Heuristic Complexity</span>
                <br>&nbsp;&nbsp;Score: {s3.get('score', 0):.3f}
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 3. Decision Visual
        st.markdown("#### 🎯 Final Decision")
        st.info(f"**Target Model**: `{info['final_model']}`")
        if info.get('fallback_chain'):
             st.caption(f"Fallback Chain: {' -> '.join(info['fallback_chain'])}")

    else:
        st.info("Waiting for input...")
        st.markdown("""
        **Try asking:**
        - "Hi" (Trivial)
        - "Solve 2x+5=0" (Math)
        - "Write a Python script" (Coding)
        - "Write a poem" (Creative)
        """)
