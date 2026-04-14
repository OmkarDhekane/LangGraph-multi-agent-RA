"""
Streamlit Observability Dashboard
----------------------------------
Full UI for the multi-agent research assistant.
Shows per-agent confidence scores, latency, hallucination flags,
agent logs, and the final answer in real time.
"""

import streamlit as st
import time
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Research Assistant — Multi-Agent RAG",
    page_icon="🔬",
    layout="wide"
)

# ── Styles ──────────────────────────────────────────────────────────────────
style_path = Path(__file__).with_name("styles.css")
if style_path.exists():
    st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔬 Adaptive Research Assistant")
st.caption("Multi-Agent LangGraph Pipeline · Supervisor → Researcher → Analyst → Critic → Reviser")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
```
START
  ↓
Supervisor      ← query planner
  ↓
Researcher      ← web search
  ↓
Analyst         ← draft synthesis
  ↓
Critic          ← hallucination check
  ↓ (if REVISE)
Reviser         ← rewrite draft
  ↓ (loop ≤2x)
END             ← final answer
```
""")

    st.divider()
    st.markdown("### 📦 Stack")
    st.markdown("LangGraph · LangChain · Gemini 1.5 Flash · Streamlit")

# ── Main Query Interface ───────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Research Query",
        placeholder="e.g. What are the latest advances in LLM reasoning capabilities?",
        label_visibility="collapsed"
    )
with col2:
    run_btn = st.button("🚀 Run Research", use_container_width=True, type="primary")

# Example queries
st.markdown("**Try:** "
    "`What is LangGraph?` · "
    "`How does RAG work?` · "
    "`What is the CAP theorem?` · "
    "`Explain transformer attention mechanisms`"
)

st.divider()

# ── Run Pipeline ────────────────────────────────────────────────────────────
if run_btn and query:
    if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "your_gemini_api_key_here":
        st.error("⚠️ Please set GOOGLE_API_KEY in your environment to run the pipeline.")
        st.stop()

    # Live progress display
    progress_bar = st.progress(0, text="Initializing pipeline...")
    status_placeholder = st.empty()
    
    pipeline_stages = [
        (0.15, "🧠 Supervisor analyzing query..."),
        (0.35, "🔍 Researcher fetching web results..."),
        (0.55, "✍️ Analyst synthesizing draft answer..."),
        (0.75, "🔎 Critic validating for hallucinations..."),
        (0.90, "✅ Finalizing answer..."),
    ]

    # Show animated progress while pipeline runs
    result_container = st.empty()
    
    try:
        from pipeline import run_pipeline

        # Run pipeline with progress simulation
        start_time = time.time()
        
        with st.spinner("Running multi-agent pipeline..."):
            result = run_pipeline(query)
        
        total_time = round(time.time() - start_time, 2)
        progress_bar.progress(1.0, text=f"✅ Complete in {total_time}s")

        # ── Results Layout ────────────────────────────────────────────────────
        
        # Final Answer
        st.markdown("## 📋 Final Answer")
        answer = result.get("final_answer") or result.get("draft_answer", "No answer generated.")
        st.markdown(answer)
        
        st.divider()

        # ── Observability Dashboard ────────────────────────────────────────────
        st.markdown("## 📊 Pipeline Observability")
        
        scores = result.get("confidence_scores", {})
        hal_flags = result.get("hallucination_flags", [])
        iterations = result.get("iteration_count", 0)
        
        # Top-level metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        
        with m1:
            st.metric("Total Time", f"{total_time}s")
        with m2:
            avg_conf = sum(v for k, v in scores.items() 
                          if isinstance(v, float) and not k.endswith("_s") and k != "analyst_citations") / max(1, len([k for k in scores if isinstance(scores[k], float) and not k.endswith("_s") and k != "analyst_citations"]))
            st.metric("Avg Confidence", f"{avg_conf:.0%}")
        with m3:
            st.metric("Hallucinations Flagged", len(hal_flags))
        with m4:
            st.metric("Critic Iterations", iterations)
        with m5:
            verdict = "✅ Approved" if not result.get("needs_revision") else "⚠️ Revised"
            st.metric("Final Verdict", verdict)

        st.divider()

        # Per-Agent Confidence Scores
        st.markdown("### 🤖 Per-Agent Metrics")
        
        agent_cols = st.columns(4)
        agents = [
            ("supervisor", "🧠 Supervisor", "#7c3aed"),
            ("researcher", "🔍 Researcher", "#0369a1"),
            ("analyst", "✍️ Analyst", "#065f46"),
            ("critic", "🔎 Critic", "#92400e"),
        ]
        
        for col, (key, label, color) in zip(agent_cols, agents):
            with col:
                conf = scores.get(key, 0)
                lat = scores.get(f"{key}_latency_s", 0)
                conf_pct = int(conf * 100)
                
                if conf_pct >= 80:
                    badge_class = "badge-green"
                elif conf_pct >= 60:
                    badge_class = "badge-yellow"
                else:
                    badge_class = "badge-red"
                
                st.markdown(f"""
<div class="agent-card" style="border-left-color: {color}">
    <strong>{label}</strong><br>
    <span class="badge {badge_class}">{conf_pct}% confidence</span><br>
    <small>⏱ {lat}s latency</small>
</div>
""", unsafe_allow_html=True)

        st.divider()

        # Hallucination flags
        st.markdown("### 🚨 Hallucination Flags")
        if hal_flags:
            for flag in hal_flags:
                st.warning(f"⚠️ {flag}")
        else:
            st.success("✅ No hallucinations detected — all claims supported by sources")

        # Search results used
        st.markdown("### 🌐 Sources Used")
        search_results = result.get("search_results", [])
        if search_results:
            for i, r in enumerate(search_results):
                with st.expander(f"[Source {i+1}] {r.get('title', 'Unknown')} — Score: {r.get('score', 0):.2f}"):
                    st.markdown(f"**URL:** {r.get('url', 'N/A')}")
                    st.markdown(r.get("content", ""))
        else:
            st.info("No sources retrieved.")

        # Agent logs
        st.markdown("### 📝 Agent Execution Logs")
        logs = result.get("agent_logs", [])
        for log in logs:
            agent = log.split("]")[0].replace("[", "").strip() if "]" in log else "SYSTEM"
            color_map = {
                "SUPERVISOR": "#7c3aed",
                "RESEARCHER": "#0369a1",
                "ANALYST": "#065f46",
                "CRITIC": "#92400e",
                "REVISER": "#be185d"
            }
            color = color_map.get(agent, "#4b5563")
            st.markdown(
                f'<div class="log-entry" style="border-left-color: {color}">{log}</div>',
                unsafe_allow_html=True
            )

        # Critic feedback
        if result.get("critique"):
            with st.expander("🔎 Full Critic Report"):
                st.code(result["critique"], language="markdown")

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)

elif run_btn and not query:
    st.warning("Please enter a research query.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangGraph · LangChain · Gemini 1.5 Flash · Streamlit")
