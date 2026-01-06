import streamlit as st
import time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_financial_chart

# --- 1. SYSTEM SETUP & AUTH ---
st.set_page_config(page_title="Investment Intel 2026", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentApp")

# Initialize Session State variables
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

# Authentication layer
if not st.session_state.auth:
    st.subheader("🔐 Secure Analysis Portal")
    pw = st.text_input("Application Password", type="password")
    if st.button("Unlock"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key:
    st.info("Please enter your API key in the sidebar to begin.")
    st.stop()

client = genai.Client(api_key=api_key)

# Helper functions for the Interactions API
def get_text(outputs):
    if not outputs: return ""
    return "\n".join(o.text for o in outputs if hasattr(o, 'text') and o.text)

def parse_tasks(text):
    return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING (Logic separated from Display) ---
target = st.text_input("Target Analysis", placeholder="e.g., Pet cremation in Phoenix, AZ")

if st.button("📋 Step 1: Generate Strategic Plan") and target:
    with st.spinner("Creating 2026 Research Plan..."):
        try:
            # Interactions API creates a stateful planning session
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Include owner contact details.", 
                store=True
            )
            # Store results in session state so they persist across reruns
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() 
        except Exception as e:
            st.error(f"Planning Error: {e}")

# DISPLAY: Rendered OUTSIDE of the button block to ensure persistence
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_tasks = []
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}"):
            selected_tasks.append(f"{t['num']}. {t['text']}")
    
    # --- STEP 2: DEEP RESEARCH ---
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Agent searching the web (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder emails/phones for:\n" + "\n".join(selected_tasks),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                
                # Polling for completion
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun() 
            except Exception as e:
                st.error(f"Research Error: {e}")

# --- STEP 3: FINAL ANALYSIS (Using corrected .run() method) ---
if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Data"): 
        st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Synthesizing final report..."):
            try:
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                               instruction=f"Build model from: {st.session_state.research_text}", 
                               tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                   instruction="Write memo with founder contact table.", 
                                   tools=[generate_html_report])
                
                # ADK SequentialAgent uses .run()
                pipeline = SequentialAgent(name="AnalysisPipeline", sub_agents=[fin, partner])
                result = pipeline.run(input="Complete Investment Memo")
                
                # Capture result
                st.session_state.final_memo = result if isinstance(result, str) else getattr(result, 'output', str(result))
                st.rerun()
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")

# Final Report Display
if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Strategic Investment Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
