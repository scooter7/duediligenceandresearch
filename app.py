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

# Session state initialization
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key:
    st.info("Please enter your API key in the sidebar.")
    st.stop()

client = genai.Client(api_key=api_key)

# Helpers for Interactions API
def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING (INTERACTIONS API) ---
target = st.text_input("Enter Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Creating Research Plan..."):
        try:
            # Interactions API handles stateful planning
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Find owner emails/phones.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() # Forces display of the results below
        except Exception as e: st.error(f"Planning Error: {e}")

# --- DISPLAY LOGIC (OUTSIDE BUTTONS) ---
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_tasks = []
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}"):
            selected_tasks.append(f"{t['num']}. {t['text']}")
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent (autonomous) searching the web (2-5 mins)..."):
            try:
                # Triggers the Deep Research Agent
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder contact info for:\n" + "\n".join(selected_tasks),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                # Polling loop for background tasks
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun()
            except Exception as e: st.error(f"Research Error: {e}")

if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Data"): st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Synthesizing final report..."):
            try:
                # ADK Pipeline uses specialized agents
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                               instruction=f"Analyze data from: {st.session_state.research_text}", 
                               tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                   instruction="Write memo with Founder Contact Table.", 
                                   tools=[generate_html_report])
                st.session_state.final_memo = SequentialAgent(name="Pipe", sub_agents=[fin, partner]).run(input="Finalize")
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.header("🏆 Strategic Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download", st.session_state.final_memo, "report.md")
