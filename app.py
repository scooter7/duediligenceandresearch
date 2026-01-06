import streamlit as st
import time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- 1. SYSTEM SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentApp")
st.set_page_config(page_title="Investment Intel 2026", layout="wide")

# Initialize Sessions (Must be at the top)
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

# --- 2. AUTHENTICATION ---
if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Clear All Data"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment Intelligence")
if not api_key:
    st.info("Enter API key in sidebar.")
    st.stop()

client = genai.Client(api_key=api_key)

# Helpers
def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING ---
target = st.text_input("Research Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Creating Research Plan..."):
        try:
            plan_prompt = f"Create a 6-step research plan for: {target}. Focus on owner names and contact emails."
            i = client.interactions.create(model="gemini-3-flash-preview", input=plan_prompt, store=True)
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() # LOCK RESULTS INTO UI
        except Exception as e: st.error(f"Error: {e}")

# --- STEP 2: DISPLAY TASKS & RUN RESEARCH ---
# This block runs OUTSIDE the button to ensure it persists
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_tasks = []
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}"):
            selected_tasks.append(f"{t['num']}. {t['text']}")
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent browsing the web (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder emails/phones for:\n" + "\n".join(selected_tasks),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                # Polling loop
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun() # LOCK RESULTS INTO UI
            except Exception as e: st.error(f"Error: {e}")

# --- STEP 3: FINAL ANALYSIS ---
if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Data"): st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Multi-Agent Analysis"):
        with st.spinner("Synthesizing final report..."):
            try:
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                               instruction=f"Extract data from: {st.session_state.research_text}", 
                               tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                   instruction="Create memo with Founder Contact Table.", 
                                   tools=[generate_html_report])
                st.session_state.final_memo = SequentialAgent(name="Pipe", sub_agents=[fin, partner]).run(input="Finalize Report")
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# --- FINAL DISPLAY ---
if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Strategic Investment Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
