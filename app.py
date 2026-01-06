import streamlit as st
import time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_financial_chart

# --- SYSTEM SETUP ---
st.set_page_config(page_title="Investment Intel 2026", layout="wide")
Path("outputs").mkdir(exist_ok=True)

# Initialize Session State 
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

# --- AUTHENTICATION ---
if not st.session_state.auth:
    pw = st.text_input("Enter App Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key:
    st.info("Please set the API Key in the sidebar.")
    st.stop()

client = genai.Client(api_key=api_key)

def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- PHASE 1: PLANNING ---
target = st.text_input("Research Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Planning..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Find owner emails/phones.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() # Refresh to show checkboxes 
        except Exception as e: st.error(f"Error: {e}")

# --- PHASE 2: RESEARCH ---
if st.session_state.tasks:
    st.divider()
    selected = [f"{t['num']}. {t['text']}" for t in st.session_state.tasks if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}")]
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent investigating (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find owner contact info for:\n" + "\n".join(selected),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                while True: # Poll for completion 
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun() # Refresh to show results 
            except Exception as e: st.error(f"Error: {e}")

# --- PHASE 3: ANALYSIS ---
if st.session_state.research_text:
    st.divider()
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Synthesizing..."):
            try:
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", instruction=f"Data: {st.session_state.research_text}", tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", instruction="Write memo with Contact Table.", tools=[generate_html_report])
                st.session_state.final_memo = SequentialAgent(name="Pipe", sub_agents=[fin, partner]).run(input="Finalize Report")
                st.rerun() # Refresh to show final memo 
            except Exception as e: st.error(f"Error: {e}")

# --- DISPLAY ---
if st.session_state.final_memo:
    st.divider()
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
