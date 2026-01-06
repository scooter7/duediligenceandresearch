import streamlit as st
import asyncio, time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- 1. SYSTEM SETUP ---
st.set_page_config(page_title="Investment Intelligence 2026", layout="wide")
Path("outputs").mkdir(exist_ok=True)

# Initialize Session State
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

# Auth Layer
if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

with st.sidebar:
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Everything"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key: st.info("Enter API Key in Sidebar."); st.stop()

client = genai.Client(api_key=api_key)

def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING ---
target = st.text_input("Analysis Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Planning..."):
        try:
            i = client.interactions.create(model="gemini-3-flash-preview", input=f"Plan research for: {target}. Find owner contact info.", store=True)
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun()
        except Exception as e: st.error(f"Error: {e}")

# --- STEP 2: RESEARCH (RENDERED PERSISTENTLY) ---
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_tasks = []
    # UNIQUE KEYS: Ensure keys are only generated once per rerun cycle
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"check_{t['num']}"):
            selected_tasks.append(f"{t['num']}. {t['text']}")
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent investigating (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder emails/phones for:\n" + "\n".join(selected_tasks),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                while True: # Poll for completion
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# --- STEP 3: ANALYSIS TEAM (RENDERED PERSISTENTLY) ---
if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Data"): st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Analysis Team"):
        with st.spinner("Orchestrating specialized agents..."):
            try:
                # Setup Pipeline Agents
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                               instruction=f"Data: {st.session_state.research_text}", tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                   instruction="Write memo with Contact Table.", tools=[generate_html_report, generate_infographic])
                
                pipeline = SequentialAgent(name="AnalysisPipeline", sub_agents=[fin, partner])

                # Official Runner Pattern for 2026 ADK
                async def run_pipeline():
                    runner = Runner(agent=pipeline)
                    events = []
                    async for event in runner.run_async(input="Finalize Report"):
                        events.append(event)
                    # Extract content from final response event
                    return events[-1].content.parts[0].text if events else "Analysis failed."

                st.session_state.final_memo = asyncio.run(run_pipeline())
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {str(e)}")

if st.session_state.final_memo:
    st.divider()
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
