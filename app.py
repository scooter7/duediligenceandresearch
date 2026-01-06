import streamlit as st
import asyncio, time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- SYSTEM SETUP ---
st.set_page_config(page_title="Investment Intel 2026", layout="wide")
Path("outputs").mkdir(exist_ok=True)

for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

if not st.session_state.auth:
    pw = st.text_input("App Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

with st.sidebar:
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

client = genai.Client(api_key=api_key) if api_key else None
if not client: st.info("Enter API Key in sidebar."); st.stop()

def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING ---
target = st.text_input("Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Planning..."):
        try:
            i = client.interactions.create(model="gemini-3-flash-preview", input=f"Plan research for: {target}.", store=True)
            st.session_state.plan_id, st.session_state.tasks = i.id, parse_tasks(get_text(i.outputs))
            st.rerun() 
        except Exception as e: st.error(f"Error: {e}")

# --- STEP 2: RESEARCH ---
if st.session_state.tasks:
    st.divider()
    selected = [f"{t['num']}. {t['text']}" for t in st.session_state.tasks if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}")]
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Researching (2-5 mins)..."):
            try:
                i = client.interactions.create(agent="deep-research-pro-preview-12-2025", input="\n".join(selected), previous_interaction_id=st.session_state.plan_id, background=True, store=True)
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# --- STEP 3: ANALYSIS TEAM ---
if st.session_state.research_text:
    st.divider()
    if st.button("📊 Step 3: Run Analysis Team"):
        with st.spinner("Orchestrating agents..."):
            try:
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", instruction=f"Data: {st.session_state.research_text}", tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", instruction="Final memo with Contact Table.", tools=[generate_html_report, generate_infographic])
                pipeline = SequentialAgent(name="Pipe", sub_agents=[fin, partner])

                async def run_analysis():
                    runner = Runner(agent=pipeline) # Use Runner for pipelines
                    events = []
                    async for event in runner.run_async(input="Finalize Report"):
                        events.append(event)
                    # Use is_final_response() check for stable output extraction
                    return events[-1].content.parts[0].text if events else "Analysis failed."

                st.session_state.final_memo = asyncio.run(run_analysis())
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.divider()
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
