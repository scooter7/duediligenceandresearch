import streamlit as st
import asyncio
import time
import re
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_financial_chart

# --- 1. INITIALIZATION & AUTH ---
st.set_page_config(page_title="Investment Intel 2026", layout="wide")

# Initialize all state variables at the top to prevent "missing" results
for key in ["tasks", "plan_id", "research_text", "final_memo", "auth"]:
    if key not in st.session_state:
        st.session_state[key] = None

if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Everything"):
        for k in ["tasks", "plan_id", "research_text", "final_memo"]:
            st.session_state[k] = None
        st.rerun()

client = genai.Client(api_key=api_key) if api_key else None
if not client:
    st.info("Please provide your API key to continue.")
    st.stop()

# Helpers
def get_text(outputs):
    return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))

def parse_tasks(text):
    return [{"num": m.group(1), "text": m.group(2).strip()} 
            for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- 3. STEP 1: PLANNING ---
target = st.text_input("Research Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan"):
    with st.spinner("Creating Research Plan..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Find owner emails.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() # Refresh to lock results into UI
        except Exception as e: st.error(f"Planning Error: {e}")

# --- 4. STEP 2: RESEARCH (PERSISTENT RENDER) ---
# This block runs OUTSIDE the button to ensure it stays on screen
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Selected Investigation Tasks")
    selected_list = []
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}"):
            selected_list.append(f"{t['num']}. {t['text']}")
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Agent searching the web (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder details for:\n" + "\n".join(selected_list),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                # Polling loop
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun() # Refresh to show research results
            except Exception as e: st.error(f"Research Error: {e}")

# --- 5. STEP 3: ANALYSIS (PERSISTENT RENDER) ---
if st.session_state.research_text:
    st.divider()
    with st.expander("View Raw Evidence"):
        st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Agents analyzing data..."):
            try:
                # Setup ADK Agents
                fin = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                               instruction=f"Build model from: {st.session_state.research_text}", 
                               tools=[generate_financial_chart])
                partner = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                   instruction="Write memo with Contact Table.", 
                                   tools=[generate_html_report])
                pipeline = SequentialAgent(name="AnalysisPipeline", sub_agents=[fin, partner])

                # ADK Python uses run_async, which we run in a local loop
                async def run_pipeline():
                    results = []
                    async for event in pipeline.run_async(input="Finalize Memo"):
                        if hasattr(event, 'output'): results.append(event.output)
                    return results[-1] if results else "No output generated."

                st.session_state.final_memo = asyncio.run(run_pipeline())
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Final Strategic Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download", st.session_state.final_memo, "report.md")
