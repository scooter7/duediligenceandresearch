import streamlit as st
import asyncio
import time
import re
import logging
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from tools import generate_html_report, generate_financial_chart

# --- 1. SESSION INITIALIZATION ---
st.set_page_config(page_title="Investment Intelligence", layout="wide")

# Persistent state keys
for key in ["tasks", "plan_id", "research_text", "final_memo", "auth"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Authentication Layer
if not st.session_state.auth:
    st.subheader("🔐 Investment Intelligence Portal")
    pw = st.text_input("Application Password", type="password")
    if st.button("Login"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# --- 2. SIDEBAR & API ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

client = genai.Client(api_key=api_key) if api_key else None
if not client:
    st.info("Please enter your API key to continue.")
    st.stop()

# Helpers
def get_text(outputs):
    return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))

def parse_tasks(text):
    return [{"num": m.group(1), "text": m.group(2).strip()} 
            for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

st.title("📈 Strategic Investment & Research Platform")

# --- 3. STEP 1: PLANNING (Interactions API) ---
target = st.text_input("Target Analysis", placeholder="e.g., Pet cremation in Phoenix, AZ")

# Action: Button triggers the work
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Generating Strategic Research Plan..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Focus on finding owner contact details.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() # FORCE UI REFRESH TO RENDER TASKS
        except Exception as e: st.error(f"Planning Error: {e}")

# Display: Renders independently of the button once data is in state
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_list = []
    for t in st.session_state.tasks:
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}"):
            selected_list.append(f"{t['num']}. {t['text']}")
    
    # --- 4. STEP 2: DEEP RESEARCH (Interactions API) ---
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent investigating (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder contact details for:\n" + "\n".join(selected_list),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                while True: # Poll for completion
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun() 
            except Exception as e: st.error(f"Research Error: {e}")

# --- 5. STEP 3: ANALYSIS (ADK Sequential Pipeline) ---
if st.session_state.research_text:
    st.divider()
    with st.expander("View Raw Evidence"):
        st.markdown(st.session_state.research_text)
    
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Orchestrating ADK Agent Team..."):
            try:
                # Setup Sub-Agents
                fin_agent = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                                     instruction=f"Build model from: {st.session_state.research_text}", 
                                     tools=[generate_financial_chart], output_key="fin_data")
                memo_agent = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                      instruction="Write final memo with contact table.", 
                                      tools=[generate_html_report], output_key="memo_text")
                
                # Setup Pipeline
                pipeline = SequentialAgent(name="InvestmentPipeline", sub_agents=[fin_agent, memo_agent])
                
                # Execute using a Runner (The stable way for ADK pipelines)
                async def run_pipeline():
                    runner = Runner(root_agent=pipeline)
                    events = []
                    async for event in runner.run_async(input="Analyze research and finalize memo."):
                        events.append(event)
                    # Extract final text from the event stream
                    return events[-1].content.parts[0].text if events else "No output."

                st.session_state.final_memo = asyncio.run(run_pipeline())
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Final Strategic Investment Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
