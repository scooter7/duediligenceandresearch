import streamlit as st
import asyncio
import time
import re
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.tools import google_search
# Ensure your local tools.py is in the same directory
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- 1. SYSTEM SETUP & AUTH ---
st.set_page_config(page_title="Strategic Investment Platform", layout="wide")

# Persistent State Initialization
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Unlock"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# Sidebar Config
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Deep Research Platform")
if not api_key:
    st.info("Please enter your API key in the sidebar.")
    st.stop()

client = genai.Client(api_key=api_key)

# Helpers
def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING ---
target = st.text_input("Research Target", placeholder="e.g., Analyze https://agno.com for Series A")

if st.button("📋 Step 1: Generate Strategic Plan") and target:
    with st.spinner("Planning investigation..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Focus on founders and contact info.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() 
        except Exception as e: st.error(f"Planning Error: {e}")

# DISPLAY: Rendered OUTSIDE button block to ensure it persists
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Focus")
    selected = [f"{t['num']}. {t['text']}" for t in st.session_state.tasks if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}")]
    
    # --- STEP 2: DEEP RESEARCH ---
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent investigating (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Thoroughly research these points, including founder contact details:\n" + "\n".join(selected),
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

# --- STEP 3: FULL 7-STAGE ADK PIPELINE ---
if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Research Findings"):
        st.markdown(st.session_state.research_text)

    if st.button("📊 Step 3: Run Full Analysis Team"):
        with st.spinner("Orchestrating 7 specialized agents..."):
            try:
                # Restoration of the full pipeline stages you provided
                research_agent = LlmAgent(name="ResearchAgent", model="gemini-3-flash-preview", 
                    instruction=f"Company info: {st.session_state.research_text}", tools=[google_search], output_key="company_info")
                
                market_agent = LlmAgent(name="MarketAgent", model="gemini-3-flash-preview",
                    instruction="Analyze {company_info} for market fit.", tools=[google_search], output_key="market_analysis")
                
                fin_agent = LlmAgent(name="FinancialAgent", model="gemini-3-pro-preview",
                    instruction="Build model for {company_info}.", tools=[generate_financial_chart], output_key="financial_model")
                
                risk_agent = LlmAgent(name="RiskAgent", model="gemini-3-pro-preview",
                    instruction="Identify risks for {company_info} and {market_analysis}.", output_key="risk_assessment")
                
                memo_agent = LlmAgent(name="MemoAgent", model="gemini-3-pro-preview",
                    instruction="Synthesize memo from all keys: {company_info}, {market_analysis}, {financial_model}, {risk_assessment}.", output_key="investor_memo")
                
                report_agent = LlmAgent(name="ReportAgent", model="gemini-3-flash-preview",
                    instruction="Format {investor_memo} as HTML.", tools=[generate_html_report])
                
                visual_agent = LlmAgent(name="VisualAgent", model="gemini-3-flash-preview",
                    instruction="Create infographic for {investor_memo}.", tools=[generate_infographic])

                # The Pipeline
                pipeline = SequentialAgent(name="DueDiligencePipeline", 
                                          sub_agents=[research_agent, market_agent, fin_agent, risk_agent, memo_agent, report_agent, visual_agent])
                
                # OFFICIAL RUNNER PATTERN (Solves the Attribute Error)
                async def execute_analysis():
                    runner = Runner(agent=pipeline)
                    final_result = ""
                    # Runner uses run_async to handle multi-agent events
                    async for event in runner.run_async(input="Finalize all artifacts."):
                        if hasattr(event, 'content'): # Extract content from the stream
                             final_result = event.content.parts[0].text
                    return final_result

                st.session_state.final_memo = asyncio.run(execute_analysis())
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Final Strategic Investment Intelligence")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "memo.md")
