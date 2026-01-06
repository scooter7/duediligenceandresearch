import streamlit as st
import time, re, logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search
from tools import generate_html_report, generate_financial_chart, generate_infographic

# --- 1. SETUP & AUTH ---
st.set_page_config(page_title="Strategic Investment Intelligence", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentApp")

for k in ["plan_id", "tasks", "research_text", "final_memo", "auth"]:
    if k not in st.session_state: st.session_state[k] = None

if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Unlock"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

# Sidebar
with st.sidebar:
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key:
    st.info("Please enter your API key in the sidebar.")
    st.stop()

client = genai.Client(api_key=api_key)

# Helpers
def get_text(outputs): return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text'))
def parse_tasks(text): return [{"num": m.group(1), "text": m.group(2).strip()} for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

# --- STEP 1: PLANNING (Interactions API) ---
target = st.text_input("Enter Target", placeholder="e.g., Analyze https://agno.com for Series A")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Creating Research Plan..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=f"Create a 6-step research plan for: {target}. Find founder contact info.", 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun() 
        except Exception as e: st.error(f"Planning Error: {e}")

# --- DISPLAY TASKS & RUN DEEP RESEARCH ---
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected = [f"{t['num']}. {t['text']}" for t in st.session_state.tasks if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"t{t['num']}")]
    
    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent (autonomous) searching the web (2-5 mins)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input="Find founder contact info for:\n" + "\n".join(selected),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True, store=True
                )
                while True:
                    interaction = client.interactions.get(i.id)
                    if interaction.status != "in_progress": break
                    time.sleep(5)
                st.session_state.research_text = get_text(interaction.outputs)
                st.rerun()
            except Exception as e: st.error(f"Research Error: {e}")

# --- STEP 3: FULL MULTI-AGENT ADK PIPELINE ---
if st.session_state.research_text:
    st.divider()
    if st.button("📊 Step 3: Run Analysis Pipeline"):
        with st.spinner("Orchestrating ADK Agent Team (Finance, Risk, Memo)..."):
            try:
                # 1. Financial Agent
                fin_agent = LlmAgent(
                    name="FinancialModelingAgent", model="gemini-3-pro-preview",
                    instruction=f"Build projections from research: {st.session_state.research_text}",
                    tools=[generate_financial_chart], output_key="financial_model"
                )
                # 2. Risk Agent
                risk_agent = LlmAgent(
                    name="RiskAssessmentAgent", model="gemini-3-pro-preview",
                    instruction=f"Analyze risks from: {st.session_state.research_text}",
                    output_key="risk_assessment"
                )
                # 3. Memo & Visuals Agent
                memo_agent = LlmAgent(
                    name="InvestorMemoAgent", model="gemini-3-pro-preview",
                    instruction="Synthesize memo with Founder Contact Table. Generate HTML & Infographic.",
                    tools=[generate_html_report, generate_infographic]
                )

                # Sequential Pipeline Orchestration
                pipeline = SequentialAgent(
                    name="DueDiligencePipeline", 
                    sub_agents=[fin_agent, risk_agent, memo_agent]
                )
                
                # FIXED: SequentialAgent uses initiate() or run() depending on internal SDK version
                # In latest ADK, initiate() is the entry point for pipeline orchestration
                result = pipeline.initiate(input="Finalize full investment intelligence report.")
                
                st.session_state.final_memo = result.output if hasattr(result, 'output') else str(result)
                st.rerun()
            except Exception as e: st.error(f"Analysis Error: {e}")

if st.session_state.final_memo:
    st.divider()
    st.header("🏆 Strategic Investment Memo")
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
