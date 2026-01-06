import streamlit as st
import time
import re
import logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_infographic, generate_financial_chart

# Configure Logging for Streamlit Console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentApp")

# --- 1. AUTHENTICATION ---
def check_password():
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", key="password_input")
        if st.button("Login"):
            if st.session_state.password_input == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Invalid password")
        return False
    return True

# --- 2. MAIN APP ---
if check_password():
    st.set_page_config(page_title="Investment Intel 2026", layout="wide")
    st.title("📈 Strategic Investment Intelligence")
    
    def get_text(outputs): 
        return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text') and o.text) or ""

    def parse_tasks(text):
        return [{"num": m.group(1), "text": m.group(2).strip()} 
                for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

    def wait_for_completion(client, iid):
        logger.info(f"Polling interaction {iid}...")
        while True:
            interaction = client.interactions.get(iid)
            if interaction.status != "in_progress": 
                logger.info(f"Interaction {iid} finished with status: {interaction.status}")
                return interaction
            time.sleep(5)

    # --- SIDEBAR & API ---
    with st.sidebar:
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("API Key", type="password")
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()

    if not api_key:
        st.info("Please provide your Google API Key.")
        st.stop()

    client = genai.Client(api_key=api_key)

    # Initialize Sessions
    for k in ["plan_id", "tasks", "research_text", "final_memo", "planning_done"]:
        if k not in st.session_state: st.session_state[k] = None

    # --- PHASE 1: PLANNING ---
    target = st.text_input("Enter Target Company/URL", placeholder="e.g., https://agno.com")
    
    if st.button("📋 Step 1: Generate Strategic Plan") and target:
        logger.info(f"Starting planning for {target}")
        with st.spinner("Creating 2026 Research Plan..."):
            try:
                plan_prompt = (
                    f"Create a 6-step research plan for: {target}. "
                    "REQUIRED: Task 1 MUST find the founders' names, professional email addresses, and phone numbers."
                )
                i = client.interactions.create(model="gemini-3-flash-preview", input=plan_prompt, store=True)
                st.session_state.plan_id = i.id
                st.session_state.tasks = parse_tasks(get_text(i.outputs))
                st.session_state.planning_done = True
                logger.info("Plan successfully generated.")
                st.rerun() # FORCE UI REFRESH
            except Exception as e:
                logger.error(f"Planning Error: {e}")
                st.error(f"Planning Error: {e}")

    # --- PHASE 2: DEEP RESEARCH ---
    if st.session_state.tasks:
        st.subheader("🔍 Select Focused Tasks")
        selected_tasks = []
        for t in st.session_state.tasks:
            if st.checkbox(f"Task {t['num']}: {t['text']}", value=True):
                selected_tasks.append(f"{t['num']}. {t['text']}")
        
        if st.button("🚀 Step 2: Start Deep Research (2-5 mins)"):
            logger.info("Deep Research Agent activated.")
            with st.spinner("Browsing web for financials and founder contact details..."):
                try:
                    i = client.interactions.create(
                        agent="deep-research-pro-preview-12-2025", 
                        input="Research these tasks. Prioritize founder emails/phones.\n" + "\n".join(selected_tasks),
                        previous_interaction_id=st.session_state.plan_id,
                        background=True, store=True
                    )
                    i = wait_for_completion(client, i.id)
                    st.session_state.research_text = get_text(i.outputs)
                    logger.info("Deep research successfully retrieved.")
                    st.rerun() # FORCE UI REFRESH
                except Exception as e:
                    logger.error(f"Research Error: {e}")
                    st.error(f"Research Error: {e}")

    # --- PHASE 3: ANALYSIS ---
    if st.session_state.research_text:
        st.divider()
        if st.button("📊 Step 3: Run Multi-Agent Analysis"):
            logger.info("Sequential analysis pipeline started.")
            with st.spinner("Synthesizing final investment memo and founder directory..."):
                try:
                    fin_agent = LlmAgent(
                        name="FinAnalyst", model="gemini-3-pro-preview",
                        instruction=f"Build projections from: {st.session_state.research_text}",
                        tools=[generate_financial_chart]
                    )
                    partner_agent = LlmAgent(
                        name="Partner", model="gemini-3-pro-preview",
                        instruction=(
                            "Review research and write an investment memo. "
                            "MANDATORY: Include a 'Founder Contact Directory' table with Emails/Phones."
                        ),
                        tools=[generate_html_report, generate_infographic]
                    )
                    pipeline = SequentialAgent(name="Pipeline", sub_agents=[fin_agent, partner_agent])
                    st.session_state.final_memo = pipeline.run(input="Finalize Report")
                    logger.info("Final memo generated.")
                    st.rerun() # FORCE UI REFRESH
                except Exception as e:
                    logger.error(f"Pipeline Error: {e}")
                    st.error(f"Analysis Error: {e}")

    # --- FINAL OUTPUTS ---
    if st.session_state.final_memo:
        st.divider()
        st.markdown("### 🏆 Final Investment Intelligence")
        st.markdown(st.session_state.final_memo)
