import streamlit as st
import time
import re
import os
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- 1. AUTHENTICATION LOGIC ---
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        # Retrieve the password from Streamlit Secrets or hardcode it (Secrets recommended)
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input("Please enter the Application Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Please enter the Application Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# --- 2. MAIN APP WRAPPER ---
if check_password():
    # --- SETUP & HELPERS ---
    st.set_page_config(page_title="Strategic Investment Intelligence", page_icon="📈", layout="wide")
    st.title("📈 Strategic Investment & Deep Research Platform")
    
    # Ensure local directory for tool artifacts exists
    Path("outputs").mkdir(exist_ok=True)

    def get_text(outputs): 
        return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text') and o.text) or ""

    def parse_tasks(text):
        return [{"num": m.group(1), "text": m.group(2).strip()} 
                for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

    def wait_for_completion(client, iid):
        while True:
            interaction = client.interactions.get(iid)
            if interaction.status != "in_progress": return interaction
            time.sleep(3)

    # --- SIDEBAR & AUTH ---
    with st.sidebar:
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("🔑 Google API Key", type="password")
        if st.button("Reset Session"):
            st.session_state.clear()
            st.rerun()

    if not api_key:
        st.info("Please enter your Google API Key to begin.")
        st.stop()

    client = genai.Client(api_key=api_key)

    # Initialize Session State for Workflow
    for k in ["plan_id", "tasks", "research_text", "final_memo"]:
        if k not in st.session_state: st.session_state[k] = None

    # --- PHASE 1: PLANNING (Interactions API) ---
    target = st.text_input("Company or Topic to Analyze", placeholder="e.g., Analyze https://replit.com for Series B")

    if st.button("📋 Step 1: Generate Strategic Plan") and target:
        with st.spinner("Planning investigation..."):
            planning_prompt = (
                f"Create a 6-step research plan for: {target}. "
                "IMPORTANT: Step 1 must focus on identifying founders and their direct contact info "
                "(professional emails and phone numbers)."
            )
            i = client.interactions.create(
                model="gemini-3-flash-preview", 
                input=planning_prompt, 
                store=True
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun()

    # --- PHASE 2: DEEP RESEARCH (Interactions API) ---
    if st.session_state.tasks:
        st.subheader("🔍 Select Research Focus")
        selected = [f"{t['num']}. {t['text']}" for t in st.session_state.tasks if st.checkbox(f"**{t['num']}**: {t['text']}", True)]
        
        if st.button("🚀 Step 2: Execute Deep Research"):
            with st.spinner("Deep Research Agent browsing the web (2-5 mins)..."):
                research_query = (
                    "Thoroughly investigate these tasks. For founder details, search specifically "
                    "for contact patterns and publicly listed professional emails or office lines.\n" 
                    + "\n".join(selected)
                )
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025", 
                    input=research_query, 
                    previous_interaction_id=st.session_state.plan_id, 
                    background=True, store=True
                )
                i = wait_for_completion(client, i.id)
                st.session_state.research_text = get_text(i.outputs)
                st.success("Deep Research Complete!")

    # --- PHASE 3: MULTI-AGENT PIPELINE (ADK) ---
    if st.session_state.research_text:
        with st.expander("View Raw Research Findings"):
            st.markdown(st.session_state.research_text)

        if st.button("📊 Step 3: Run Analysis Pipeline"):
            with st.spinner("Specialized agents are synthesizing reports..."):
                
                financial_agent = LlmAgent(
                    name="FinancialAgent", model="gemini-3-pro-preview",
                    instruction=f"Using this research: {st.session_state.research_text}, create projections.",
                    tools=[generate_financial_chart]
                )
                
                memo_agent = LlmAgent(
                    name="MemoAgent", model="gemini-3-pro-preview",
                    instruction=(
                        "Synthesize the research into a final investor memo. "
                        "REQUIRED: Create a 'Founder Contact Directory' section. "
                        "List every founder's name, email, and phone number found in the research. "
                        "If missing, explicitly state 'Not Discovered' for that field."
                    ),
                    tools=[generate_html_report, generate_infographic]
                )

                pipeline = SequentialAgent(name="InvestmentPipeline", sub_agents=[financial_agent, memo_agent])
                result = pipeline.run(input="Finalize full investment intelligence report.")
                st.session_state.final_memo = result
                st.rerun()

    if st.session_state.final_memo:
        st.divider()
        st.header("📊 Final Investment Memo")
        st.markdown(st.session_state.final_memo)
