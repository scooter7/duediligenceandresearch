import streamlit as st
import time
import re
import logging
from pathlib import Path
from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from tools import generate_html_report, generate_infographic, generate_financial_chart

# --- 1. SYSTEM SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentApp")

st.set_page_config(page_title="Investment Intel 2026", layout="wide")

# Initialize Sessions at the very top
for k in ["plan_id", "tasks", "research_text", "final_memo"]:
    if k not in st.session_state:
        st.session_state[k] = None

# --- 2. AUTHENTICATION LAYER ---
def check_password():
    if "password_correct" not in st.session_state:
        st.subheader("🔐 Restricted Access")
        pw = st.text_input("Application Password", type="password")
        if st.button("Unlock"):
            if pw == st.secrets.get("APP_PASSWORD", "admin123"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Invalid password")
        return False
    return True

if check_password():
    # --- UI Components ---
    with st.sidebar:
        st.header("⚙️ Controls")
        api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
        if st.button("Clear All Data"):
            st.session_state.clear()
            st.rerun()

    st.title("📈 Strategic Investment Intelligence")

    if not api_key:
        st.info("Enter API key in the sidebar to start.")
        st.stop()

    client = genai.Client(api_key=api_key)

    def get_text(outputs): 
        return "\n".join(o.text for o in (outputs or []) if hasattr(o, 'text') and o.text) or ""

    def parse_tasks(text):
        return [{"num": m.group(1), "text": m.group(2).strip()} 
                for m in re.finditer(r'^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)]

    def wait_for_completion(client, iid):
        while True:
            interaction = client.interactions.get(iid)
            if interaction.status != "in_progress": return interaction
            time.sleep(5)

    # --- STEP 1: PLANNING ---
    target = st.text_input("Research Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
    if st.button("📋 Step 1: Generate Plan"):
        with st.spinner("Creating Research Plan..."):
            try:
                plan_prompt = f"Create a 6-step research plan for: {target}. Task 1 must focus on founder contact info."
                i = client.interactions.create(model="gemini-3-flash-preview", input=plan_prompt, store=True)
                st.session_state.plan_id = i.id
                st.session_state.tasks = parse_tasks(get_text(i.outputs))
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # --- STEP 2: RESEARCH (Renders if tasks exist) ---
    if st.session_state.tasks:
        st.divider()
        st.subheader("🔍 Investigation Phase")
        selected_tasks = []
        for t in st.session_state.tasks:
            if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=f"check_{t['num']}"):
                selected_tasks.append(f"{t['num']}. {t['text']}")
        
        if st.button("🚀 Step 2: Start Deep Research"):
            with st.spinner("Deep Research Agent browsing the web (2-5 mins)..."):
                try:
                    i = client.interactions.create(
                        agent="deep-research-pro-preview-12-2025", 
                        input="Research these tasks. Find founder emails/phones.\n" + "\n".join(selected_tasks),
                        previous_interaction_id=st.session_state.plan_id,
                        background=True, store=True
                    )
                    i = wait_for_completion(client, i.id)
                    st.session_state.research_text = get_text(i.outputs)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- STEP 3: ANALYSIS (Renders if research_text exists) ---
    if st.session_state.research_text:
        st.divider()
        with st.expander("Peek at Raw Research Findings"):
            st.markdown(st.session_state.research_text)

        if st.button("📊 Step 3: Run Multi-Agent Analysis"):
            with st.spinner("Synthesizing final report..."):
                try:
                    fin_agent = LlmAgent(name="Fin", model="gemini-3-pro-preview", 
                                        instruction=f"Extract numbers from: {st.session_state.research_text}", 
                                        tools=[generate_financial_chart])
                    partner_agent = LlmAgent(name="Partner", model="gemini-3-pro-preview", 
                                            instruction="Create a memo with a Founder Contact Directory.", 
                                            tools=[generate_html_report])
                    pipeline = SequentialAgent(name="Pipeline", sub_agents=[fin_agent, partner_agent])
                    st.session_state.final_memo = pipeline.run(input="Finalize Report")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- FINAL REPORT DISPLAY ---
    if st.session_state.final_memo:
        st.divider()
        st.header("🏆 Strategic Investment Memo")
        st.markdown(st.session_state.final_memo)
        st.download_button("📥 Download Report", st.session_state.final_memo, "memo.md", "text/markdown")
