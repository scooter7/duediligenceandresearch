import streamlit as st
import asyncio, time, re, logging, os
from pathlib import Path

from google import genai
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools import generate_html_report, generate_infographic, generate_financial_chart

logger = logging.getLogger(__name__)

# --- 1. SYSTEM SETUP ---
st.set_page_config(page_title="Investment Intelligence 2026", layout="wide")
Path("outputs").mkdir(exist_ok=True)

# Initialize Session State variables
for k in ["plan_id", "tasks", "research_text", "final_memo", "auth", "adk_session_id", "step2_debug"]:
    if k not in st.session_state:
        st.session_state[k] = None

# Create a stable session id for this Streamlit user session
if not st.session_state.adk_session_id:
    st.session_state.adk_session_id = "session_01"

# Auth Layer
if not st.session_state.auth:
    pw = st.text_input("Application Password", type="password")
    if st.button("Unlock"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")

    if st.button("Reset Everything"):
        st.session_state.clear()
        st.rerun()

st.title("📈 Strategic Investment & Research Platform")
if not api_key:
    st.info("Enter API Key in Sidebar.")
    st.stop()

# Make the API key available to tool code (tools.py) and any underlying SDK clients
os.environ["GOOGLE_API_KEY"] = api_key

# Gemini client for Steps 1-2
client = genai.Client(api_key=api_key)

def parse_tasks(text):
    return [
        {"num": m.group(1), "text": m.group(2).strip()}
        for m in re.finditer(
            r"^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    ]

def extract_text_any(obj) -> str:
    """
    Robustly pull text out of various Google GenAI/ADK output shapes.
    Handles:
      - obj.text
      - obj.parts[].text
      - obj.content.parts[].text
      - strings
      - lists/tuples of any of the above
    """
    if obj is None:
        return ""

    # Direct string
    if isinstance(obj, str):
        return obj

    # List / tuple -> join
    if isinstance(obj, (list, tuple)):
        return "\n".join([t for t in (extract_text_any(x) for x in obj) if t]).strip()

    chunks = []

    # Common: obj.text
    t = getattr(obj, "text", None)
    if isinstance(t, str) and t.strip():
        chunks.append(t.strip())

    # Common: obj.parts (list of Part-like)
    parts = getattr(obj, "parts", None)
    if isinstance(parts, (list, tuple)):
        for p in parts:
            pt = getattr(p, "text", None)
            if isinstance(pt, str) and pt.strip():
                chunks.append(pt.strip())

    # Common: obj.content.parts
    content = getattr(obj, "content", None)
    if content is not None:
        cparts = getattr(content, "parts", None)
        if isinstance(cparts, (list, tuple)):
            for p in cparts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt.strip():
                    chunks.append(pt.strip())

    return "\n".join(chunks).strip()

def get_text(outputs) -> str:
    # outputs is usually a list of output objects; extract robustly
    return extract_text_any(outputs)

# Persist session service across Streamlit reruns (best practice for ADK in Streamlit)
@st.cache_resource
def get_session_service():
    return InMemorySessionService()

# --- STEP 1: PLANNING ---
target = st.text_input("Analysis Target", placeholder="e.g., Pet cremation in Phoenix, AZ")
if st.button("📋 Step 1: Generate Plan") and target:
    with st.spinner("Planning..."):
        try:
            i = client.interactions.create(
                model="gemini-3-flash-preview",
                input=f"Plan research for: {target}. Find owner contact info.",
                store=True,
            )
            st.session_state.plan_id = i.id
            st.session_state.tasks = parse_tasks(get_text(i.outputs))
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- STEP 2: RESEARCH (RENDERED PERSISTENTLY) ---
if st.session_state.tasks:
    st.divider()
    st.subheader("🔍 Investigation Phase")
    selected_tasks = []

    for idx, t in enumerate(st.session_state.tasks):
        unique_key = f"check_{t['num']}_{idx}"
        if st.checkbox(f"Task {t['num']}: {t['text']}", value=True, key=unique_key):
            selected_tasks.append(f"{t['num']}. {t['text']}")

    if st.button("🚀 Step 2: Start Deep Research"):
        with st.spinner("Deep Research Agent investigating (polling)..."):
            try:
                i = client.interactions.create(
                    agent="deep-research-pro-preview-12-2025",
                    input="Find founder emails/phones for:\n" + "\n".join(selected_tasks),
                    previous_interaction_id=st.session_state.plan_id,
                    background=True,
                    store=True,
                )

                # Poll for completion
                last_status = None
                while True:
                    interaction = client.interactions.get(i.id)
                    last_status = getattr(interaction, "status", None)
                    if last_status != "in_progress":
                        break
                    time.sleep(5)

                # Save debug info no matter what (helps if output is empty)
                st.session_state.step2_debug = {
                    "interaction_id": getattr(interaction, "id", None),
                    "status": last_status,
                    "has_outputs": bool(getattr(interaction, "outputs", None)),
                    "outputs_type": str(type(getattr(interaction, "outputs", None))),
                    "raw_outputs_repr": repr(getattr(interaction, "outputs", None))[:4000],  # cap
                }

                text = get_text(getattr(interaction, "outputs", None))

                if not text.strip():
                    # If the run completed but we got no parseable text, show a useful warning
                    st.warning(
                        "Deep Research completed, but no text was extracted from outputs. "
                        "Open the debug expander below to see the raw outputs structure."
                    )

                st.session_state.research_text = text
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")

# Show Step 2 debug info if present
if st.session_state.step2_debug:
    with st.expander("🛠 Step 2 Debug (click if results look empty)"):
        st.json(st.session_state.step2_debug)

# --- STEP 3: ANALYSIS TEAM (RENDERED PERSISTENTLY) ---
if st.session_state.research_text:
    st.divider()
    with st.expander("Peek at Raw Data"):
        st.markdown(st.session_state.research_text)

    if st.button("📊 Step 3: Run Analysis Team"):
        with st.spinner("Orchestrating specialized agents..."):
            try:
                session_service = get_session_service()

                fin = LlmAgent(
                    name="Fin",
                    model="gemini-3-pro-preview",
                    instruction=(
                        "Use this research data to produce financial projections and charts where helpful.\n\n"
                        f"DATA:\n{st.session_state.research_text}"
                    ),
                    tools=[generate_financial_chart],
                )

                partner = LlmAgent(
                    name="Partner",
                    model="gemini-3-pro-preview",
                    instruction=(
                        "Write a professional investment memo with a clear Contact Table. "
                        "Use the provided tools to generate an HTML report and an infographic if helpful."
                    ),
                    tools=[generate_html_report, generate_infographic],
                )

                pipeline = SequentialAgent(name="AnalysisPipeline", sub_agents=[fin, partner])

                async def run_pipeline():
                    runner = Runner(
                        agent=pipeline,
                        session_service=session_service,
                        app_name="investment_intel_2026",
                    )

                    user_id = "streamlit_user"
                    session_id = st.session_state.adk_session_id

                    # ✅ Explicitly create the session (prevents Session not found)
                    await session_service.create_session(
                        app_name="investment_intel_2026",
                        user_id=user_id,
                        session_id=session_id,
                    )

                    input_content = types.Content(
                        role="user",
                        parts=[types.Part(text="Finalize Report")],
                    )

                    final_text = "Analysis failed."
                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session_id,
                        new_message=input_content,
                    ):
                        if event.is_final_response():
                            parts = getattr(event.content, "parts", []) or []
                            final_text = "\n".join(
                                p.text for p in parts if hasattr(p, "text") and p.text
                            ) or final_text
                    return final_text

                st.session_state.final_memo = asyncio.run(run_pipeline())
                st.rerun()

            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")

if st.session_state.final_memo:
    st.divider()
    st.markdown(st.session_state.final_memo)
    st.download_button("📥 Download Report", st.session_state.final_memo, "report.md")
