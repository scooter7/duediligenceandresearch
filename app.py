import streamlit as st
import asyncio
import time
import re
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
st.set_page_config(page_title="Investment Intelligence 2026", layout="wide")
Path("outputs").mkdir(exist_ok=True)

# Initialize Session State
DEFAULT_STATE = {
    "plan_id": None,
    "tasks": None,
    "research_text": None,
    "final_memo": None,
    "auth": None,
    "step2_status": None,
    "step3_status": None,
    "artifacts": [],
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- AUTHENTICATION ---
if not st.session_state.auth:
    st.title("🔒 Investment Intelligence Platform")
    pw = st.text_input("Application Password", type="password")
    if st.button("Unlock"):
        if pw == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.text_input("Gemini API Key", type="password")
    
    st.divider()
    if st.button("🔄 Reset Everything", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != "auth":
                del st.session_state[key]
        st.rerun()
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Step 1**: Enter target and generate plan
        2. **Step 2**: Select tasks and run research
        3. **Step 3**: Generate analysis report
        
        Each step builds on the previous one.
        """)

# --- MAIN UI ---
st.title("📈 Strategic Investment & Research Platform")

if not api_key:
    st.warning("⚠️ Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()

# Make API key available to environment
os.environ["GOOGLE_API_KEY"] = api_key
client = genai.Client(api_key=api_key)

# --- HELPER FUNCTIONS ---

def parse_tasks(text: str) -> List[Dict[str, str]]:
    """Extract numbered tasks from text."""
    tasks = []
    for match in re.finditer(
        r"^(\d+)[\.\)\-]\s*(.+?)(?=\n\d+[\.\)\-]|\n\n|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        tasks.append({
            "num": match.group(1),
            "text": match.group(2).strip()
        })
    return tasks


def extract_text_from_response(response) -> str:
    """Robustly extract text from Gemini API response."""
    if response is None:
        return ""
    
    # Try direct text attribute
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    
    # Try candidates structure
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                content = candidate.content
                if hasattr(content, 'parts') and content.parts:
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts).strip()
    
    # Fallback to string representation
    return str(response).strip()


def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with error handling."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        logger.error(f"API call failed: {str(e)}", exc_info=True)
        return None, str(e)


# --- STEP 1: PLANNING ---
st.header("Step 1: Planning 📋")

col1, col2 = st.columns([3, 1])
with col1:
    target = st.text_input(
        "Analysis Target",
        placeholder="e.g., Pet cremation services in Phoenix, AZ",
        help="Describe the business or market you want to analyze"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    plan_button = st.button("🎯 Generate Plan", type="primary", use_container_width=True)

if plan_button and target:
    with st.spinner("🤔 Creating research plan..."):
        prompt = f"""Create a detailed research plan for analyzing this investment opportunity: {target}

Break down the research into 5-7 specific, actionable tasks. Include:
- Market research tasks
- Competitive analysis
- Financial analysis
- Contact information discovery (founders, key executives)

Format as a numbered list."""

        response, error = safe_api_call(
            client.models.generate_content,
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        if error:
            st.error(f"❌ Planning failed: {error}")
        elif response:
            text = extract_text_from_response(response)
            st.session_state.tasks = parse_tasks(text)
            st.session_state.plan_id = f"plan_{int(time.time())}"
            st.session_state.research_text = None  # Reset downstream
            st.session_state.final_memo = None
            st.success("✅ Plan generated successfully!")
            st.rerun()

# Display generated plan
if st.session_state.tasks:
    st.divider()
    with st.expander("📋 Research Plan", expanded=True):
        for task in st.session_state.tasks:
            st.markdown(f"**{task['num']}.** {task['text']}")

# --- STEP 2: RESEARCH ---
if st.session_state.tasks:
    st.header("Step 2: Deep Research 🔍")
    
    st.info("💡 Select which research tasks to execute. This uses AI to gather detailed information.")
    
    selected_tasks = []
    cols = st.columns(2)
    
    for idx, task in enumerate(st.session_state.tasks):
        col_idx = idx % 2
        with cols[col_idx]:
            checked = st.checkbox(
                f"**Task {task['num']}**: {task['text'][:60]}...",
                value=True,
                key=f"task_check_{task['num']}_{idx}"
            )
            if checked:
                selected_tasks.append(f"{task['num']}. {task['text']}")
    
    st.write(f"**Selected: {len(selected_tasks)} of {len(st.session_state.tasks)} tasks**")
    
    if st.button("🚀 Start Research", type="primary", disabled=len(selected_tasks) == 0):
        with st.spinner("🔬 Running deep research (this may take 2-5 minutes)..."):
            
            # Create comprehensive research prompt
            research_prompt = f"""Conduct thorough research on: {target}

Focus on these specific tasks:
{chr(10).join(selected_tasks)}

For each task, provide:
1. Detailed findings with specific data points
2. Sources and references where applicable
3. Key contacts (names, titles, emails, phone numbers if available)

Structure your response clearly with headers for each task."""

            try:
                # Use direct generate_content for more stability
                response, error = safe_api_call(
                    client.models.generate_content,
                    model="gemini-2.0-flash-exp",
                    contents=research_prompt
                )
                
                if error:
                    st.error(f"❌ Research failed: {error}")
                    st.session_state.step2_status = "error"
                elif response:
                    text = extract_text_from_response(response)
                    
                    if text and len(text) > 100:
                        st.session_state.research_text = text
                        st.session_state.step2_status = "complete"
                        st.session_state.final_memo = None  # Reset downstream
                        st.success("✅ Research completed successfully!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Research returned minimal results. Try adjusting your tasks.")
                        st.session_state.step2_status = "incomplete"
                        
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.step2_status = "error"
                logger.error(f"Research error: {str(e)}", exc_info=True)

# Display research results
if st.session_state.research_text:
    st.divider()
    with st.expander("📊 Research Results", expanded=False):
        st.markdown(st.session_state.research_text)
    
    # Show preview
    preview = st.session_state.research_text[:500]
    st.info(f"**Research Preview**: {preview}... *(click expander above for full results)*")

# --- STEP 3: ANALYSIS ---
if st.session_state.research_text:
    st.header("Step 3: Generate Analysis Report 📊")
    
    st.info("💡 This will create a comprehensive investment memo with financial projections.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        include_financials = st.checkbox("Include Financial Projections", value=True)
    with col2:
        include_contacts = st.checkbox("Include Contact Table", value=True)
    with col3:
        format_html = st.checkbox("Generate HTML Report", value=False)
    
    if st.button("📝 Generate Investment Memo", type="primary"):
        with st.spinner("✍️ Creating professional investment memo..."):
            
            # Build analysis prompt
            analysis_prompt = f"""You are an expert investment analyst. Create a comprehensive investment memo based on this research:

{st.session_state.research_text}

Your memo should include:

1. **Executive Summary** (2-3 paragraphs)
   - Key investment thesis
   - Market opportunity
   - Competitive positioning

2. **Market Analysis**
   - Market size and growth
   - Key trends
   - Competitive landscape

3. **Business Model & Operations**
   - Revenue model
   - Key metrics
   - Operational highlights

"""
            
            if include_financials:
                analysis_prompt += """
4. **Financial Projections**
   - Create 3-year revenue projections with Bear/Base/Bull scenarios
   - Format as a simple table with clear assumptions
   - Include key financial metrics
"""
            
            if include_contacts:
                analysis_prompt += """
5. **Key Contacts**
   - Create a formatted table with: Name, Title, Email, Phone
   - Include all contacts found in the research
"""
            
            analysis_prompt += """
6. **Investment Recommendation**
   - Clear recommendation (Strong Buy/Buy/Hold/Pass)
   - Key risks and mitigations
   - Next steps

Format professionally with clear sections and markdown formatting."""
            
            try:
                response, error = safe_api_call(
                    client.models.generate_content,
                    model="gemini-2.0-flash-exp",
                    contents=analysis_prompt
                )
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                    st.session_state.step3_status = "error"
                elif response:
                    memo_text = extract_text_from_response(response)
                    
                    if memo_text and len(memo_text) > 100:
                        # If HTML requested, convert
                        if format_html:
                            html_prompt = f"""Convert this investment memo to professional HTML with CSS styling:

{memo_text}

Use clean, modern styling with proper headers, tables, and formatting. Include inline CSS."""
                            
                            html_response, html_error = safe_api_call(
                                client.models.generate_content,
                                model="gemini-2.0-flash-exp",
                                contents=html_prompt
                            )
                            
                            if html_response and not html_error:
                                html_text = extract_text_from_response(html_response)
                                # Clean up markdown code blocks if present
                                html_text = html_text.replace("```html", "").replace("```", "").strip()
                                
                                # Save HTML file
                                html_file = Path("outputs") / f"memo_{int(time.time())}.html"
                                html_file.write_text(html_text, encoding="utf-8")
                                st.session_state.artifacts.append(str(html_file))
                                
                                st.session_state.final_memo = memo_text
                                st.success(f"✅ HTML report saved to: {html_file}")
                            else:
                                st.warning("⚠️ HTML conversion failed, showing markdown version")
                                st.session_state.final_memo = memo_text
                        else:
                            st.session_state.final_memo = memo_text
                        
                        st.session_state.step3_status = "complete"
                        st.success("✅ Investment memo generated successfully!")
                        st.rerun()
                    else:
                        st.warning("⚠️ Analysis returned minimal results.")
                        st.session_state.step3_status = "incomplete"
                        
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.step3_status = "error"
                logger.error(f"Analysis error: {str(e)}", exc_info=True)

# Display final memo
if st.session_state.final_memo:
    st.divider()
    st.success("🎉 Analysis Complete!")
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Memo (Markdown)",
            st.session_state.final_memo,
            "investment_memo.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.artifacts:
            latest_artifact = st.session_state.artifacts[-1]
            if Path(latest_artifact).exists():
                with open(latest_artifact, 'r', encoding='utf-8') as f:
                    st.download_button(
                        "📥 Download Report (HTML)",
                        f.read(),
                        "investment_report.html",
                        mime="text/html",
                        use_container_width=True
                    )
    
    # Display memo
    st.markdown("---")
    st.markdown(st.session_state.final_memo)

# --- FOOTER ---
st.divider()
st.caption("Investment Intelligence Platform 2026 | Powered by Google Gemini")
