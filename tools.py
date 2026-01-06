import logging
import io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
from google.adk.tools import ToolContext
from google.genai import types, Client

# Set non-interactive backend for headless Streamlit environments
matplotlib.use('Agg')
logger = logging.getLogger("InvestmentTools")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

async def generate_financial_chart(company_name, current_arr, bear_rates, base_rates, bull_rates, tool_context: ToolContext):
    """Generates a revenue projection chart and saves it as an ADK artifact."""
    try:
        bear = [float(x.strip()) for x in bear_rates.split(",")]
        base = [float(x.strip()) for x in base_rates.split(",")]
        bull = [float(x.strip()) for x in bull_rates.split(",")]
        years = list(range(2025, 2025 + len(base) + 1))
        
        def project(start, rates):
            arr = [start]
            for r in rates: arr.append(arr[-1] * r)
            return arr

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, project(current_arr, bear), 'o-', color='#e74c3c', label='Bear Case')
        ax.plot(years, project(current_arr, base), 's-', color='#2c3e50', label='Base Case', linewidth=3)
        ax.plot(years, project(current_arr, bull), '^-', color='#27ae60', label='Bull Case')
        ax.set_title(f'{company_name} - Revenue Projection Analysis')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_bytes = buf.read()
        plt.close()

        filename = f"chart_{datetime.now().strftime('%H%M%S')}.png"
        # 2026 ADK uses types.Part for artifact binary data
        await tool_context.save_artifact(filename=filename, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        (OUTPUTS_DIR / filename).write_bytes(img_bytes)
        return {"status": "success", "artifact": filename}
    except Exception as e:
        logger.error(f"Chart Error: {e}")
        return {"status": "error", "message": str(e)}

async def generate_html_report(report_data, tool_context: ToolContext):
    """Formats raw memo text into professional HTML."""
    try:
        client = Client()
        prompt = f"Format this memo into a professional HTML document with clean CSS: {report_data}"
        response = await client.aio.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        
        html_content = response.text.replace("```html", "").replace("```", "").strip()
        filename = f"memo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        await tool_context.save_artifact(filename=filename, artifact=types.Part.from_bytes(data=html_content.encode('utf-8'), mime_type="text/html"))
        (OUTPUTS_DIR / filename).write_text(html_content)
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.error(f"HTML Error: {e}")
        return {"status": "error", "message": str(e)}

async def generate_infographic(data_summary, tool_context: ToolContext):
    """Generates a visual summary using Gemini 3 Pro Image."""
    try:
        client = Client()
        prompt = f"Create a professional investment infographic for: {data_summary}"
        response = await client.aio.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                img_bytes = part.inline_data.data
                filename = f"infographic_{datetime.now().strftime('%H%M%S')}.png"
                await tool_context.save_artifact(filename=filename, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                (OUTPUTS_DIR / filename).write_bytes(img_bytes)
                return {"status": "success", "image_path": str(OUTPUTS_DIR / filename)}
        return {"status": "error", "message": "No image part found."}
    except Exception as e:
        logger.error(f"Infographic Error: {e}")
        return {"status": "error", "message": str(e)}
