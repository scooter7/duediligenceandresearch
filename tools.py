import logging
import io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
from google.adk.tools import ToolContext
from google.genai import types, Client

# Set non-interactive backend for headless Streamlit
matplotlib.use('Agg')
logger = logging.getLogger("InvestmentTools")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

async def generate_financial_chart(company_name, current_arr, bear_rates, base_rates, bull_rates, tool_context: ToolContext):
    try:
        bear = [float(x.strip()) for x in bear_rates.split(",")]
        base = [float(x.strip()) for x in base_rates.split(",")]
        bull = [float(x.strip()) for x in bull_rates.split(",")]
        years = list(range(2025, 2025 + len(base) + 1))
        
        def project(start, rates):
            arr = [start]
            for r in rates: arr.append(arr[-1] * r)
            return arr

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, project(current_arr, bear), 'o-', label='Bear')
        ax.plot(years, project(current_arr, base), 's-', label='Base', linewidth=3)
        ax.plot(years, project(current_arr, bull), '^-', label='Bull')
        ax.set_title(f'{company_name} - Revenue Projections')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_bytes = buf.read()
        plt.close()

        fname = f"chart_{datetime.now().strftime('%H%M%S')}.png"
        await tool_context.save_artifact(filename=fname, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        (OUTPUTS_DIR / fname).write_bytes(img_bytes)
        return {"status": "success", "artifact": fname}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def generate_html_report(report_data, tool_context: ToolContext):
    try:
        client = Client()
        prompt = f"Format this memo into professional HTML: {report_data}"
        response = await client.aio.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        html = response.text.replace("```html", "").replace("```", "").strip()
        fname = f"memo_{datetime.now().strftime('%H%M%S')}.html"
        await tool_context.save_artifact(filename=fname, artifact=types.Part.from_bytes(data=html.encode('utf-8'), mime_type="text/html"))
        (OUTPUTS_DIR / fname).write_text(html)
        return {"status": "success", "filename": fname}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def generate_infographic(data_summary, tool_context: ToolContext):
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
                fname = f"infographic_{datetime.now().strftime('%H%M%S')}.png"
                await tool_context.save_artifact(filename=fname, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                (OUTPUTS_DIR / fname).write_bytes(img_bytes)
                return {"status": "success", "image_path": str(OUTPUTS_DIR / fname)}
        return {"status": "error", "message": "No image generated."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
