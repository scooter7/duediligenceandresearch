import logging
import io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
from google.adk.tools import ToolContext
from google.genai import types, Client

matplotlib.use('Agg')
logger = logging.getLogger("DueDiligencePipeline")
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
        ax.plot(years, project(current_arr, bear), 'o-', label='Bear Case')
        ax.plot(years, project(current_arr, base), 's-', label='Base Case')
        ax.plot(years, project(current_arr, bull), '^-', label='Bull Case')
        ax.set_title(f'{company_name} Revenue Projection')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_bytes = buf.read()
        plt.close()

        artifact_name = f"chart_{datetime.now().strftime('%H%M%S')}.png"
        await tool_context.save_artifact(filename=artifact_name, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        (OUTPUTS_DIR / artifact_name).write_bytes(img_bytes)
        return {"status": "success", "artifact": artifact_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def generate_html_report(report_data, tool_context: ToolContext):
    # Simplified version for hosting stability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"report_{timestamp}.html"
    html_content = f"<html><body>{report_data}</body></html>"
    await tool_context.save_artifact(filename=artifact_name, artifact=types.Part.from_bytes(data=html_content.encode('utf-8'), mime_type="text/html"))
    (OUTPUTS_DIR / artifact_name).write_text(html_content)
    return {"status": "success", "artifact": artifact_name}

async def generate_infographic(data_summary, tool_context: ToolContext):
    # This tool requires the pro-image model as per your original file
    return {"status": "partial", "message": "Infographic generation called."}
