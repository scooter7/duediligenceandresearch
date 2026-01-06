import logging
import io
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
from google.adk.tools import ToolContext
from google.genai import types

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InvestmentTools")

matplotlib.use('Agg')
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

async def generate_financial_chart(company_name, current_arr, bear_rates, base_rates, bull_rates, tool_context: ToolContext):
    logger.info(f"Generating chart for {company_name}...")
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
        ax.plot(years, project(current_arr, bear), 'o-', label='Bear', color='#e74c3c')
        ax.plot(years, project(current_arr, base), 's-', label='Base', color='#2c3e50')
        ax.plot(years, project(current_arr, bull), '^-', label='Bull', color='#27ae60')
        ax.set_title(f'{company_name} Revenue Analysis')
        ax.set_ylabel('ARR ($M)')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_bytes = buf.read()
        plt.close()

        artifact_name = f"chart_{datetime.now().strftime('%H%M%S')}.png"
        await tool_context.save_artifact(filename=artifact_name, artifact=types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        (OUTPUTS_DIR / artifact_name).write_bytes(img_bytes)
        logger.info(f"Successfully saved chart: {artifact_name}")
        return {"status": "success", "artifact": artifact_name}
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return {"status": "error", "message": str(e)}

async def generate_html_report(report_data, tool_context: ToolContext):
    logger.info("Generating HTML Report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"report_{timestamp}.html"
    html_content = f"""
    <html>
    <head><style>body{{font-family:sans-serif; line-height:1.6; color:#333; max-width:800px; margin:auto; padding:20px;}} 
    h1{{color:#2c3e50; border-bottom:2px solid #2c3e50;}}</style></head>
    <body><h1>Investment Intelligence Report</h1>{report_data}</body>
    </html>"""
    await tool_context.save_artifact(filename=artifact_name, artifact=types.Part.from_bytes(data=html_content.encode('utf-8'), mime_type="text/html"))
    (OUTPUTS_DIR / artifact_name).write_text(html_content)
    logger.info(f"Report saved: {artifact_name}")
    return {"status": "success", "artifact": artifact_name}

async def generate_infographic(data_summary, tool_context: ToolContext):
    logger.info("Infographic tool triggered.")
    return {"status": "partial", "message": "Visual infographic logic activated."}
