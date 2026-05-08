import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
from reports.charts import generate_all_charts
from reports.report import generate_report

with open("test/fixtures/sample_state.json") as f:
    state = json.load(f)

charts_dir  = Path("test/output/charts")
chart_paths = generate_all_charts(state, charts_dir)

pdf_path = generate_report(
    final_state     = state,
    user_path       = Path("test/output"),
    charts_dir     = chart_paths,
    divergence_data = None,   # placeholder shown in section 2
)

print(f"Report: {pdf_path}")