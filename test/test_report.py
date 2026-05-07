import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
from reports.charts import generate_all_charts
from reports.report import generate_report

with open("test/fixtures/sample_state.json") as f:
    state = json.load(f)

# Generate charts first
charts_dir  = Path("test/output/charts")
chart_paths = generate_all_charts(state, charts_dir)

# Generate report
user_path = Path("test/output")
pdf_path  = generate_report(state, user_path, charts_dir)
print(f"\nReport generated: {pdf_path}")