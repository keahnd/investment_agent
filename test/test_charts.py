import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from pathlib import Path
from reports.charts import generate_all_charts

with open("test/fixtures/sample_state.json") as f:
    state = json.load(f)

charts_dir  = Path("test/output/charts")
charts_dir.mkdir(exist_ok=True, parents=True)
chart_paths = generate_all_charts(state, charts_dir)
