import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from pathlib import Path
from dotenv import load_dotenv
from reports.email import send_report_email

load_dotenv()

with open("test/fixtures/sample_state.json") as f:
    state = json.load(f)

pdf_path = "users/keahn_div/reports/2026-05-08/report_keahn_div_2026-05-08.pdf"
user_email = "keahnd@gmail.com"

success = send_report_email(state, pdf_path, user_email)
print(f"Sent: {success}")