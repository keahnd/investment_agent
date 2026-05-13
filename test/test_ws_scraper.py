# test_login.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from playwright.sync_api import sync_playwright
import pyotp
from dotenv import load_dotenv

from scraper.wealthsimple_scraper import ws_login, get_holdings

load_dotenv()

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True,   # visible window
        args=["--disable-blink-features=AutomationControlled"]
	)
    context = browser.new_context(
		user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
	)
    page = context.new_page()
    
    page.goto("https://my.wealthsimple.com/app/login")
    page.wait_for_timeout(2000)  # watch it load
    
    ws_login(page)
    get_holdings(page, Path("users/keahn_div"))
    
    page.pause()
    
    browser.close()