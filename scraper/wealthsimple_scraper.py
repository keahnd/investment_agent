import os
from dotenv import load_dotenv
from pathlib import Path
from playwright.sync_api import sync_playwright
import pyotp
import time
import tempfile
import shutil

load_dotenv()


def ws_login(page):
	page.goto("https://my.wealthsimple.com/app/login")
	page.wait_for_selector('[data-qa="login-email"] input', timeout=15000)
	
	page.fill('[data-qa="login-email"] input', os.environ["WS_EMAIL"])
	page.fill('[data-qa="login-password"] input', os.environ["WS_PASSWORD"])
	page.click('[data-testid="login-form-submit-ftux"]')
	
	page.wait_for_selector('[data-testid="otp-input"]')
	totp = pyotp.TOTP(os.environ["WS_SECRET"])
	if (totp.interval - (time.time() % totp.interval)) < 5:
		time.sleep(5)
	otp_input = page.locator('[data-testid="otp-input"] input')
	otp_input.click()
	otp_input.press_sequentially(totp.now())
	page.click('[data-testid="otp-submit-button"]')
	page.wait_for_url(lambda url: "login" not in url, timeout=15000)


def get_holdings(page, user_path: Path):
	save_path = user_path/"portfolio.csv"
	page.goto("https://my.wealthsimple.com/app/holdings-dashboard")

	page.wait_for_selector("#holdings-dashboard-container", timeout=15000)
	page.click('[data-testid="button-download-holdings"]')

	checkbox = page.get_by_role("checkbox", name="TFSA")
	checkbox.click()

	page.wait_for_timeout(500)  # let popup animate in
	# Capture the CSV download
	with page.expect_download() as dl:
		page.get_by_test_id("generate-documents-cta-button").click()
	dl.value.save_as(save_path)
	print(f"  [ok] Holdings saved to {save_path}")
	return save_path
	# page.pause()  # freeze here with popup visible
	

def scrape_user_holdings(user_path: Path):
	playwright = sync_playwright().start()
	browser = playwright.chromium.launch(
		headless=True,   # visible window
		args=["--disable-blink-features=AutomationControlled"]
	)
	context = browser.new_context(
		user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
	)
	page = context.new_page()
	
	try:
		ws_login(page)
		get_holdings(page, user_path)
	finally:
		browser.close()
		playwright.stop()
	