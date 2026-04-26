import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent1_data_harvester import (
    scrape_finviz,
    fetch_aaii_sentiment,
    fetch_fear_greed,
    fetch_earnings_data,
    scrape_yahoo_finance,
    scrape_seeking_alpha
)

# Test 1 — Finviz
# print("Testing Finviz...")
# results = scrape_finviz("AAPL")
# print(f"  Got {len(results)} headlines")
# if results:
#     print(f"  First: {results}")

# Test 2 — AAII
# print("\nTesting AAII...")
# aaii = fetch_aaii_sentiment()
# print(f"  Result: {aaii}")

# # Test 3 — Fear & Greed
# print("\nTesting Fear & Greed...")
# fg = fetch_fear_greed()
# print(f"  Result: {fg}")

# Test 4 — Earnings dates
# print("\nTesting earnings dates...")
# earnings = fetch_earnings_data(["VZ"])
# print(f"  Result: {earnings}")

# # Test 5 — Yahoo Finance Articles
# print("\nTesting Yahoo Finance News...")
# yf_news = scrape_yahoo_finance("AAPL")
# print(f"  Result: {yf_news}")

# Test 5 — Seeking Alpha Headlines
# print("\nTesting Seeking Alpha Headlines...")
# sa_news = scrape_seeking_alpha("AAPL")
# print(f"  Result: {sa_news}")

# Test 6 - Podcast Retrieval