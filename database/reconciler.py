import logging
from datetime import datetime, timedelta, date
from database.schema import init_database, insert_portfolio_row, insert_model_output, insert_recommendation, insert_virtual_portfolio
import requests, zipfile, io
from curl_cffi import requests
import time
import math
import pandas as pd
import yfinance as yf

from agents.agent2_quant import fetch_prices, _to_yfinance_ticker

logger = logging.getLogger("investment_agent")


def _fetch_opening_prices(tickers, start):
    """
    Fetche opening prices for the tickers from the start date to today

    Args:
        tickers: list of tickers
        start: start date

    Returns:
        price dataframe
    """
    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")
    for attempt in range(5):
        wait = 30 * (attempt + 1)   # 30s, 60s, 90s, 120s, 150s
        try:
            end = datetime.today() + timedelta(days=1)
            session = requests.Session(impersonate="chrome")
            yf_tickers = [_to_yfinance_ticker(t) for t in tickers]
            back_map   = {yf_t: orig for orig, yf_t in zip(tickers, yf_tickers)}
            data = yf.download(
                yf_tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=False, session=session
            )['Open']
            if data.empty:
                raise ValueError("Download returned empty DataFrame (likely rate limited).")
            data = data.rename(columns=back_map)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            opening_prices = data.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else None)
            missing = [t for t, p in opening_prices.items() if p is None or (hasattr(p, '__float__') and pd.isna(p))]
            if missing:
                logger.warning(f"No opening price found for {missing}, extending start by 1 days...")
                start = start - timedelta(days=1)
                # This could result in grabbing opening prices BEFORE the recommendation ran. Giving the
                # VP false price values to work off of.
                continue
            
            actual_date = data.dropna(how='all').index[0].date()
            return opening_prices, actual_date

        except Exception as e:
            if attempt < 4:
                logger.warning(f"Download failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to download opening price data after 5 attempts: {e}")

def _validate_dates(conn, today, file):
    """
    Queries the DB for the most recent recommendation and virtual portfolio dates.

    Args:
        conn: Active database connection
        today: Today's date string (YYYY-MM-DD)
        file: File object to write warnings to

    Returns:
        Tuple of (last_rec_date, last_vp_date, opening_date), or None if
        either date is missing from the DB
    """
    last_rec_date = conn.execute("SELECT MAX(date) FROM recommendations WHERE date < ?", (today,)).fetchone()[0]
    if last_rec_date is None:
        print("  [ERROR] No recommendations found before today.", file=file)
        return None

    last_vp_date = conn.execute("SELECT MAX(date) FROM virtual_portfolio WHERE date < ?", (today,)).fetchone()[0]
    if last_vp_date is None:
        print("  [INFO] No prior virtual portfolio found. Will seed from real portfolio.", file=file)
    
    opening_date = datetime.strptime(last_rec_date, "%Y-%m-%d") + timedelta(days=1)
    today_dt = datetime.strptime(today, "%Y-%m-%d") if isinstance(today, str) else today
    days_since = (today_dt - datetime.strptime(last_rec_date, "%Y-%m-%d")).days
    if days_since > 14:
        print(f"  [WARNING] {days_since} days since last recommendation — larger gap than expected.", file=file)

    return last_rec_date, last_vp_date, opening_date


def _load_recommendations(conn, last_rec_date, file):
    """
    Fetches recommendations for a given date and groups weights by strategy.

    Args:
        conn: Active database connection
        last_rec_date: Date string (YYYY-MM-DD) to fetch recommendations for
        file: File object to write errors to

    Returns:
        Tuple of (last_rec, weights_by_strategy) where last_rec is a list of
        (ticker, weight, strategy) tuples and weights_by_strategy is
        {strategy: {ticker: weight}}
    """
    last_rec = conn.execute("SELECT ticker, recommended_weight, strategy FROM recommendations WHERE date = ?", (last_rec_date,)).fetchall()

    weights_by_strategy = {}
    for ticker, weight, strategy in last_rec:
        weights_by_strategy.setdefault(strategy, {})[ticker] = weight

    for strategy, weights in weights_by_strategy.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"  [ERROR] {strategy} weights sum to {total:.6f}, expected 1.0", file=file)

    return last_rec, weights_by_strategy


def _build_virtual_portfolio(last_rec, opening_vp_value, opening_prices):
    """
    Constructs the virtual portfolio dict from recommendations and opening prices.

    Args:
        last_rec: List of (ticker, weight, strategy) tuples from recommendations
        opening_vp_value: {strategy: total_value} — capital to deploy per strategy
        opening_prices: Series of {ticker: opening_price}

    Returns:
        {strategy: {ticker: {weight, shares, price, market_value}}}
    """
    weights_by_strategy = {}
    for ticker, weight, strategy in last_rec:
        weights_by_strategy.setdefault(strategy, {})[ticker] = weight

    virtual_portfolio = {}
    for ticker, weight, strategy in last_rec:
        price = opening_prices.get(ticker) if hasattr(opening_prices, "get") else opening_prices[ticker]
        if price is None or (isinstance(price, float) and (math.isnan(price) or price == 0.0)):
            logger.warning(f"{ticker}: invalid opening price ({price}) — skipping from VP build.")
            continue
        total_w = sum(weights_by_strategy[strategy].values())
        if total_w == 0:
            logger.warning(f"{strategy}: all weights are zero — skipping VP build for this strategy.")
            continue
        normalised_weight = weight / total_w
        market_value = opening_vp_value.get(strategy, 0.0) * normalised_weight
        virtual_portfolio.setdefault(strategy, {})[ticker] = {
            "weight": normalised_weight,
            "shares": market_value / price,
            "price": price,
            "market_value": market_value,
        }
    return virtual_portfolio


def _load_or_create_vp(conn, last_rec, opening_date, today, file):
    """
    Loads today's virtual portfolio from the DB if already written, otherwise
    builds it from last run's recommendations and opening prices, then writes it.

    Args:
        conn: Active database connection
        last_rec: List of (ticker, weight, strategy) tuples
        opening_date: datetime — date trades are executed at open
        today: Today's date string (YYYY-MM-DD)
        file: File object to write warnings to

    Returns:
        Tuple of ({strategy: {ticker: {weight, shares, price, market_value}}}, freshly_created: bool)
    """
    tickers = {ticker for ticker, _, _ in last_rec}
    today_str = today if isinstance(today, str) else today.isoformat()

    opening_prices, actual_opening_date = _fetch_opening_prices(tickers, opening_date)
    actual_opening_date_str = actual_opening_date.strftime("%Y-%m-%d")

    existing = conn.execute("SELECT COUNT(*) FROM virtual_portfolio WHERE date = ?", (actual_opening_date_str,)).fetchone()[0]
    valid_shares = conn.execute(
        "SELECT COUNT(*) FROM virtual_portfolio WHERE date = ? AND quantity IS NOT NULL AND quantity != 0",
        (actual_opening_date_str,)
    ).fetchone()[0]

    if existing > 0 and valid_shares == 0:
        print(f"  [WARN] Stored VP for {actual_opening_date_str} has all-NULL/zero shares. Deleting and rebuilding.", file=file)
        conn.execute("DELETE FROM virtual_portfolio WHERE date = ?", (actual_opening_date_str,))
        conn.commit()
        existing = 0

    if existing > 0:
        print(f"  [INFO] Virtual portfolio already written for {actual_opening_date_str}. Loading from DB.", file=file)
        rows = conn.execute("SELECT ticker, strategy, weight, price, quantity, market_value FROM virtual_portfolio WHERE date = ?",
                            (actual_opening_date_str,)).fetchall()
        virtual_portfolio = {}
        for ticker, strategy, weight, price, shares, market_value in rows:
            virtual_portfolio.setdefault(strategy, {})[ticker] = {
                "weight": weight, "shares": shares, "price": price, "market_value": market_value,
            }
        return virtual_portfolio, False, actual_opening_date_str

    opening_vp_value = {}
    # seed each strategy with the real portfolio's total value
    rp_total = conn.execute(
        "SELECT SUM(market_value) FROM portfolios WHERE date = ?", (today_str,)
    ).fetchone()[0] or 0.0
    strategies = {strategy for _, _, strategy in last_rec}
    for strategy in strategies:
        opening_vp_value[strategy] = rp_total

    virtual_portfolio = _build_virtual_portfolio(last_rec, opening_vp_value, opening_prices)

    for strategy, positions in virtual_portfolio.items():
        computed = sum(p["market_value"] for p in positions.values())
        if abs(computed - opening_vp_value[strategy]) > 0.01:
            print(f"  [WARN] {strategy} VP value mismatch: computed={computed:.2f}, expected={opening_vp_value[strategy]:.2f}", file=file)

    insert_virtual_portfolio(conn, actual_opening_date, virtual_portfolio)
    return virtual_portfolio, True, actual_opening_date_str


def _print_divergence_summary(virtual_portfolio, curr_vp_values, last_vp_values,
                               curr_rp_value, last_rp_value, real_weights,
                               raw_prices, conn, today, file):
    """
    Prints a structured divergence report comparing virtual and real portfolio performance.

    Sections: top-line returns per strategy, per-ticker weight/return contributions
    (top 10 by absolute impact), and cumulative divergence history from the DB.

    Args:
        virtual_portfolio: {strategy: {ticker: {weight, shares, price, market_value}}}
        curr_vp_values: {strategy: current_total_value}
        last_vp_values: {strategy: value_at_last_snapshot}
        curr_rp_value: Current real portfolio total market value
        last_rp_value: Real portfolio value at last recommendation date
        real_weights: {ticker: weight} of current real portfolio
        raw_prices: Price DataFrame (dates x tickers) from fetch_prices
        conn: Active database connection (for cumulative history query)
        today: Today's date string (YYYY-MM-DD)
        file: File object to write to
    """
    real_pct_change = (curr_rp_value - last_rp_value) / last_rp_value if last_rp_value else None
    virtual_pct_change = {
        s: (curr_vp_values[s] - last_vp_values[s]) / last_vp_values[s] if last_vp_values.get(s) else None for s in curr_vp_values
    }
    dollar_divergence = {s: curr_vp_values[s] - curr_rp_value for s in curr_vp_values}
    return_pct_divergence = {s: dollar_divergence[s] / curr_rp_value if curr_rp_value else None for s in dollar_divergence}

    print(f"\n{'='*70}", file=file)
    print(f"  PORTFOLIO DIVERGENCE SUMMARY — {today}", file=file)
    print(f"{'='*70}", file=file)

    print(f"\n  CURRENT REAL PORTFOLIO VALUE — ${curr_rp_value:,.2f}", file=file)
    rp_ret = f"{real_pct_change:>+9.2%}" if real_pct_change is not None else f"{'N/A':>9}"
    print(f"\n  CURRENT REAL PORTFOLIO RETURN — {rp_ret}", file=file)
    print(f"\n  {'Strategy':<15} {'Virtual Ret':>13} {'$ Diverg':>12} {'% Diverg':>12} {'$ Value':>12}", file=file)
    print(f"  {'-'*68}", file=file)
    for s in curr_vp_values:
        vp_ret  = f"{virtual_pct_change[s]:>+13.2%}" if virtual_pct_change[s] is not None else f"{'N/A':>12}"
        pct_div = f"{return_pct_divergence[s]:>+12.2%}" if return_pct_divergence[s] is not None else f"{'N/A':>12}"
        print(f"  {s:<15} {vp_ret}  {dollar_divergence[s]:>+12.2f} {pct_div} {curr_vp_values[s]:>12,.2f}", file=file)

    print(f"\n  KEY ASSET CONTRIBUTIONS", file=file)
    print(f"  Contrib = (Virtual Weight - Real Weight) x Ticker Return.", file=file)
    print(f"  A positive contrib means the VP's different weighting added return relative to the real portfolio.", file=file)
    print(f"  A negative contrib means the VP's weighting cost return. Top 10 by absolute impact shown.", file=file)
    for strategy, positions in virtual_portfolio.items():
        print(f"\n  [{strategy}]", file=file)
        print(f"  {'Ticker':<10} {'Virt Wt':>9} {'Real Wt':>9} {'Wt Diff':>9} {'Return':>9} {'Contrib':>10}", file=file)
        contributions = []
        for ticker, pos in positions.items():
            if ticker not in raw_prices.columns:
                continue
            entry_price = pos["price"]
            if not entry_price or (isinstance(entry_price, float) and (math.isnan(entry_price) or entry_price == 0.0)):
                continue
            curr_price = float(raw_prices[ticker].dropna().iloc[-1])
            ret = (curr_price - entry_price) / entry_price
            virt_w = pos["weight"]
            real_w = real_weights.get(ticker, 0.0)
            contributions.append((ticker, virt_w, real_w, ret, (virt_w - real_w) * ret))

        for ticker, virt_w, real_w, ret, contrib in sorted(contributions, key=lambda x: abs(x[4]), reverse=True)[:10]:
            print(f"  {ticker:<10} {virt_w:>9.1%} {real_w:>9.1%} {virt_w-real_w:>+9.1%} {ret:>+9.2%} {contrib:>+10.2%}", file=file)

    cumulative = conn.execute("""
        SELECT strategy, SUM(div) as total_divergence
        FROM (
            SELECT strategy, MAX(divergence) as div
            FROM virtual_portfolio
            WHERE market_value > 0 AND divergence IS NOT NULL
            GROUP BY DATE(date), strategy
        )
        GROUP BY strategy
        ORDER BY strategy
    """).fetchall()

    print(f"\n  CUMULATIVE DIVERGENCE", file=file)
    print(f"  {'Strategy':<20} {'Total Divergence':>16}", file=file)
    for strategy, total_div in cumulative:
        if total_div is None:
            continue
        print(f"  {strategy:<20} {total_div:>+16.2f}", file=file)

    print(f"\n{'='*70}", file=file)


def reconcile_virtual_portfolio(conn, today, file=None):
    """
    Tracks a virtual portfolio based on the most recent recommendations.
    Executes trades at the opening price of the next trading day and compares
    performance against the user's actual portfolio.

    Args:
        conn: Active database connection
        today: Today's date string (YYYY-MM-DD)
        file: File object to write output to

    Returns:
        Dict with curr_vp_values, last_vp_values, virtual_portfolio, real_weights,
        raw_prices — used by build_divergence_data. None if insufficient data.
    """
    if conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0] == 0:
        print("  [WARNING] No recommendations found. First run.", file=file)
        return None

    dates = _validate_dates(conn, today, file)
    if dates is None:
        return None
    last_rec_date, last_vp_date, opening_date = dates

    last_rec, weights_by_strategy = _load_recommendations(conn, last_rec_date, file)
    
    portfolio_tickers = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM portfolios WHERE date = ?", (today,)
    ).fetchall()}
    rec_tickers = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM recommendations WHERE date = ?", (last_rec_date,)
    ).fetchall()}
    tickers = sorted(portfolio_tickers | rec_tickers)

    raw_prices = fetch_prices(tickers, last_rec_date, today)
    
    last_vp_values = dict(conn.execute("SELECT strategy, SUM(market_value) FROM virtual_portfolio WHERE date = ? GROUP BY strategy",
                                       (last_vp_date,)).fetchall()) if last_vp_date else {}
    last_rp_value = conn.execute("SELECT SUM(market_value) FROM portfolios WHERE date = ?", (last_rec_date,)).fetchone()[0] or 0.0

    virtual_portfolio, freshly_created, actual_opening_date_str = _load_or_create_vp(conn, last_rec, opening_date, today, file)
    if freshly_created:
        last_vp_values = {}

    curr_rp_value = conn.execute("SELECT SUM(market_value) FROM portfolios WHERE date = ?", (today,)).fetchone()[0] or 0.0
    curr_vp_values = {
        s: sum(pos["shares"] * float(raw_prices[t].dropna().iloc[-1])
               for t, pos in positions.items()
               if t in raw_prices.columns and pos["shares"] is not None and pos["shares"] != 0)
        for s, positions in virtual_portfolio.items()
    }
    for strategy, vp_val in curr_vp_values.items():
        conn.execute(
            "UPDATE virtual_portfolio SET divergence = ? WHERE date = ? AND strategy = ?",
            (vp_val - curr_rp_value, actual_opening_date_str, strategy)
        )
    conn.commit()

    real_weights = dict(conn.execute("SELECT ticker, weight FROM portfolios WHERE date = ?", (today,)).fetchall())

    _print_divergence_summary(virtual_portfolio, curr_vp_values, last_vp_values,
                               curr_rp_value, last_rp_value, real_weights,
                               raw_prices, conn, today, file)

    return {
        "curr_vp_values":    curr_vp_values,
        "last_vp_values":    last_vp_values,
        "virtual_portfolio": virtual_portfolio,
        "real_weights":      real_weights,
        "raw_prices":        raw_prices,
    }



def build_divergence_data(conn, today: str, reconcile_result: dict = None) -> dict:
    """
    Queries the database to build the divergence summary dict for the report.
    Called from pipeline.py after reconciliation.

    Args:
        reconcile_result: Return value of reconcile_virtual_portfolio. When provided,
                          curr_vp_values (revalued at today's prices) and
                          last_vp_values are used for virtual return calculation,
                          and per-ticker contributions are included.

    Returns a dict with current values, returns, history, and contributions.
    Returns None if insufficient data exists.
    """
    has_rp = conn.execute(
        "SELECT COUNT(*) FROM portfolios WHERE date = ?", (today,)
    ).fetchone()[0]

    # Need either live reconcile data or a DB VP snapshot
    curr_vp_date = conn.execute(
        "SELECT MAX(DATE(date)) FROM virtual_portfolio WHERE market_value > 0"
    ).fetchone()[0]

    if (not reconcile_result and not curr_vp_date) or not has_rp:
        return None

    # Current real portfolio value
    curr_rp_value = conn.execute(
        "SELECT SUM(market_value) FROM portfolios WHERE date = ?", (today,)
    ).fetchone()[0] or 0.0

    last_rp_date = conn.execute(
        "SELECT MAX(DATE(date)) FROM portfolios WHERE date < ?", (today,)
    ).fetchone()[0]

    last_rp_value = 0.0
    if last_rp_date:
        last_rp_value = conn.execute(
            "SELECT SUM(market_value) FROM portfolios WHERE date = ?",
            (last_rp_date,)
        ).fetchone()[0] or 0.0

    real_return = (
        (curr_rp_value - last_rp_value) / last_rp_value
        if last_rp_value else None
    )

    # Current and last VP values — prefer live reconcile data (revalued at today's prices)
    if reconcile_result:
        curr_vp = reconcile_result["curr_vp_values"]
        last_vp = reconcile_result["last_vp_values"]
        # Fallback: VP seeded today, no prior DB snapshot exists yet.
        # Derive opening values from pos["market_value"] = shares × opening_price.
        if not last_vp and reconcile_result.get("virtual_portfolio"):
            for strategy, positions in reconcile_result["virtual_portfolio"].items():
                last_vp[strategy] = sum(pos["market_value"] for pos in positions.values())
    else:
        curr_vp_rows = conn.execute("""
            SELECT strategy, SUM(market_value) as total
            FROM virtual_portfolio WHERE DATE(date) = ? GROUP BY strategy
        """, (curr_vp_date,)).fetchall()
        curr_vp = {row[0]: row[1] for row in curr_vp_rows}

        last_vp_date = conn.execute(
            "SELECT MAX(DATE(date)) FROM virtual_portfolio WHERE date < ? AND market_value > 0",
            (curr_vp_date,)
        ).fetchone()[0]
        last_vp = {}
        if last_vp_date:
            last_vp_rows = conn.execute("""
                SELECT strategy, SUM(market_value) as total
                FROM virtual_portfolio WHERE date = ? GROUP BY strategy
            """, (last_vp_date,)).fetchall()
            last_vp = {row[0]: row[1] for row in last_vp_rows}

    # Build per-strategy summary
    strategies = {}
    for strategy, curr_val in curr_vp.items():
        last_val = last_vp.get(strategy, 0.0)
        virtual_return = (curr_val - last_val) / last_val if last_val else None
        dollar_div = curr_val - curr_rp_value
        pct_div    = dollar_div / curr_rp_value if curr_rp_value else None

        strategies[strategy] = {
            "curr_vp_value":     curr_val,
            "last_vp_value":     last_val,
            "virtual_return":    virtual_return,
            "dollar_divergence": dollar_div,
            "pct_divergence":    pct_div,
        }

    # Per-ticker contributions (requires live reconcile data)
    contributions = {}
    if reconcile_result:
        virtual_portfolio = reconcile_result["virtual_portfolio"]
        real_weights      = reconcile_result["real_weights"]
        raw_prices        = reconcile_result["raw_prices"]
        for strategy, positions in virtual_portfolio.items():
            rows = []
            for ticker, pos in positions.items():
                if ticker not in raw_prices.columns:
                    continue
                entry_price = pos["price"]
                if not entry_price or (isinstance(entry_price, float) and (math.isnan(entry_price) or entry_price == 0.0)):
                    continue
                curr_price = float(raw_prices[ticker].dropna().iloc[-1])
                ret        = (curr_price - entry_price) / entry_price
                virt_w     = pos["weight"]
                real_w     = real_weights.get(ticker, 0.0)
                rows.append((ticker, virt_w, real_w, ret, (virt_w - real_w) * ret))
            contributions[strategy] = sorted(rows, key=lambda x: abs(x[4]), reverse=True)[:10]

    # Cumulative divergence per strategy
    cumulative_rows = conn.execute("""
        SELECT strategy, SUM(div) as total_divergence
        FROM (
            SELECT strategy, MAX(divergence) as div
            FROM virtual_portfolio
            WHERE market_value > 0 AND divergence IS NOT NULL
            GROUP BY DATE(date), strategy
        )
        GROUP BY strategy
        ORDER BY strategy
    """).fetchall()
    cumulative_divergence = {row[0]: row[1] for row in cumulative_rows if row[1] is not None}

    return {
        "today":                 today,
        "curr_rp_value":         curr_rp_value,
        "last_rp_value":         last_rp_value,
        "real_return":           real_return,
        "strategies":            strategies,
        "contributions":         contributions,
        "cumulative_divergence": cumulative_divergence,
    }