from datetime import datetime, timedelta, date
from database.schema import init_database, insert_portfolio_row, insert_model_output, insert_recommendation, insert_virtual_portfolio
import requests, zipfile, io
import time
import yfinance as yf

from agents.agent2_quant import fetch_prices


TODAY = datetime.today()


def _fetch_opening_prices(tickers, start):
    """
    Fetche opening prices for the tickers from the start date to today
    
    Args:
        tickers: list of tickers
        start: start date
        
    Returns:
        price dataframe
    """
    for attempt in range(5):
        wait = 30 * (attempt + 1)   # 30s, 60s, 90s, 120s, 150s
        try:
            session = requests.Session(impersonate="chrome")
            data = yf.download(
                tickers, start=start, end=TODAY,
                auto_adjust=True, progress=False, threads=False, session=session
            )['Open']
            if data.empty:
                raise ValueError("Download returned empty DataFrame (likely rate limited).")
            
            opening_prices = data.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else None)
            missing = [t for t, p in opening_prices.items() if p is None]
            if missing:
                print(f"  [WARN] No opening price found for {missing}, extending start by 1 days...")
                start = start - timedelta(days=1)
                # This could result in grabbing opening prices BEFORE the recommendation ran. Giving the
                # VP false price values to work off of.
                continue

            return opening_prices

        except Exception as e:
            if attempt < 4:
                print(f"  [WARN] Download failed ({e}). Retrying in {wait}s...")
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
        total_w = sum(weights_by_strategy[strategy].values())
        normalised_weight = weight / total_w
        market_value = opening_vp_value[strategy] * normalised_weight
        virtual_portfolio.setdefault(strategy, {})[ticker] = {
            "weight": normalised_weight,
            "shares": market_value / opening_prices[ticker],
            "price": opening_prices[ticker],
            "market_value": market_value,
        }
    return virtual_portfolio


def _load_or_create_vp(conn, last_rec, opening_date, last_vp_date, today, file):
    """
    Loads today's virtual portfolio from the DB if already written, otherwise
    builds it from last run's recommendations and opening prices, then writes it.

    Args:
        conn: Active database connection
        last_rec: List of (ticker, weight, strategy) tuples
        opening_date: datetime — date trades are executed at open
        last_vp_date: Date string of the most recent VP snapshot in the DB
        today: Today's date string (YYYY-MM-DD)
        file: File object to write warnings to

    Returns:
        {strategy: {ticker: {weight, shares, price, market_value}}}
    """
    today_str = today if isinstance(today, str) else today.isoformat()
    opening_date_str = opening_date.strftime("%Y-%m-%d") if isinstance(opening_date, datetime) else opening_date

    if conn.execute("SELECT COUNT(*) FROM virtual_portfolio WHERE date = ?", (opening_date_str,)).fetchone()[0] > 0:
        print(f"  [WARN] Virtual portfolio already written for {opening_date_str}. Loading from DB.", file=file)
        rows = conn.execute("SELECT ticker, strategy, weight, price, quantity, market_value FROM virtual_portfolio WHERE date = ?",
                            (opening_date_str,)).fetchall()
        virtual_portfolio = {}
        for ticker, strategy, weight, price, shares, market_value in rows:
            virtual_portfolio.setdefault(strategy, {})[ticker] = {
                "weight": weight, "shares": shares, "price": price, "market_value": market_value,
            }
        return virtual_portfolio

    tickers = {ticker for ticker, _, _ in last_rec}
    opening_prices = _fetch_opening_prices(tickers, opening_date)

    opening_vp_value = {}
    if last_vp_date is None:
        # First VP run — seed each strategy with the real portfolio's total value
        rp_total = conn.execute(
            "SELECT SUM(market_value) FROM portfolios WHERE date = ?", (today_str,)
        ).fetchone()[0] or 0.0
        strategies = {strategy for _, _, strategy in last_rec}
        for strategy in strategies:
            opening_vp_value[strategy] = rp_total
    else:
        last_vp = conn.execute("SELECT ticker, price, quantity, strategy FROM virtual_portfolio WHERE date = ?", (last_vp_date,)).fetchall()
        for ticker, price, shares, strategy in last_vp:
            curr = opening_prices.get(ticker)
            if curr and abs(curr - price) > 0.25 * price:
                print(f"  [WARN] {ticker} moved >25% since last VP entry.", file=file)
            opening_vp_value[strategy] = opening_vp_value.get(strategy, 0) + shares * opening_prices[ticker]

    virtual_portfolio = _build_virtual_portfolio(last_rec, opening_vp_value, opening_prices)

    for strategy, positions in virtual_portfolio.items():
        computed = sum(p["market_value"] for p in positions.values())
        if abs(computed - opening_vp_value[strategy]) > 0.01:
            print(f"  [WARN] {strategy} VP value mismatch: computed={computed:.2f}, expected={opening_vp_value[strategy]:.2f}", file=file)

    insert_virtual_portfolio(conn, opening_date, virtual_portfolio)
    return virtual_portfolio


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
    return_pct_divergence = {s: dollar_divergence[s] / curr_rp_value for s in dollar_divergence}

    print(f"\n{'='*70}", file=file)
    print(f"  PORTFOLIO DIVERGENCE SUMMARY — {today}", file=file)
    print(f"{'='*70}", file=file)

    print(f"\n  CURRENT REAL PORTFOLIO VALUE — ${curr_rp_value:,.2f}", file=file)
    rp_ret = f"{real_pct_change:>+9.2%}" if real_pct_change is not None else f"{'N/A':>9}"
    print(f"\n  CURRENT REAL PORTFOLIO RETURN — {rp_ret}", file=file)
    print(f"\n  {'Strategy':<15} {'Virtual Ret':>13} {'$ Diverg':>12} {'% Diverg':>12} {'$ Value':>12}", file=file)
    print(f"  {'-'*68}", file=file)
    for s in curr_vp_values:
        vp_ret = f"{virtual_pct_change[s]:>+13.2%}" if virtual_pct_change[s] is not None else f"{'N/A':>12}"
        print(f"  {s:<15} {vp_ret}  {dollar_divergence[s]:>+12.2f} {return_pct_divergence[s]:>+12.2%} {curr_vp_values[s]:>12,.2f}", file=file)

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
            curr_price = float(raw_prices[ticker].dropna().iloc[-1])
            ret = (curr_price - pos["price"]) / pos["price"]
            virt_w = pos["weight"]
            real_w = real_weights.get(ticker, 0.0)
            contributions.append((ticker, virt_w, real_w, ret, (virt_w - real_w) * ret))

        for ticker, virt_w, real_w, ret, contrib in sorted(contributions, key=lambda x: abs(x[4]), reverse=True)[:10]:
            print(f"  {ticker:<10} {virt_w:>9.1%} {real_w:>9.1%} {virt_w-real_w:>+9.1%} {ret:>+9.2%} {contrib:>+10.2%}", file=file)

    history = conn.execute("""
        SELECT v.date, v.strategy, SUM(v.market_value), p.rp_val
        FROM virtual_portfolio v
        JOIN (SELECT date, SUM(market_value) AS rp_val FROM portfolios GROUP BY date) p
            ON v.date = p.date
        GROUP BY v.date, v.strategy ORDER BY v.date
    """).fetchall()

    print(f"\n  CUMULATIVE DIVERGENCE HISTORY", file=file)
    print(f"  {'Date':<12} {'Strategy':<15} {'VP Value':>12} {'RP Value':>12} {'Divergence':>12}", file=file)
    for date, strategy, vp_val, rp_val in history:
        print(f"  {date:<12} {strategy:<15} {vp_val:>12.2f} {rp_val:>12.2f} {vp_val - rp_val:>+12.2f}", file=file)

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
        None. Writes VP snapshot to DB and prints divergence summary to file.
    """
    if conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0] == 0:
        print("  [WARNING] No recommendations found. First run.", file=file)
        return

    dates = _validate_dates(conn, today, file)
    if dates is None:
        return
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
    last_rp_value = conn.execute("SELECT SUM(market_value) FROM portfolios WHERE date = ?", (last_rec_date,)).fetchone()[0]

    virtual_portfolio = _load_or_create_vp(conn, last_rec, opening_date, last_vp_date, today, file)

    curr_rp_value = conn.execute("SELECT SUM(market_value) FROM portfolios WHERE date = ?", (today,)).fetchone()[0]
    curr_vp_values = {
        s: sum(pos["shares"] * float(raw_prices[t].dropna().iloc[-1])
               for t, pos in tickers.items() if t in raw_prices.columns)
        for s, tickers in virtual_portfolio.items()
    }
    real_weights = dict(conn.execute("SELECT ticker, weight FROM portfolios WHERE date = ?", (today,)).fetchall())

    _print_divergence_summary(virtual_portfolio, curr_vp_values, last_vp_values,
                               curr_rp_value, last_rp_value, real_weights,
                               raw_prices, conn, today, file)