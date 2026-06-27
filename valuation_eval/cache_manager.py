# class SectorCacheManager:
#     """
#     Manages cache invalidation with both time-based TTL and
#     market-move event triggers. Stores SPY price at last fetch
#     alongside the sector data so it can detect regime changes.
#     """

#     def __init__(self, db_path: str):
#         self.db_path = db_path
#         self._ensure_schema()

#     def _ensure_schema(self):
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("""
#                 CREATE TABLE IF NOT EXISTS sector_cache (
#                     cache_key        TEXT PRIMARY KEY,
#                     sector           TEXT,
#                     metric           TEXT,
#                     value_json       TEXT,
#                     fetched_at       TEXT,
#                     spy_at_fetch     REAL
#                 )
#             """)

#     def is_stale(self, cache_key: str, metric: str) -> tuple[bool, str]:
#         """
#         Returns (is_stale, reason).
#         Checks time TTL first, then market move if configured.
#         """
#         config = CACHE_CONFIG.get(metric, {"ttl_days": 30, "market_move_pct": None})

#         with sqlite3.connect(self.db_path) as conn:
#             row = conn.execute(
#                 "SELECT fetched_at, spy_at_fetch FROM sector_cache WHERE cache_key = ?",
#                 (cache_key,)
#             ).fetchone()

#         if not row:
#             return True, "no cached value"

#         fetched_at, spy_at_fetch = row
#         age_days = (datetime.utcnow() - datetime.fromisoformat(fetched_at)).days

#         # Time-based TTL
#         if age_days > config["ttl_days"]:
#             return True, f"TTL expired ({age_days} days old, limit {config['ttl_days']})"

#         # Market move trigger
#         market_move_pct = config.get("market_move_pct")
#         if market_move_pct and spy_at_fetch:
#             spy_now = self._get_spy_price()
#             if spy_now:
#                 move = abs(spy_now / spy_at_fetch - 1) * 100
#                 if move > market_move_pct:
#                     return True, (
#                         f"market moved {move:.1f}% since last fetch "
#                         f"(SPY {spy_at_fetch:.0f} → {spy_now:.0f}), "
#                         f"threshold {market_move_pct:.0f}%"
#                     )

#         return False, "cache fresh"

#     def get(self, cache_key: str) -> Optional[dict]:
#         with sqlite3.connect(self.db_path) as conn:
#             row = conn.execute(
#                 "SELECT value_json FROM sector_cache WHERE cache_key = ?",
#                 (cache_key,)
#             ).fetchone()
#         return json.loads(row[0]) if row else None

#     def set(self, cache_key: str, sector: str, metric: str, value: dict):
#         spy_price = self._get_spy_price()
#         with sqlite3.connect(self.db_path) as conn:
#             conn.execute("""
#                 INSERT OR REPLACE INTO sector_cache
#                     (cache_key, sector, metric, value_json, fetched_at, spy_at_fetch)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             """, (
#                 cache_key,
#                 sector,
#                 metric,
#                 json.dumps(value),
#                 datetime.utcnow().isoformat(),
#                 spy_price,
#             ))

#     def _get_spy_price(self) -> Optional[float]:
#         try:
#             price = yf.Ticker("SPY").info.get("regularMarketPrice")
#             return float(price) if price else None
#         except Exception:
#             return None
        

# def get_sector_pe(sector: str, db_path: str) -> Optional[dict]:
#     cache_key = f"peer_pe:{sector}"
#     manager   = SectorCacheManager(db_path)

#     stale, reason = manager.is_stale(cache_key, "peer_pe")
#     if stale:
#         logger.info(f"Refreshing sector PE for {sector}: {reason}")
#         data = fetch_sector_metrics_from_peers(sector)   # existing function
#         manager.set(cache_key, sector, "peer_pe", data)
#     else:
#         data = manager.get(cache_key)

#     return data


# def get_sector_ev_ebitda(sector: str, db_path: str) -> Optional[float]:
#     cache_key = f"peer_ev_ebitda:{sector}"
#     manager   = SectorCacheManager(db_path)

#     stale, reason = manager.is_stale(cache_key, "peer_ev_ebitda")
#     if stale:
#         logger.info(f"Refreshing sector EV/EBITDA for {sector}: {reason}")
#         data = fetch_peer_ev_ebitda_median(sector)
#         manager.set(cache_key, sector, "peer_ev_ebitda", {"median": data})
#     else:
#         cached = manager.get(cache_key)
#         data   = cached["median"] if cached else None

#     return data


# def get_damodaran_table(db_path: str) -> pd.DataFrame:
#     cache_key = "damodaran:pe_table"
#     manager   = SectorCacheManager(db_path)

#     stale, reason = manager.is_stale(cache_key, "damodaran")
#     if stale:
#         logger.info(f"Refreshing Damodaran table: {reason}")
#         df = fetch_damodaran_pe_table()
#         if not df.empty:
#             manager.set(cache_key, "global", "damodaran",
#                        {"rows": df.to_dict(orient="records")})
#         return df
#     else:
#         cached = manager.get(cache_key)
#         return pd.DataFrame(cached["rows"]) if cached else pd.DataFrame()