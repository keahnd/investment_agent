from database.schema		import init_database
from database.reconciler	import reconcile_virtual_portfolio, build_divergence_data

conn = init_database(user_path)
try:
	with open(vp_output, 'w', encoding='utf-8') as f:
		reconcile_virtual_portfolio(conn, today, f)
except Exception as e:
	print(f"  [warn] Reconciler failed: {e}")

# Step 5 — build divergence data for report
try:
	divergence_data = build_divergence_data(conn, today)
except Exception as e:
	print(f"  [warn] Divergence data build failed: {e}")
	divergence_data = None