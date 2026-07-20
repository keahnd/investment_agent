"""
reports/email.py
=================
Handles email delivery of the weekly portfolio report PDF.
Uses Gmail SMTP (smtplib) for delivery.

Two functions:
  send_report_email()  — weekly report to user
  send_error_email()   — pipeline failure alert to operator
"""

import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from datetime import date

logger = logging.getLogger("investment_agent")

SERVER = "smtp.gmail.com"
PORT = 465


def _send(message: EmailMessage) -> None:
	sender = os.environ.get("SENDER_EMAIL")
	app_password = os.environ.get("GMAIL_APP_PASS")

	with smtplib.SMTP_SSL(SERVER, PORT) as smtp:
		smtp.login(sender, app_password)
		smtp.send_message(message)


def _build_subject(rec_table: list, run_date: str) -> str:
	"""
	Builds the email subject line.
	Format: Portfolio Report — 2025-04-28 — BUY GOOGL, SELL MSFT
	Includes the top two consensus actions by delta magnitude.
	"""
	# Find the highest conviction actions by delta magnitude
	# Use max_sharpe delta as the primary signal
	actions = []
	for row in rec_table:
		consensus = row.get("consensus_action", "HOLD")
		if consensus in ("BUY", "SELL"):
			# Use the average absolute delta across strategies as conviction measure
			strategies = [k.replace("_weight", "") for k in row.keys()
						 if k.endswith("_weight") and not k == "current_weight"]
			deltas = [abs(row.get(f"{s}_delta", 0) or 0) for s in strategies]
			avg_delta = sum(deltas) / len(deltas) if deltas else 0
			actions.append((consensus, row["ticker"], avg_delta))

	# Sort by conviction magnitude
	actions.sort(key=lambda x: x[2], reverse=True)

	if not actions:
		action_str = "No changes recommended"
	elif len(actions) == 1:
		action_str = f"{actions[0][0]} {actions[0][1]}"
	else:
		action_str = ", ".join(f"{a[0]} {a[1]}" for a in actions[:2])

	return f"Portfolio Report — {run_date} — {action_str}"


def send_report_email(
	final_state: dict,
	pdf_path:    str,
	user_email:  str,
) -> bool:
	"""
	Sends the weekly portfolio report PDF to the user.

	Args:
		final_state: Complete pipeline state for subject line construction
		pdf_path:    Path to the generated PDF file
		user_email:  Recipient email from user config.json

	Returns:
		True if sent successfully, False otherwise
	"""
	sender     = os.environ.get("SENDER_EMAIL")
	run_date   = final_state.get("run_date", str(date.today()))
	rec_table  = final_state.get("recommendation_table") or []
	user_name  = final_state.get("user_name", "").replace("_", " ").title()
	total_val  = final_state.get("total_portfolio_value", 0)

	subject    = _build_subject(rec_table, run_date)

	# Plain text body — brief summary before the PDF
	buy_tickers  = [r["ticker"] for r in rec_table if r.get("consensus_action") == "BUY"]
	sell_tickers = [r["ticker"] for r in rec_table if r.get("consensus_action") == "SELL"]
	hold_tickers = [r["ticker"] for r in rec_table if r.get("consensus_action") == "HOLD"]

	body = f"""Hi {user_name},

	Your weekly portfolio report for {run_date} is attached.

	Current Portfolio Value: ${total_val:,.2f} CAD

	Consensus Recommendations:
	BUY:  {', '.join(buy_tickers)  if buy_tickers  else 'None'}
	SELL: {', '.join(sell_tickers) if sell_tickers else 'None'}
	HOLD: {', '.join(hold_tickers) if hold_tickers else 'None'}

	Full analysis, simulation results, and commentary are in the attached PDF.

	---
	This report is generated automatically.
	Not financial advice.
	"""

	# Build message
	message = EmailMessage()
	message["From"]    = sender
	message["To"]      = user_email
	message["Subject"] = subject
	message.set_content(body)

	# Attach PDF
	pdf_name = Path(pdf_path).name
	with open(pdf_path, "rb") as f:
		pdf_bytes = f.read()
	message.add_attachment(
		pdf_bytes,
		maintype="application",
		subtype="pdf",
		filename=pdf_name,
	)

	try:
		_send(message)
		logger.info(f"[Email] Report sent to {user_email}")
		return True

	except Exception as e:
		logger.error(f"[Email] Failed to send report to {user_email}: {e}")
		return False


def send_error_email(
	user_name:     str,
	error_message: str,
	traceback_str: str = "",
) -> bool:
	"""
	Sends a pipeline failure alert to the operator email.
	Called when any user's pipeline run raises an unhandled exception.

	Args:
		user_name:     Name of the user whose pipeline failed
		error_message: The exception message
		traceback_str: Full traceback as string for debugging

	Returns:
		True if sent successfully, False otherwise
	"""
	sender   = os.environ.get("SENDER_EMAIL")
	operator = os.environ.get("OPERATOR_EMAIL")

	if not operator:
		logger.warning("[Email] OPERATOR_EMAIL not set — cannot send error alert")
		return False

	subject = f"[PIPELINE ERROR] Failed for {user_name} — {date.today()}"

	body = f"""
Pipeline failure alert.

User:  {user_name}
Date:  {date.today()}
Error: {error_message}

Traceback:
{traceback_str if traceback_str else 'Not available'}

Check the pipeline logs for more detail.
	"""

	message = EmailMessage()
	message["From"]    = sender
	message["To"]      = operator
	message["Subject"] = subject
	message.set_content(body)

	try:
		_send(message)
		logger.info("[Email] Error alert sent to operator")
		return True

	except Exception as e:
		logger.error(f"[Email] Error sending failure alert: {e}")
		return False
