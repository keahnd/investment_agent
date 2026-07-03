"""
reports/email.py
=================
Handles email delivery of the weekly portfolio report PDF.
Uses SendGrid API for reliable delivery.

Two functions:
  send_report_email()  — weekly report to user
  send_error_email()   — pipeline failure alert to operator
"""

import logging
import os
import base64
from pathlib import Path
from datetime import date

from sendgrid import SendGridAPIClient

logger = logging.getLogger("investment_agent")
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition
)


def _get_api_client() -> SendGridAPIClient:
    api_key = os.environ.get("SENDGRID_API_KEY")
    if not api_key:
        raise ValueError("SENDGRID_API_KEY not set in environment")
    return SendGridAPIClient(api_key)


def _encode_pdf(pdf_path: str) -> str:
    """Base64 encodes a PDF file for SendGrid attachment."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    

def _build_subject(final_state: dict) -> str:
    """
    Builds the email subject line.
    Format: Portfolio Report — 2025-04-28 — BUY GOOGL, SELL MSFT
    Includes the top two consensus actions by delta magnitude.
    """
    run_date  = final_state.get("run_date", str(date.today()))
    rec_table = final_state.get("recommendation_table") or []

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
    subject    = _build_subject(final_state)
    run_date   = final_state.get("run_date", str(date.today()))
    user_name  = final_state.get("user_name", "").replace("_", " ").title()
    total_val  = final_state.get("total_portfolio_value", 0)

    # Plain text body — brief summary before the PDF
    rec_table  = final_state.get("recommendation_table") or []
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
    message = Mail(
        from_email    = sender,
        to_emails     = user_email,
        subject       = subject,
        plain_text_content = body,
    )

    # Attach PDF
    pdf_name    = Path(pdf_path).name
    encoded_pdf = _encode_pdf(pdf_path)

    attachment = Attachment(
        FileContent(encoded_pdf),
        FileName(pdf_name),
        FileType("application/pdf"),
        Disposition("attachment"),
    )
    message.attachment = attachment

    try:
        client   = _get_api_client()
        response = client.send(message)

        if response.status_code in (200, 202):
            logger.info(f"[Email] Report sent to {user_email} — status {response.status_code}")
            return True
        else:
            logger.warning(f"[Email] Unexpected status {response.status_code} sending to {user_email}")
            return False

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

    message = Mail(
        from_email         = sender,
        to_emails          = operator,
        subject            = subject,
        plain_text_content = body,
    )

    try:
        client   = _get_api_client()
        response = client.send(message)

        if response.status_code in (200, 202):
            logger.info(f"[Email] Error alert sent to operator — status {response.status_code}")
            return True
        else:
            logger.warning(f"[Email] Failed to send error alert — status {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"[Email] Error sending failure alert: {e}")
        return False