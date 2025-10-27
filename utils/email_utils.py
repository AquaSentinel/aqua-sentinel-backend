# email_utils.py
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from dotenv import load_dotenv
import os

load_dotenv()

MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_TO   = os.getenv("MAIL_TO")
MAIL_HOST = os.getenv("MAIL_HOST", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "465"))
MAIL_USER = os.getenv("MAIL_USER")
MAIL_PASS = os.getenv("MAIL_PASS")

def send_mail_with_attachment(
    subject: str,
    html_body: str,
    attachment_name: str = None,
    attachment_bytes: bytes = None,
    to_addr: str = None
):
    """Send an HTML email (optionally with attachment)."""
    # Validate SMTP config
    pairs = [("MAIL_FROM", MAIL_FROM), ("MAIL_USER", MAIL_USER), ("MAIL_PASS", MAIL_PASS)]
    missing = [name for name, val in pairs if not val]
    if missing:
        raise RuntimeError(f"Missing mail configuration: {missing}")

    to_addr = to_addr or MAIL_TO
    if not to_addr:
        raise RuntimeError("No recipient address configured or provided")

    msg = MIMEMultipart()
    msg["From"] = MAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject

    msg.attach(MIMEText(html_body, "html"))

    if attachment_bytes and attachment_name:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
        msg.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(MAIL_HOST, MAIL_PORT, context=context) as server:
        server.login(MAIL_USER, MAIL_PASS)
        server.sendmail(MAIL_FROM, [to_addr], msg.as_string())

    print(f" Email sent to {to_addr} with subject: {subject}")
