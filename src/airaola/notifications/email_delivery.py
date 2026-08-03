from __future__ import annotations

import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Mapping

from airaola.reporting.weekly_report import (
    WeeklyReport,
)


DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_TIMEOUT_SECONDS = 30.0

ENV_SENDER_EMAIL = "AIRAOLA_EMAIL_SENDER"
ENV_RECIPIENT_EMAIL = "AIRAOLA_EMAIL_RECIPIENT"
ENV_APP_PASSWORD = "AIRAOLA_EMAIL_APP_PASSWORD"
ENV_SMTP_HOST = "AIRAOLA_EMAIL_SMTP_HOST"
ENV_SMTP_PORT = "AIRAOLA_EMAIL_SMTP_PORT"


@dataclass(frozen=True)
class EmailConfiguration:
    """Store validated SMTP delivery configuration."""

    sender_email: str
    recipient_email: str
    app_password: str

    smtp_host: str = DEFAULT_SMTP_HOST
    smtp_port: int = DEFAULT_SMTP_PORT
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class EmailDeliveryResult:
    """Describe the outcome of one email delivery attempt."""

    sent: bool
    recipient_email: str
    subject: str

    message_id: str | None = None
    error_message: str | None = None


def _normalise_email(
    value: str,
    field_name: str,
) -> str:
    """Validate one basic email-address value."""

    email_address = str(
        value
    ).strip()

    if not email_address:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    if (
        "@" not in email_address
        or email_address.startswith("@")
        or email_address.endswith("@")
    ):
        raise ValueError(
            f"{field_name} must be a valid email address."
        )

    return email_address


def _normalise_password(
    value: str,
) -> str:
    """Normalise an SMTP app password."""

    password = (
        str(value)
        .replace(" ", "")
        .strip()
    )

    if not password:
        raise ValueError(
            "Email app password cannot be empty."
        )

    return password


def _parse_port(
    value: str | int,
) -> int:
    """Validate an SMTP port."""

    try:
        port = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "SMTP port must be an integer."
        ) from error

    if not 1 <= port <= 65535:
        raise ValueError(
            "SMTP port must be between 1 and 65535."
        )

    return port


def load_email_configuration(
    environment: Mapping[str, str] | None = None,
) -> EmailConfiguration:
    """Load SMTP settings from environment variables."""

    values = (
        os.environ
        if environment is None
        else environment
    )

    missing_variables = [
        variable_name
        for variable_name in (
            ENV_SENDER_EMAIL,
            ENV_RECIPIENT_EMAIL,
            ENV_APP_PASSWORD,
        )
        if not str(
            values.get(
                variable_name,
                "",
            )
        ).strip()
    ]

    if missing_variables:
        raise ValueError(
            "Email configuration is missing environment variables: "
            + ", ".join(
                missing_variables
            )
        )

    smtp_host = str(
        values.get(
            ENV_SMTP_HOST,
            DEFAULT_SMTP_HOST,
        )
    ).strip()

    if not smtp_host:
        raise ValueError(
            "SMTP host cannot be empty."
        )

    return EmailConfiguration(
        sender_email=_normalise_email(
            values[ENV_SENDER_EMAIL],
            ENV_SENDER_EMAIL,
        ),
        recipient_email=_normalise_email(
            values[ENV_RECIPIENT_EMAIL],
            ENV_RECIPIENT_EMAIL,
        ),
        app_password=_normalise_password(
            values[ENV_APP_PASSWORD]
        ),
        smtp_host=smtp_host,
        smtp_port=_parse_port(
            values.get(
                ENV_SMTP_PORT,
                DEFAULT_SMTP_PORT,
            )
        ),
    )


def build_report_email(
    report: WeeklyReport,
    configuration: EmailConfiguration,
) -> EmailMessage:
    """Build a multipart text and HTML decision-report email."""

    if not report.text_content.strip():
        raise ValueError(
            "Weekly report text content cannot be empty."
        )

    if not report.html_content.strip():
        raise ValueError(
            "Weekly report HTML content cannot be empty."
        )

    subject = (
        "Project Airaola | "
        f"Gameweek {report.gameweek} Decision Report"
    )

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = configuration.sender_email
    message["To"] = configuration.recipient_email

    message.set_content(
        report.text_content
    )

    message.add_alternative(
        report.html_content,
        subtype="html",
    )

    return message


def attach_report_files(
    message: EmailMessage,
    report: WeeklyReport,
) -> None:
    """Attach saved text and HTML report files when available."""

    attachments = (
        (
            report.text_path,
            "plain",
        ),
        (
            report.html_path,
            "html",
        ),
    )

    for file_path, subtype in attachments:
        if file_path is None:
            continue

        path = Path(
            file_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Report attachment not found: {path}"
            )

        message.add_attachment(
            path.read_bytes(),
            maintype="text",
            subtype=subtype,
            filename=path.name,
        )


def send_report_email(
    report: WeeklyReport,
    configuration: EmailConfiguration,
    attach_files: bool = True,
) -> EmailDeliveryResult:
    """Send one weekly report through authenticated SMTP."""

    message = build_report_email(
        report=report,
        configuration=configuration,
    )

    if attach_files:
        attach_report_files(
            message=message,
            report=report,
        )

    subject = str(
        message["Subject"]
    )

    tls_context = ssl.create_default_context()

    try:
        with smtplib.SMTP(
            host=configuration.smtp_host,
            port=configuration.smtp_port,
            timeout=configuration.timeout_seconds,
        ) as smtp:
            smtp.ehlo()
            smtp.starttls(
                context=tls_context
            )
            smtp.ehlo()

            smtp.login(
                configuration.sender_email,
                configuration.app_password,
            )

            refused_recipients = smtp.send_message(
                message
            )

    except (
        smtplib.SMTPException,
        OSError,
    ) as error:
        return EmailDeliveryResult(
            sent=False,
            recipient_email=(
                configuration.recipient_email
            ),
            subject=subject,
            error_message=str(
                error
            ),
        )

    if refused_recipients:
        return EmailDeliveryResult(
            sent=False,
            recipient_email=(
                configuration.recipient_email
            ),
            subject=subject,
            error_message=(
                "SMTP server refused one or more recipients: "
                + ", ".join(
                    str(recipient)
                    for recipient in refused_recipients
                )
            ),
        )

    return EmailDeliveryResult(
        sent=True,
        recipient_email=(
            configuration.recipient_email
        ),
        subject=subject,
        message_id=message.get(
            "Message-ID"
        ),
    )